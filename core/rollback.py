"""
Agent 回退系统：持久化 checkpoint + workspace 快照 + rollback 指令。

三层设计：
1. SqliteCheckpointer — 替代 MemorySaver，checkpoint 持久化到 SQLite，重启可恢复
2. WorkspaceSnapshot — 每轮实验前自动 tar.gz 快照，保留最近 N 个
3. Rollback 指令 — HUMAN_DIRECTIVE.md 中写 "rollback" 触发回退

面试价值：Agent 容错不只是 try-catch，是 checkpoint + snapshot + rollback 三位一体。
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import sqlite3
import tarfile
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger("autoresearcher.rollback")

# LangGraph 默认的序列化格式
_SERDE = JsonPlusSerializer()


# ═══════════════════════════════════════════════════════════════════
# 1. SqliteCheckpointer — 持久化 checkpoint
# ═══════════════════════════════════════════════════════════════════

class SqliteCheckpointer(BaseCheckpointSaver, AbstractContextManager):
    """SQLite 持久化的 LangGraph Checkpointer，替代 MemorySaver。

    每个 checkpoint 序列化为 JSON 存入 SQLite，
    重启后自动恢复最近一次 graph 状态。
    """

    def __init__(self, db_path: Path):
        super().__init__(serde=_SERDE)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            # WAL:并发读(dashboard 每轮读同一库)不阻塞写,减少 Windows
            # 上的 "database is locked";busy_timeout 兜底瞬时竞争
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id        TEXT NOT NULL,
                    checkpoint_ns    TEXT NOT NULL DEFAULT '',
                    checkpoint_id    TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    checkpoint_type  TEXT NOT NULL DEFAULT 'msgpack',
                    checkpoint_data  BLOB NOT NULL,
                    metadata_json    TEXT NOT NULL DEFAULT '{}',
                    created_at       REAL NOT NULL,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoint_writes (
                    thread_id       TEXT NOT NULL,
                    checkpoint_ns   TEXT NOT NULL DEFAULT '',
                    checkpoint_id   TEXT NOT NULL,
                    task_id         TEXT NOT NULL,
                    task_path       TEXT NOT NULL,
                    channel         TEXT NOT NULL,
                    value_type      TEXT NOT NULL DEFAULT 'msgpack',
                    value_data      BLOB NOT NULL,
                    task_ts         REAL NOT NULL,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cp_thread "
                "ON checkpoints(thread_id, checkpoint_ns, checkpoint_id DESC)"
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Core API (override BaseCheckpointSaver)
    # ------------------------------------------------------------------

    def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        """读取指定 config 的最新 checkpoint。"""
        thread_id = config.get("configurable", {}).get("thread_id", "")
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

        with sqlite3.connect(str(self.db_path)) as conn:
            if checkpoint_id:
                row = conn.execute(
                    "SELECT checkpoint_id, parent_checkpoint_id, checkpoint_type, "
                    "checkpoint_data, metadata_json "
                    "FROM checkpoints WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?",
                    (thread_id, checkpoint_ns, checkpoint_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT checkpoint_id, parent_checkpoint_id, checkpoint_type, "
                    "checkpoint_data, metadata_json "
                    "FROM checkpoints WHERE thread_id=? AND checkpoint_ns=? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (thread_id, checkpoint_ns),
                ).fetchone()

        if not row:
            return None

        cid, parent_cid, cp_type, cp_data, meta_json = row
        checkpoint = self.serde.loads_typed((cp_type, cp_data))
        metadata = json.loads(meta_json) if meta_json else {}

        # 读取 pending writes
        pending_writes = self._load_writes(thread_id, checkpoint_ns, cid)

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": cid,
                },
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_cid,
                },
            } if parent_cid else None,
            pending_writes=pending_writes if pending_writes else None,
        )

    def put(self, config: dict, checkpoint, metadata: dict, new_versions: dict) -> dict:
        """写入 checkpoint。返回新的 config。"""
        thread_id = config.get("configurable", {}).get("thread_id", "")
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")

        cp_type, cp_data = self.serde.dumps_typed(checkpoint)
        meta_json = json.dumps(metadata, ensure_ascii=False)
        checkpoint_id = checkpoint.get("id", str(time.time()))
        parent_checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO checkpoints "
                "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, "
                " checkpoint_type, checkpoint_data, metadata_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                 cp_type, cp_data, meta_json, time.time()),
            )
            conn.commit()

        logger.debug("Checkpoint saved: %s/%s/%s", thread_id, checkpoint_ns, checkpoint_id)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            },
        }

    def put_writes(self, config: dict, writes, task_id: str, task_path: str = "") -> None:
        """写入 pending writes（interrupt 时暂存）。"""
        if not writes:
            return

        thread_id = config.get("configurable", {}).get("thread_id", "")
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id", "")

        with sqlite3.connect(str(self.db_path)) as conn:
            for idx, (channel, value) in enumerate(writes):
                v_type, v_data = self.serde.dumps_typed(value)
                conn.execute(
                    "INSERT OR REPLACE INTO checkpoint_writes "
                    "(thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, "
                    " channel, value_type, value_data, task_ts) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (thread_id, checkpoint_ns, checkpoint_id, task_id,
                     task_path if task_path else str(idx),
                     channel, v_type, v_data, time.time()),
                )
            conn.commit()

    def list(self, config: Optional[dict] = None, *, filter=None, before=None, limit=10) -> list[CheckpointTuple]:
        """列出 checkpoints（最新在前）。"""
        thread_id = config.get("configurable", {}).get("thread_id", "") if config else ""
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "") if config else ""

        query = (
            "SELECT checkpoint_id, parent_checkpoint_id, checkpoint_type, "
            "checkpoint_data, metadata_json "
            "FROM checkpoints WHERE thread_id=? AND checkpoint_ns=? "
            "ORDER BY created_at DESC LIMIT ?"
        )
        results = []
        with sqlite3.connect(str(self.db_path)) as conn:
            for row in conn.execute(query, (thread_id, checkpoint_ns, limit)):
                cid, parent_cid, cp_type, cp_data, meta_json = row
                checkpoint = self.serde.loads_typed((cp_type, cp_data))
                metadata = json.loads(meta_json) if meta_json else {}
                results.append(CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": cid,
                        },
                    },
                    checkpoint=checkpoint,
                    metadata=metadata,
                    parent_config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_cid,
                        },
                    } if parent_cid else None,
                ))
        return results

    def delete_thread(self, thread_id: str) -> None:
        """删除某个 thread 的所有 checkpoint。"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM checkpoints WHERE thread_id=?", (thread_id,))
            conn.execute("DELETE FROM checkpoint_writes WHERE thread_id=?", (thread_id,))
            conn.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_writes(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> list:
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT channel, value_type, value_data FROM checkpoint_writes "
                "WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=? "
                "ORDER BY task_path",
                (thread_id, checkpoint_ns, checkpoint_id),
            ).fetchall()
        return [
            (channel, self.serde.loads_typed((v_type, v_data)))
            for channel, v_type, v_data in rows
        ]

    def get_checkpoint_count(self, thread_id: str = "") -> int:
        """获取 checkpoint 数量（用于监控）。"""
        with sqlite3.connect(str(self.db_path)) as conn:
            if thread_id:
                return conn.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE thread_id=?", (thread_id,)
                ).fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]

    def prune_old(self, keep: int = 50) -> int:
        """删除旧 checkpoint，只保留最新 N 个。返回删除数。"""
        with sqlite3.connect(str(self.db_path)) as conn:
            # 获取要删除的 checkpoint 数量
            total = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
            if total <= keep:
                return 0

            # 删除旧记录
            cursor = conn.execute(
                "DELETE FROM checkpoints WHERE rowid NOT IN ("
                "  SELECT rowid FROM checkpoints ORDER BY created_at DESC LIMIT ?"
                ")", (keep,)
            )
            deleted = cursor.rowcount
            conn.commit()
            logger.info("Pruned %d old checkpoints, kept %d", deleted, keep)
            return deleted

    def __exit__(self, *args):
        pass  # SQLite 自动管理连接


# ═══════════════════════════════════════════════════════════════════
# 2. WorkspaceSnapshot — 每轮实验前快照
# ═══════════════════════════════════════════════════════════════════

class WorkspaceSnapshot:
    """工作区快照管理器。

    策略（兼顾体积和续训能力）：
    - 小文件（代码/配置/日志）→ 打入 tar.gz
    - 大文件（模型权重 .pth/.ckpt）→ 不打入，但记录到 manifest.json
    - rollback 时先恢复代码，再检查模型文件是否还在 → 在则续训，不在则重头来
    """

    # 打入快照时跳过的模式（大文件/临时文件/SQLite 瞬态文件）
    # -wal/-shm/-journal 是 SQLite 运行期瞬态文件:① Windows 上偶发
    #   读共享冲突(冒烟实测 cycle0 快照 Errno 13,非致命但丢归档);
    #   ② 打进快照会让 .db 与不同时刻的 WAL 错配,restore 反而更危险。
    #   跳过它们,SqliteCheckpointer 恢复时会自动重建 WAL。
    SKIP_IN_ARCHIVE = {
        "*.tar.gz", "__pycache__", "*.pyc", ".git",
        "costs.jsonl", "audit.jsonl", "bad_cases.jsonl",
        "*.db-wal", "*.db-shm", "*.db-journal",
        # 数据集/静态资产不入快照:快照保存代码与权重,数据集是本地静态
        # 资产(重下/重拷成本高,实验不修改它)。T6 实测:60k PNG 的
        # CIFAR-10 让快照 tar 10+ 分钟,阻塞整个 cycle。
        "data", "CIFAR-10", "*.png", "*.jpg", "*.jpeg", "*.gif",
    }

    # 不打包但记录到 manifest 的模型文件
    MODEL_PATTERNS = {"*.pth", "*.pt", "*.ckpt", "*.safetensors", "*.bin", "*.h5", "*.keras"}

    def __init__(self, workspace: Path, snapshots_dir: Optional[Path] = None,
                 keep: int = 10):
        self.workspace = workspace
        self.snapshots_dir = snapshots_dir or (workspace / ".snapshots")
        self.keep = keep
        self._manifest_path = self.workspace / ".rollback_manifest.json"

    def create(self, label: str = "", cycle: int = 0) -> Path:
        """创建快照。

        1. 打包代码/配置等小文件 → tar.gz
        2. 扫描模型文件 → manifest.json（只记路径+hash，不打包）

        返回快照文件路径。
        """
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_label = label.replace(" ", "_").replace("/", "_")[:30] if label else "auto"
        name = f"snap_{ts}_{safe_label}.tar.gz"
        path = self.snapshots_dir / name

        # ── Step 1: 扫描文件（递归）──
        small_files: list[str] = []
        model_files: list[dict] = []

        def _scan_dir(directory: Path, prefix: str = ""):
            for item in sorted(directory.iterdir()):
                rel = f"{prefix}{item.name}" if prefix else item.name
                if item.name.startswith(".") and item.name != ".cycle_counter":
                    continue
                if any(item.match(pat) for pat in self.SKIP_IN_ARCHIVE):
                    continue
                if any(item.match(pat) for pat in self.MODEL_PATTERNS):
                    model_files.append(self._describe_model_file(item))
                elif item.is_dir():
                    _scan_dir(item, f"{rel}/")
                elif item.is_file():
                    small_files.append(rel)

        _scan_dir(self.workspace)

        # ── 权重位置引用：即使 checkpoints/ 此刻为空（execute 前快照），
        #    也记录为「预期模型文件」。restore 时校验它们是否存在，
        #    存在则 can_resume_training=true（权重是训练过程中写入的）。
        ckpt_dir = self.workspace / "checkpoints"
        if ckpt_dir.exists():
            seen = {os.path.relpath(mf["path"], self.workspace).replace("\\", "/")
                    for mf in model_files}
            if not ckpt_dir.is_dir() or ckpt_dir.is_symlink():
                pass  # 不追踪符号链接目录
            else:
                for pat in self.MODEL_PATTERNS:
                    for p in sorted(ckpt_dir.glob(pat)) if ckpt_dir.exists() else []:
                        rel = os.path.relpath(p, self.workspace).replace("\\", "/")
                        if rel not in seen:
                            model_files.append(self._describe_model_file(p))
                            seen.add(rel)
                # 若 checkpoints 目录存在但没有任何匹配模型文件，仍记录占位，
                # 让 restore 能发现「目录在但权重还没生成」
                if not model_files:
                    model_files.append({
                        "path": "checkpoints/*.pth",
                        "size_mb": 0,
                        "md5_partial": "",
                        "expected": True,
                    })

        def _filter(tarinfo: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
            # 二次过滤：跳过不应该出现的模型文件
            for pat in self.MODEL_PATTERNS:
                if tarinfo.name.endswith(pat.replace("*", "")):
                    return None
            return tarinfo

        try:
            with gzip.open(path, "wb") as gz:
                with tarfile.open(fileobj=gz, mode="w") as tar:
                    for item_name in small_files:
                        item = self.workspace / item_name
                        if item.is_file():
                            tar.add(str(item), arcname=item_name, filter=_filter)
            archive_size_kb = path.stat().st_size / 1024
        except Exception as exc:
            logger.warning("Snapshot archive creation failed: %s", exc, exc_info=True)
            path.touch()
            archive_size_kb = 0

        # ── Step 2: 写 manifest（记录模型文件位置）──
        manifest = {
            "snapshot": name,
            "created_at": ts,
            "cycle": cycle,
            "archive_size_kb": round(archive_size_kb, 1),
            "small_files": small_files,
            "model_files": model_files,
        }

        manifest_path = self.snapshots_dir / f"{path.stem}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        total_model_mb = sum(mf.get("size_mb", 0) for mf in model_files)
        logger.info(
            "Snapshot created: %s | code: %.0f KB | models recorded: %d files, %.0f MB",
            name, archive_size_kb, len(model_files), total_model_mb,
        )
        return path

    def _describe_model_file(self, path: Path) -> dict:
        """描述一个模型文件：路径、大小、MD5。"""
        import hashlib

        try:
            stat = path.stat()
            size_mb = stat.st_size / (1024 * 1024)
        except OSError:
            size_mb = 0

        # 只对前 1MB 做 hash（大模型文件全量 hash 太慢）
        try:
            h = hashlib.md5()
            with open(path, "rb") as f:
                h.update(f.read(1024 * 1024))  # 首 1MB
                f.seek(max(0, stat.st_size - 1024 * 1024))  # 尾 1MB
                h.update(f.read(1024 * 1024))
            file_hash = h.hexdigest()
        except Exception:
            file_hash = "unreadable"

        return {
            "path": str(path.relative_to(self.workspace)),
            "size_mb": round(size_mb, 1),
            "md5_partial": file_hash,
        }

    def list_snapshots(self) -> list[Path]:
        """列出所有快照 tar.gz（最新在前）。"""
        if not self.snapshots_dir.exists():
            return []
        return sorted(
            self.snapshots_dir.glob("snap_*.tar.gz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def get_manifest(self, snapshot_path: Path) -> Optional[dict]:
        """读取快照对应的 manifest。"""
        manifest_path = self.snapshots_dir / f"{snapshot_path.stem}.manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        return None

    def restore(self, snapshot_path: Optional[Path] = None) -> dict:
        """
        恢复工作区到指定快照。

        返回恢复结果：
        {
            "success": True/False,
            "snapshot": "snap_20240101_120000_cycle5.tar.gz",
            "files_restored": int,         # 恢复的文件数
            "model_files_found": int,       # 模型文件还在的个数
            "model_files_missing": int,     # 模型文件丢了的个数
            "can_resume_training": bool,    # 是否可以续训
            "message": "描述",
        }
        """
        result = {
            "success": False,
            "snapshot": "",
            "files_restored": 0,
            "model_files_found": 0,
            "model_files_missing": 0,
            "can_resume_training": False,
            "message": "",
        }

        if snapshot_path is None:
            snapshots = self.list_snapshots()
            if not snapshots:
                result["message"] = "No snapshots available for rollback"
                return result
            snapshot_path = snapshots[0]

        if not snapshot_path.exists():
            result["message"] = f"Snapshot not found: {snapshot_path}"
            return result

        result["snapshot"] = snapshot_path.name

        # 恢复前备份当前状态
        pre_rollback_snap = self.create(label="pre_rollback")

        # ── Step 1: 恢复代码文件 ──
        try:
            archive_members: set[str] = set()
            with gzip.open(snapshot_path, "rb") as gz:
                with tarfile.open(fileobj=gz, mode="r") as tar:
                    archive_members = {
                        m.name.rstrip("/") for m in tar.getmembers() if not m.isdir()
                    }
                    tar.extractall(str(self.workspace))
            result["success"] = True

            # ── 清理快照后 agent 新建的文件（避免混合状态）。
            #    只删不在 archive 里的文件；绝不碰 checkpoints/（权重）、
            #    data/（数据集）、.snapshots/、.directive_archive/。
            self._cleanup_non_archive_files(archive_members)
        except Exception as exc:
            result["message"] = (
                f"Restore FAILED. Workspace may be inconsistent. "
                f"Pre-rollback snapshot: {pre_rollback_snap.name}. Error: {exc}"
            )
            logger.error(result["message"])
            return result

        # ── Step 2: 检查模型文件是否还在 ──
        manifest = self.get_manifest(snapshot_path)
        if manifest:
            for mf in manifest.get("model_files", []):
                mf_path = self.workspace / mf["path"]
                if mf_path.exists():
                    # 验证文件完整性（快速检查：大小和 hash）
                    try:
                        current_size = mf_path.stat().st_size / (1024 * 1024)
                        if abs(current_size - mf.get("size_mb", 0)) < 0.1:
                            result["model_files_found"] += 1
                        else:
                            result["model_files_missing"] += 1  # 大小变了，视为不匹配
                    except OSError:
                        result["model_files_missing"] += 1
                else:
                    result["model_files_missing"] += 1

            result["can_resume_training"] = (
                result["model_files_found"] > 0
                and result["model_files_missing"] == 0
            )

        # ── Step 3: 写恢复报告 ──
        found = result["model_files_found"]
        missing = result["model_files_missing"]
        if found == 0 and missing == 0:
            result["message"] = (
                f"Restored to {snapshot_path.name}. "
                f"No model checkpoints were recorded — training will start from scratch."
            )
        elif result["can_resume_training"]:
            result["message"] = (
                f"Restored to {snapshot_path.name}. "
                f"All {found} model checkpoint(s) intact → can RESUME training."
            )
        else:
            result["message"] = (
                f"Restored to {snapshot_path.name}. "
                f"Code restored but {missing} model checkpoint(s) missing (of {found + missing} total) → "
                f"training will restart from scratch."
            )

        # 写 manifest 到 workspace，让 agent 知道可以续训
        if result["success"]:
            self._manifest_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        logger.info(result["message"] + f" (pre-rollback: {pre_rollback_snap.name})")
        return result

    def _cleanup_non_archive_files(self, archive_members: set[str]) -> int:
        """删除快照后 agent 新建、且不在 archive 里的文件。

        绝不碰：checkpoints/（权重）、data/（数据集）、.snapshots/、
        .directive_archive/、.user_inputs/。只删这些目录之外的、不在
        archive 记录中的文件（代码/日志/辅助脚本）。
        """
        protected_dirs = {"checkpoints", "data", ".snapshots",
                          ".directive_archive", ".user_inputs"}
        deleted = 0

        def _walk(directory: Path):
            nonlocal deleted
            for item in sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name)):
                rel = item.relative_to(self.workspace).as_posix()
                if item.is_dir():
                    if item.name in protected_dirs:
                        continue
                    _walk(item)
                    # 空目录清理
                    try:
                        if not any(item.iterdir()):
                            item.rmdir()
                    except OSError:
                        pass
                elif item.is_file() and rel not in archive_members:
                    try:
                        item.unlink()
                        deleted += 1
                        logger.info("Cleaned post-snapshot file: %s", rel)
                    except OSError:
                        pass

        try:
            _walk(self.workspace)
        except Exception as exc:
            logger.warning("Cleanup of non-archive files failed (non-fatal): %s", exc)
        if deleted:
            logger.info("Rollback cleanup removed %d post-snapshot file(s)", deleted)
        return deleted

    def cleanup(self) -> int:
        """删除旧快照和旧 manifest，只保留最近 keep 个。"""
        snapshots = self.list_snapshots()
        deleted = 0
        for snap in snapshots[self.keep:]:
            try:
                snap.unlink()
                deleted += 1
                # 同时删 manifest
                mf = self.snapshots_dir / f"{snap.stem}.manifest.json"
                if mf.exists():
                    mf.unlink()
            except OSError:
                pass
        if deleted:
            logger.info("Cleaned up %d old snapshots", deleted)
        return deleted

    @property
    def latest(self) -> Optional[Path]:
        """最新快照路径。"""
        snaps = self.list_snapshots()
        return snaps[0] if snaps else None

    @property
    def rollback_manifest(self) -> Optional[dict]:
        """读取最近一次 rollback 的恢复报告。"""
        if self._manifest_path.exists():
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        return None


# ═══════════════════════════════════════════════════════════════════
# 3. Rollback 指令处理
# ═══════════════════════════════════════════════════════════════════

class RollbackHandler:
    """处理 HUMAN_DIRECTIVE.md 中的 rollback 指令。

    支持的指令（写到 workspace/HUMAN_DIRECTIVE.md）：
      rollback                    → 回退代码到上一个快照，检查模型文件是否可续训
      rollback --list             → 列出所有可用快照及模型文件状态
      rollback --checkpoint       → 只回退 LangGraph State（不回退文件）
      rollback --snapshot <name>  → 回退到指定快照
      rollback + resume <path>    → 回退后再从指定权重续训（epoch/best 回退用）
    """

    def __init__(self, workspace: Path, snapshot: WorkspaceSnapshot,
                 cycle_counter_path: Path):
        self.workspace = workspace
        self.snapshot = snapshot
        self._cycle_counter = cycle_counter_path
        self._directive_path = workspace / "HUMAN_DIRECTIVE.md"

    def check_directive(self) -> Optional[str]:
        """检查并处理 HUMAN_DIRECTIVE.md 中的 rollback 指令。返回 None 或结果描述。"""
        if not self._directive_path.exists():
            return None

        content = self._directive_path.read_text(encoding="utf-8").strip()
        if not content or "rollback" not in content.lower():
            return None

        logger.info("Rollback directive detected: %.200s", content)

        lines = content.split("\n")
        cmd_line = next((l for l in lines if "rollback" in l.lower()), content)

        if "--list" in cmd_line:
            result = self._handle_list()
        elif "--checkpoint" in cmd_line:
            result = self._handle_checkpoint_only()
        elif "--snapshot" in cmd_line:
            import re
            match = re.search(r"--snapshot\s+(\S+)", cmd_line)
            if match:
                snap_name = match.group(1)
                snap_path = self.snapshot.snapshots_dir / snap_name
                result = self._handle_full_rollback(snapshot_path=snap_path)
            else:
                result = "rollback --snapshot requires a name. Use --list to see available snapshots."
        else:
            result = self._handle_full_rollback()

        # 若指令同时含 resume <path>：把续训指令写进 RESUME_DIRECTIVE.md，
        # 让 agent 在下一轮 think 读到并从该权重续训（epoch/best 回退）。
        resume_line = next((l for l in lines if l.strip().lower().startswith("resume")), None)
        if resume_line:
            parts = resume_line.split(None, 1)
            if len(parts) < 2 or not parts[1].strip():
                # 指令只有 "resume" 没有路径:不写续训指令,也不崩溃(历史 bug:
                # 直接 [1] 取路径 → IndexError,带 rollback 指令时启动即崩)
                logger.warning("rollback 指令含 'resume' 但缺少权重路径,已忽略续训部分")
            else:
                resume_path = parts[1].strip()
                resume_dir = self.workspace / "RESUME_DIRECTIVE.md"
                resume_dir.write_text(
                    f"回退已完成。现在从以下权重续训：\n"
                    f"python train.py --resume {resume_path}\n"
                    f"（这是本轮的明确任务，不要重新从头训练）",
                    encoding="utf-8",
                )
                logger.info("Resume directive written: %s", resume_path)
                result = f"{result}\n→ 将从 {resume_path} 续训"

        # 保留非 rollback 的用户指令（BUG 7 修复）：
        # "rollback ...\n然后 try label smoothing" 中的后半句不能丢。
        # 归档整个文件前，把非 rollback 行写回 HUMAN_DIRECTIVE.md，
        # 供 nodes._consume_directive 下一轮读取执行。
        non_rollback_lines = [
            l for l in lines
            if "rollback" not in l.lower() and l.strip()
        ]
        self._archive_directive(content)
        if non_rollback_lines:
            try:
                self._directive_path.write_text(
                    "\n".join(non_rollback_lines) + "\n", encoding="utf-8")
                logger.info("Preserved non-rollback directive: %s",
                            " / ".join(non_rollback_lines)[:120])
            except OSError:
                pass
        return result

    def _handle_full_rollback(self, snapshot_path: Optional[Path] = None) -> str:
        """完整回退：恢复文件 + 检查模型文件。"""
        restore_result = self.snapshot.restore(snapshot_path)

        if not restore_result["success"]:
            return f"❌ {restore_result['message']}"

        # 回退 cycle counter
        manifest = self.snapshot.get_manifest(
            self.snapshot.snapshots_dir / restore_result["snapshot"]
        )
        if manifest:
            snap_cycle = manifest.get("cycle", 0)
            try:
                current = int(self._cycle_counter.read_text().strip())
                self._cycle_counter.write_text(str(snap_cycle))
            except (ValueError, OSError):
                self._cycle_counter.write_text(str(snap_cycle))

        # 格式化输出
        icon = "✅" if restore_result["can_resume_training"] else "⚠️"
        msg = f"{icon} {restore_result['message']}"
        return msg

    def _handle_checkpoint_only(self) -> str:
        """仅回退 Graph State（不改文件）。"""
        try:
            current = int(self._cycle_counter.read_text().strip())
            self._cycle_counter.write_text(str(max(0, current - 1)))
        except (ValueError, OSError):
            pass
        return "rollback --checkpoint: State reverted to previous cycle, files unchanged."

    def _handle_list(self) -> str:
        """列出可用快照及模型文件状态。"""
        snapshots = self.snapshot.list_snapshots()
        if not snapshots:
            return "No snapshots available."

        lines = ["Available snapshots:"]
        for i, snap in enumerate(snapshots):
            size_kb = snap.stat().st_size / 1024
            mtime = time.strftime("%m-%d %H:%M", time.localtime(snap.stat().st_mtime))
            manifest = self.snapshot.get_manifest(snap)
            n_models = len(manifest.get("model_files", [])) if manifest else 0
            cycle = manifest.get("cycle", "?") if manifest else "?"

            # 检查模型文件是否还在
            models_ok = 0
            if manifest:
                for mf in manifest.get("model_files", []):
                    if (self.workspace / mf["path"]).exists():
                        models_ok += 1

            status = ""
            if n_models > 0:
                if models_ok == n_models:
                    status = f" [models: {models_ok}/{n_models} OK ✓]"
                else:
                    status = f" [models: {models_ok}/{n_models} MISSING ✗]"

            lines.append(
                f"  [{i + 1}] cycle={cycle} | {size_kb:.0f}KB | {mtime}{status}"
            )
        return "\n".join(lines)

    def _archive_directive(self, content: str):
        archive_dir = self.workspace / ".directive_archive"
        archive_dir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"directive_{ts}.md"
        archive_path.write_text(content, encoding="utf-8")
        self._directive_path.write_text("", encoding="utf-8")
        logger.info("Directive archived to %s", archive_path)
