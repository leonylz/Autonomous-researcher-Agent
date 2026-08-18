"""
AutoResearcher Dashboard — FastAPI 监控 + 控制面板。

与 agent（core/nodes.py）完全解耦：
  - 只读 workspace 下的文件（audit.jsonl / costs.jsonl / experiments.jsonl /
    memory.db / .snapshots/ / 各类 markdown / 日志）
  - 只写一个文件：workspace/HUMAN_DIRECTIVE.md（agent 最高优先级指令通道）
  - 以子进程方式启动/停止 agent（sys.executable -m core.nodes ...）

刻意不用 `from core.xxx import ...`：core/__init__.py 会拉起整个 LLM 栈
（langgraph/langchain），dashboard 作为监控服务不该背这个启动开销，也用不到。
全部用标准库直接读文件 —— 快、稳、零依赖（除 fastapi/uvicorn）。

用法:
    python -m core.dashboard --project examples/toy_experiment --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import io
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

try:
    from fastapi import FastAPI, File, HTTPException, Request, UploadFile
    from fastapi.responses import HTMLResponse, StreamingResponse
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise SystemExit("fastapi 未安装。请运行: pip install fastapi uvicorn") from exc

logger = logging.getLogger("autoresearcher.dashboard")

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_MODULE = "core.nodes"
DEFAULT_CONFIG = "config.yaml"
DEFAULT_WORKSPACE = "workspace"

# 事件日志读取（只读尾随，不触发 agent 侧依赖）
try:
    from .event_journal import EventJournal
except ImportError:  # pragma: no cover
    class EventJournal:  # type: ignore[no-redef]
        """缺失时降级：SSE 仅推快照，不推事件。"""

        def __init__(self, path):  # noqa: D107
            self.path = path

        def read_from(self, after_seq=0, limit=200, types=None) -> list:
            return []

# ═══════════════════════════════════════════════════════════════════════
# 纯标准库文件读取（缺失 → 空值，绝不抛异常）
# ═══════════════════════════════════════════════════════════════════════


def _read_text(path: Path) -> str:
    """读文本文件，缺失/坏编码 → 空串。"""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _jsonl(path: Path) -> list[dict]:
    """读 JSONL 文件，跳过坏行。"""
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            if isinstance(d, dict):
                out.append(d)
        except json.JSONDecodeError:
            continue
    return out


def _poll_rollback_result(console: Path, start_pos: int, manifest: Path,
                          manifest_ts: float, mode: str) -> tuple:
    """在独立线程轮询回退结果（45s 上限），避免阻塞异步事件循环。

    返回 (result, polled_seconds)。result 为 None 表示超时未出现。
    """
    import time as _t
    result = None
    polled = 0.0
    deadline = _t.time() + 45
    while _t.time() < deadline:
        _t.sleep(1)
        polled += 1
        if console.exists():
            full = _read_text(console)
            scan_from = start_pos
            if start_pos > 0:
                # 从 start_pos 之后第一个换行开始（避免半行截断）
                nl = full.find("\n", max(0, start_pos - 1))
                scan_from = nl + 1 if nl >= 0 else start_pos
            for line in full[scan_from:].splitlines():
                if "[rollback]" in line:
                    result = line.split("[rollback]", 1)[1].strip()
                    break
        if result:
            break
        if mode in ("default", "snapshot") and manifest.exists() and manifest.stat().st_mtime > manifest_ts:
            try:
                result = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:
                result = {"manifest": "unreadable"}
            break
    return result, polled


def _tail_lines(path: Path, n: int) -> list[str]:
    """返回文件最后 n 行。"""
    lines = _read_text(path).splitlines()
    return lines[-n:] if n > 0 else lines


# 日志噪音行（HTTP 请求等），SSE 实时日志里过滤掉，避免淹没训练进度
_LOG_NOISE_PATTERNS = (
    "HTTP Request: POST", "httpx", "openai._base_client",
    "Retrying request to", "Request timed out",
)


def _is_log_noise(line: str) -> bool:
    """判断一行日志是否是噪音（HTTP 调用/重试）。"""
    return any(p in line for p in _LOG_NOISE_PATTERNS)


def _tail_merged(sources: list[tuple[str, Path]], n: int = 100) -> list[str]:
    """合并多个日志文件尾部，每行加文件名前缀，供 SSE 推送。

    过滤 HTTP 请求等噪音；训练日志（train）优先保留，其余源次之。
    """
    train_lines: list[str] = []
    other_lines: list[str] = []
    for name, path in sources:
        for line in _tail_lines(path, n):
            if _is_log_noise(line):
                continue
            prefixed = f"[{name}] {line}"
            if name == "train":
                train_lines.append(prefixed)
            else:
                other_lines.append(prefixed)
    # 训练日志 + 其他日志合并，训练进度靠前
    merged = train_lines + other_lines
    return merged[-n:]


def _load_config(project_dir: Path) -> dict:
    """读 config.yaml 拿 project.name / project.workspace。pyyaml 缺失则返回 {}。"""
    try:
        import yaml
    except ImportError:
        return {}
    path = project_dir / DEFAULT_CONFIG
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _checkpoint_count(workspace: Path) -> int:
    """checkpoints.db 中 LangGraph State 快照数（Dashboard 只读）。"""
    db = workspace / "checkpoints.db"
    if not db.exists():
        return 0
    try:
        with sqlite3.connect(str(db)) as conn:
            return conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
    except Exception:
        return 0


def _read_int(path: Path, default: int = 0) -> int:
    """读整数文件，缺失/坏 → default。"""
    try:
        return int(_read_text(path).strip() or default)
    except ValueError:
        return default


def _cost_for(workspace: Path) -> float:
    """sum workspace/costs.jsonl 的 cost_usd。"""
    return round(sum(float(e.get("cost_usd", 0) or 0) for e in _jsonl(workspace / "costs.jsonl")), 6)


def _render_brief(name: str, goal: str, success_criteria: str, constraints: str,
                  dataset_path: str = "", model_path: str = "",
                  literature: str = "") -> str:
    """生成新项目的 PROJECT_BRIEF.md（结构对齐 toy_experiment）。

    dataset_path / model_path：用户提交的外部绝对路径，写进 ## Inputs 段让 agent
    在 train.py 里直接引用（只记录路径，不拷贝）。
    literature：用户提供的文献列表（每行一篇），idea_agent 优先分析，绕过 429。
    """
    title = goal.strip().splitlines()[0] if goal.strip() else name
    data_line = (f"- Data: {dataset_path.strip()} (user-provided absolute path)"
                 if dataset_path.strip()
                 else "- Data: (auto-download or user-provided)")
    brief = (
        f"# {name} — {title}\n\n"
        f"## Goal\n{goal.strip()}\n\n"
        f"## Codebase\n"
        f"- Training script: train.py (to be created by the agent)\n"
        f"{data_line}\n"
        f"- Checkpoints: ./checkpoints/ (模型权重保存目录)\n\n"
        f"## Constraints\n{constraints.strip()}\n\n"
        f"## Success Criteria\n{success_criteria.strip()}\n\n"
        f"## MUST-DO (训练脚本硬性要求，违反将被 launch_experiment 拒绝)\n"
        f"- 训练脚本必须基于框架模板：`cp core/train_template.py train.py` "
        f"后只改模型/数据/训练三处 TODO，不得删除 checkpoint 与 dry-run 逻辑\n"
        f"- checkpoint 行为由 config.yaml 的 checkpoint 段控制：\n"
        f"  · 每 checkpoint.save_every_n_epochs 个 epoch 存 ./checkpoints/checkpoint_epoch_{{N}}.pth\n"
        f"  · 最优存 ./checkpoints/best_model.pth（始终保留）\n"
        f"  · 脚本必须能加载 ./checkpoints/best_model.pth 或 checkpoint_epoch_{{N}}.pth 续训\n"
        f"- 真实训练前必须先跑 dry-run（写 dry_run_log.json），否则 launch_experiment 拒绝\n"
        f"- dry-run/试跑 的权重存到 /tmp/dryrun（或 ./checkpoints_dryrun），"
        f"严禁与真实训练共用 ./checkpoints/，也严禁真实训练用 dry-run 权重续训\n"
    )
    inputs = []
    if dataset_path.strip():
        inputs.append(f"- Dataset: {dataset_path.strip()}")
    if literature.strip():
        lit_lines = [l.strip() for l in literature.splitlines() if l.strip()]
        inputs.append(
            f"- User literature（用户提供，idea_agent 必须优先分析，不要调 search_papers）:\n"
            + "\n".join(f"  - {l}" for l in lit_lines[:20])
        )
    if model_path.strip():
        # 目录 = 用户提供的模型框架（如 GitHub 项目）→ agent 迭代起点
        # 文件(.pth) = baseline 权重 → 评估后对比超越
        inputs.append(
            f"- Model framework: {model_path.strip()}\n"
            f"  → 这是用户提供的模型框架/代码，agent 必须基于它迭代改进，"
            f"不要从零重写。若它含已训练权重，先评估出 baseline 指标。\n"
            f"  → 之后每轮实验结果都必须和这个 baseline 对比，目标是超越它。"
            f"（HUMAN_DIRECTIVE 永远优先于以上要求）"
        )
    if inputs:
        brief += ("\n## Inputs (用户提交，训练脚本必须直接引用这些绝对路径)\n"
                  + "\n".join(inputs) + "\n")
    return brief


def _render_config(name: str) -> str:
    """生成新项目的最小 config.yaml（对齐 toy_experiment，默认 deepseek）。"""
    return (
        f'project:\n'
        f'  name: "{name}"\n'
        f'  brief: "PROJECT_BRIEF.md"\n\n'
        f'agent:\n'
        f'  provider: "deepseek"\n'
        f'  model: "deepseek-chat"\n'
        f'  max_cycles: 3\n'
        f'  max_steps_per_cycle: 2\n'
        f'  cooldown_interval: 60\n\n'
        f'gpu:\n'
        f'  auto_detect: true\n'
        f'  reserve_last: false\n\n'
        f'monitor:\n'
        f'  poll_interval: 30\n'
        f'  zero_llm: true\n\n'
        f'experiment:\n'
        f'  mandatory_dry_run: true\n'
        f'  max_parallel: 1\n\n'
        f'checkpoint:\n'
        f'  save_every_n_epochs: 5\n'
        f'  keep_best: true\n'
    )


# ═══ 草案生成（Draft & Confirm）：LLM 展开一句话目标为三字段 ═══
# 只做一次性调用，不写任何文件；最终 PROJECT_BRIEF.md 仍由
# /api/project/new → ProjectRegistry.create → _render_brief 生成。
# 失败一律降级为「用户手动填写」，绝不阻塞建项目流程。

# 国内 API preset（复制自 nodes.py 的 PRESETS，dashboard 刻意不 import core）
_DRAFT_PROVIDER_PRESETS = {
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "dashscope": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "moonshot": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
    "kimi": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
    "zhipu": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPUAI_API_KEY"),
    "glm": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPUAI_API_KEY"),
}

_DRAFT_DEFAULT_URL = "https://api.deepseek.com/v1"
_DRAFT_DEFAULT_MODEL = "deepseek-chat"
_DRAFT_DEFAULT_KEY_ENV = "DEEPSEEK_API_KEY"


def _draft_llm_settings() -> tuple[str, str, str]:
    """草案调用的 (base_url, model, api_key) 解析。

    默认 deepseek（与 _render_config 的新项目默认一致、用户已验证可用）。
    仓库根 config.yaml 的 agent 段只有指定国内 preset 且对应 env 存在时
    才被采用（含 model）；否则（如当前的 anthropic 配置）静默回退
    deepseek 默认，避免草案调用拿错 key。
    """
    agent = _load_config(REPO_ROOT).get("agent") or {}
    provider = str(agent.get("provider") or "").strip().lower()
    preset = _DRAFT_PROVIDER_PRESETS.get(provider)
    if preset:
        url, key_env = preset
        api_key = os.environ.get(key_env, "")
        if api_key:
            model = str(agent.get("model") or "").strip() or _DRAFT_DEFAULT_MODEL
            return url, model, api_key
    return (_DRAFT_DEFAULT_URL, _DRAFT_DEFAULT_MODEL,
            os.environ.get(_DRAFT_DEFAULT_KEY_ENV, ""))


_DRAFT_SYSTEM = (
    "你是一个深度学习实验设计助手。用户要启动一个 24/7 自动研究 agent（AutoResearcher）"
    "来做一个实验项目，用户只给了一句话目标，请把它扩展为三个字段。要求：\n"
    "1. goal（研究目标）：用中文完整重述目标，写清任务、模型方向、数据与目标效果。\n"
    "2. success_criteria（成功标准）：必须量化——只用用户目标里出现的指标 + 数值阈值"
    "（例如目标说准确率，就写\"测试集准确率>85%\"；可补充\"训练成功完成无报错\"作为必要条件）。"
    "用户目标没提到的指标（如 loss）禁止添加阈值——不要编造用户没要求的数字。\n"
    "3. constraints（约束）：给出适合深度学习实验 agent 的现实默认约束，"
    "必须包含：PyTorch 框架、最大 epoch 数（如 50）、GPU 显存/卡数限制、"
    "训练前必须先跑 dry-run。禁止编造任何数据路径或本地文件路径。\n"
    "只输出一个 JSON 对象，不要输出任何解释、markdown 代码块或其他文字，格式如下：\n"
    '{"goal": "...", "success_criteria": "...", "constraints": "..."}\n'
    "三个字段的值都必须是中文非空字符串。"
)


def _draft_prompt(goal: str, name: str = "") -> str:
    """组装草案 user 消息；项目名可选，帮助生成更贴合的草案。"""
    lines = []
    if name:
        lines.append(f"项目名：{name}")
    lines.append(f"一句话目标：{goal}")
    return "\n".join(lines)


def _parse_draft_json(text: str) -> dict:
    """解析 LLM 返回的草案 JSON。容忍 ``` 围栏；坏 JSON / 缺字段 → ValueError。"""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("LLM returned non-JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("LLM returned non-object JSON")
    for key in ("goal", "success_criteria", "constraints"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"LLM JSON missing/invalid field: {key}")
    return {k: data[k].strip() for k in ("goal", "success_criteria", "constraints")}


def _call_draft_llm(base_url: str, api_key: str, model: str, prompt: str) -> str:
    """一次性 OpenAI 兼容 chat 调用（惰性 import，保持 dashboard 零 LLM 启动开销）。"""
    import openai  # lazy: openai>=1.30.0 为 requirements 主依赖

    client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=60)
    last_exc: Optional[Exception] = None
    for attempt in range(2):  # 瞬时错误重试 1 次
        if attempt:
            time.sleep(1.0)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": _DRAFT_SYSTEM},
                          {"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # openai 错误 / 超时 / 连接失败
            last_exc = exc
    raise RuntimeError(str(last_exc)) from last_exc


class ProjectRegistry:
    """扫描/创建 projects_root 下的研究项目。"""

    def __init__(self, projects_root: Path):
        self.projects_root = projects_root

    def discover(self) -> list[dict]:
        """列出 projects_root 下同时有 config.yaml + PROJECT_BRIEF.md 的子目录。"""
        if not self.projects_root.exists():
            return []
        out = []
        for d in sorted(self.projects_root.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            if not ((d / DEFAULT_CONFIG).exists() and (d / "PROJECT_BRIEF.md").exists()):
                continue
            cfg = _load_config(d)
            name = (cfg.get("project", {}) or {}).get("name") or d.name
            ws = d / ((cfg.get("project", {}) or {}).get("workspace") or DEFAULT_WORKSPACE)
            out.append({
                "name": name, "path": str(d), "dir_name": d.name,
                "cycle": _read_int(ws / ".cycle_counter"),
                "has_workspace": ws.exists(),
                "cost_usd": _cost_for(ws),
            })
        return out

    def create(self, name: str, goal: str, success_criteria: str,
               constraints: str, dataset_path: str = "", model_path: str = "",
               literature: str = "") -> dict:
        """创建新项目目录 + PROJECT_BRIEF.md + 最小 config.yaml。

        dataset_path：数据集路径，只写进 BRIEF（不拷贝，数据集大）。
        model_path：模型框架。若是目录（如 GitHub 项目）→ 拷进
        workspace/model_framework/ 作为 agent 迭代起点；若是 .pth → 只记录路径。
        """
        safe = re.sub(r"[^\w\-]", "_", name.strip())
        if not safe:
            raise ValueError("invalid project name (empty after sanitize)")
        proj_dir = self.projects_root / safe
        if proj_dir.exists():
            raise FileExistsError(f"project already exists: {safe}")
        proj_dir.mkdir(parents=True, exist_ok=False)
        proj_dir.joinpath("PROJECT_BRIEF.md").write_text(
            _render_brief(safe, goal, success_criteria, constraints,
                          dataset_path, model_path, literature), encoding="utf-8")
        proj_dir.joinpath(DEFAULT_CONFIG).write_text(
            _render_config(safe), encoding="utf-8")

        # 用户提供的文献 → 写入 USER_LITERATURE.md（idea_agent 优先读它，绕过 429）
        if literature.strip():
            ws = proj_dir / DEFAULT_WORKSPACE
            ws.mkdir(parents=True, exist_ok=True)
            ws.joinpath("USER_LITERATURE.md").write_text(
                literature.strip() + "\n", encoding="utf-8")

        # 模型框架目录 → 拷进 workspace（agent 的迭代起点）
        framework_copied = None
        if model_path.strip():
            mp = Path(model_path.strip())
            if mp.is_dir():
                ws = proj_dir / DEFAULT_WORKSPACE
                ws.mkdir(parents=True, exist_ok=True)
                dst = ws / "model_framework"
                try:
                    import shutil as _sh
                    _sh.copytree(mp, dst,
                                 ignore=_sh.ignore_patterns(
                                     ".git", "__pycache__", "*.pyc",
                                     "node_modules", ".venv"))
                    framework_copied = str(dst)
                except (OSError, shutil.Error) as exc:
                    raise FileExistsError(
                        f"failed to copy model framework: {exc}") from exc

        result = {"name": safe, "path": str(proj_dir)}
        if framework_copied:
            result["framework_copied_to"] = framework_copied
        return result


def _memory_store(workspace: Path, project_name: str) -> dict:
    """读 memory.db 的 memories 表：stats + 该项目近期 insights（不加载向量模型）。"""
    db = workspace / "memory.db"
    stats: dict = {"total_entries": 0}
    insights: list[dict] = []
    if db.exists():
        try:
            with sqlite3.connect(str(db)) as conn:
                total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                stats["total_entries"] = total
                rows = conn.execute(
                    """SELECT id, project, namespace, text, metadata, created_at
                       FROM memories WHERE project = ?
                       ORDER BY created_at DESC LIMIT 20""",
                    (project_name,),
                ).fetchall()
            insights = [
                {
                    "id": r[0], "project": r[1], "namespace": r[2], "text": r[3],
                    "metadata": json.loads(r[4]) if r[4] else {}, "created_at": r[5],
                }
                for r in rows
            ]
            stats["by_project"] = {project_name: total}
            stats["db_size_kb"] = round(db.stat().st_size / 1024, 1)
        except Exception:
            pass
    return {"stats": stats, "insights": insights}


def _snapshots(workspace: Path) -> list[dict]:
    """列出 .snapshots/ 下的快照 + manifest + 模型文件存在性（最新在前）。"""
    snap_dir = workspace / ".snapshots"
    if not snap_dir.exists():
        return []
    paths = sorted(snap_dir.glob("snap_*.tar.gz"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[dict] = []
    for i, p in enumerate(paths):
        m: dict = {}
        manifest_path = snap_dir / f"{p.stem}.manifest.json"
        if manifest_path.exists():
            try:
                m = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                m = {}
        model_files = []
        for mf in (m.get("model_files") or []):
            rel = str(mf.get("path", ""))
            model_files.append({
                "path": rel, "size_mb": mf.get("size_mb", 0),
                "md5_partial": mf.get("md5_partial", ""),
                "exists": (workspace / rel).exists(),
            })
        out.append({
            "name": p.name,
            "size_bytes": p.stat().st_size,
            "mtime": p.stat().st_mtime,
            "created_at": m.get("created_at", ""),
            "cycle": m.get("cycle", 0),
            "archive_size_kb": m.get("archive_size_kb", round(p.stat().st_size / 1024, 1)),
            "small_files": m.get("small_files", []),
            "model_files": model_files,
            "models_total": len(model_files),
            "models_ok": sum(1 for x in model_files if x["exists"]),
            "latest": i == 0,
        })
    return out


MODEL_GLOBS = ["*.pth", "*.pt", "*.ckpt", "*.safetensors", "*.h5", "*.onnx"]


def _render_summary_md(dash: "_Dashboard") -> str:
    """生成指标报告 markdown（交付包核心内容之一）。"""
    lines = [f"# {dash.project_name} 实验结果报告", ""]
    # 实验配置（脚本 + 参数 + 优化器）
    cfg = _parse_training_config(dash)
    if cfg.get("script"):
        lines.append("## 实验配置")
        lines.append(f"- **训练脚本**: `{cfg['script']}`")
        if cfg.get("optimizer"):
            lines.append(f"- **优化器**: {cfg['optimizer']}")
        if cfg.get("criterion"):
            lines.append(f"- **Loss**: {cfg['criterion']}")
        if cfg.get("params"):
            p = cfg["params"]
            parts = [f"{k}={v}" for k, v in p.items() if k in
                     ("epochs", "batch_size", "lr", "dropout", "resume_from")]
            if parts:
                lines.append(f"- **参数**: {', '.join(parts)}")
        lines.append("")
    # 最近实验
    exps = dash.experiment_entries
    if exps:
        lines.append("## 实验记录")
        for e in exps[-5:]:
            m = e.get("metrics") or {}
            m_str = ", ".join(f"{k}={v}" for k, v in m.items()) if m else "—"
            lines.append(f"- **cycle {e.get('cycle')}** ({e.get('status')}): {e.get('conclusion','')}")
            lines.append(f"  - 指标: {m_str}")
        lines.append("")
    # 训练日志（每 epoch 参数）
    log = dash.workspace / "checkpoints" / "training_log.json"
    if log.exists():
        try:
            data = json.loads(log.read_text(encoding="utf-8", errors="replace"))
            hist = data.get("history") or []
            if hist:
                lines.append("## 训练曲线")
                lines.append("| epoch | lr | train_loss | train_acc | test_acc |")
                lines.append("|-------|-----|-----------|----------|---------|")
                for h in hist:
                    lines.append(f"| {h.get('epoch')} | {h.get('lr','-')} | "
                                 f"{h.get('train_loss','-')} | {h.get('train_acc','-')} | "
                                 f"{h.get('test_acc','-')} |")
                lines.append("")
            if data.get("final_test_acc") is not None:
                lines.append(f"**最优测试准确率: {data['final_test_acc']}**")
        except (json.JSONDecodeError, OSError):
            pass
    # 最优权重信息
    ckpts = _list_checkpoints(dash)
    if ckpts:
        best = [c for c in ckpts if c["kind"] == "best"]
        lines.append("## 模型权重")
        if best:
            b = best[0]
            lines.append(f"- **best_model.pth** ({b['size_mb']} MB) — 训练最优权重")
        epochs = [c for c in ckpts if c["kind"] == "epoch"]
        if epochs:
            names = ", ".join(c["name"] for c in epochs)
            lines.append(f"- 分 epoch 权重: {names}")
    return "\n".join(lines)


def _list_checkpoints(dash: "_Dashboard") -> list[dict]:
    """列出 checkpoints/ 下的权重（供交付包选权重 + /api/checkpoints 复用）。"""
    ckpt_dir = dash.workspace / "checkpoints"
    out = []
    if ckpt_dir.is_dir():
        for p in sorted(ckpt_dir.glob("*.pth")):
            try:
                mb = round(p.stat().st_size / 1e6, 2)
            except OSError:
                mb = 0
            out.append({"name": p.name,
                        "kind": "best" if p.name == "best_model.pth" else "epoch",
                        "size_mb": mb, "path": str(p)})
    return out


def _build_deliverable(dash: "_Dashboard",
                       weights: str = "best") -> tuple[bytes, str, dict]:
    """把项目成果打包成 tar.gz：模型脚本 + 指标报告 + 权重（可选）。

    Parameters
    ----------
    weights : str
        "best"   → 只含 best_model.pth（默认，最常用）
        "all"    → 所有 checkpoints/*.pth
        "none"   → 不含权重
        其他     → 指定的权重文件名（如 "checkpoint_epoch_5.pth"）

    Returns
    -------
    (tar_bytes, filename, summary)
    """
    buf = io.BytesIO()
    summary: dict = {"files": [], "models": [], "total_mb": 0.0}
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        def _add(path: Path, arcname: str):
            if path.is_file():
                tar.add(str(path), arcname=arcname)
                summary["files"].append(arcname)
        # 1. 模型脚本（训练框架）
        for name in ("train.py", "train_enhanced_cnn.py"):
            _add(dash.workspace / name, name)
        # 2. 指标报告（自动生成）
        md = _render_summary_md(dash)
        summary_md = io.BytesIO(md.encode("utf-8"))
        ti = tarfile.TarInfo("RESULT.md")
        ti.size = len(md.encode("utf-8"))
        tar.addfile(ti, summary_md)
        summary["files"].append("RESULT.md")
        # 原始指标 json
        _add(dash.workspace / "checkpoints" / "training_log.json",
             "training_log.json")
        _add(dash.workspace / "experiments.jsonl", "experiments.jsonl")
        # 3. 权重（可选）
        ckpts = _list_checkpoints(dash)
        if weights == "best":
            selected = [c for c in ckpts if c["kind"] == "best"]
        elif weights == "all":
            selected = ckpts
        elif weights == "none":
            selected = []
        else:
            selected = [c for c in ckpts if c["name"] == weights]
        for c in selected:
            _add(Path(c["path"]), f"weights/{c['name']}")
            summary["models"].append({"name": c["name"], "mb": c["size_mb"]})
            summary["total_mb"] += c["size_mb"]
    summary["total_mb"] = round(summary["total_mb"], 1)
    filename = f"deliverable_{dash.project_dir.name}_{time.strftime('%Y%m%d_%H%M%S')}.tar.gz"
    return buf.getvalue(), filename, summary


def _preview_asset(path: Path, kind: str) -> dict:
    """预览外部资产路径（不拷贝、不递归扫大目录）。"""
    if not path.exists():
        raise HTTPException(status_code=400, detail="path not found")
    if path.is_dir():
        entries = []
        total = 0
        sample = []
        try:
            for e in os.scandir(path):
                if e.is_dir():
                    entries.append(f"[dir] {e.name}")
                else:
                    try:
                        total += e.stat().st_size
                    except OSError:
                        pass
                    entries.append(e.name)
                if len(entries) <= 5:
                    sample.append(e.name)
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"cannot read dir: {exc}")
        return {
            "kind": "dir", "path": str(path),
            "entries": entries[:50], "entry_count": len(entries),
            "sample": sample[:5],
            "size_mb": round(total / 1e6, 2),
            "is_model": False,
        }
    # 单文件
    try:
        size_mb = round(path.stat().st_size / 1e6, 2)
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"cannot stat file: {exc}")
    ext = path.suffix.lower()
    is_model = ext in (".pth", ".pt", ".ckpt", ".safetensors", ".h5", ".onnx")
    if kind == "model":
        note = ("模型框架目录 → 将拷入 workspace/model_framework 作为迭代起点"
                if not is_model else "模型权重文件 → agent 评估后作为 baseline 对比")
    else:
        note = "模型权重文件" if is_model else ("目录或数据集文件" if kind == "dataset" else "文件（非常见权重格式，仍会记录路径）")
    return {
        "kind": "file", "path": str(path), "name": path.name,
        "size_mb": size_mb, "is_model": is_model, "note": note,
    }


# workspace 核心文件（清理时绝不删除）
_CLEANUP_PROTECTED = {
    "PROJECT_BRIEF.md", "config.yaml", "MEMORY_LOG.md", "INSIGHTS.md",
    "DEAD_ENDS.md", "HUMAN_DIRECTIVE.md", "state.json", ".cycle_counter",
    "checkpoints.db", "memory.db", "train.py", "experiments.jsonl",
    "audit.jsonl", "costs.jsonl", "training_log.json",
    "training_output.log", "dry_run_log.json",
}
# 可清理的辅助文件模式（agent 测试/检查时创建的杂项）
_CLEANUP_PATTERNS = (
    "dryrun_*.py", "dryrun_*.txt", "inspect_*.py", "resume_*.py",
    "test_*.py", "quick_*.py", "*_info.txt", "*_keys.txt", "*.out",
)


def _scan_cleanup_candidates(workspace: Path) -> list[dict]:
    """扫描 workspace 下的可清理辅助文件（不碰核心/权重/数据/快照）。"""
    out = []
    if not workspace.is_dir():
        return out
    import fnmatch
    for p in sorted(workspace.iterdir()):
        if p.name.startswith("."):
            continue
        if p.name in _CLEANUP_PROTECTED:
            continue
        if p.is_dir() and p.name in ("data", "checkpoints", ".snapshots",
                                     "directive_archive", ".user_inputs"):
            continue
        # 匹配辅助文件模式
        matched = any(fnmatch.fnmatch(p.name, pat) for pat in _CLEANUP_PATTERNS)
        if matched and p.is_file():
            try:
                kb = round(p.stat().st_size / 1024, 1)
            except OSError:
                kb = 0
            out.append({"name": p.name, "size_kb": kb,
                        "kind": "auxiliary", "selected": False})
    return out


def _parse_approvals(workspace: Path) -> list[dict]:
    """解析 PENDING_APPROVALS.md → 审批请求列表（含是否已回复）。"""
    path = workspace / "PENDING_APPROVALS.md"
    content = _read_text(path)
    if not content.strip():
        return []
    reqs: list[dict] = []
    id_re = re.compile(r"##\s*\[([0-9a-f]+)\]\s*(\S+)\s*\(risk:\s*(\w+)\)")
    for m in id_re.finditer(content):
        rid, action, risk = m.group(1), m.group(2), m.group(3)
        # 从该行往后找 cost/detail/time 和 decision
        block = content[m.end():]
        block = block[:block.find("## [")] if "## [" in block else block
        cost = ""
        detail = ""
        ts = ""
        mc = re.search(r"- Cost estimate: \$([0-9.]+)", block)
        md = re.search(r"- Detail: (.+)", block)
        mt = re.search(r"- Time: (.+)", block)
        if mc: cost = mc.group(1)
        if md: detail = md.group(1).strip()
        if mt: ts = mt.group(1).strip()
        # 是否已回复（与 approval.py check_response 一致：容忍 **APPROVE** 加粗）
        decision = None
        lines = block.split("\n")
        for j in range(min(len(lines), 30)):
            m = re.search(r"^\s*\*?\*?(APPROVE|DENY|REJECT)\b",
                          lines[j], re.IGNORECASE)
            if m:
                decision = "approved" if m.group(1).upper() == "APPROVE" else "denied"
                break
        reqs.append({
            "id": rid, "action": action, "risk": risk,
            "cost": cost, "detail": detail, "time": ts,
            "decision": decision,  # None = 待审批
        })
    return reqs


def _respond_approval(workspace: Path, req_id: str, decision: str) -> bool:
    """在 PENDING_APPROVALS.md 的 [id] 块里写入 APPROVE/DENY。

    格式须与 core/approval.py 的 check_response() 兼容：它找 [id] 行后
    扫描后 5 行，命中 approve*/deny*/reject* 前缀即返回决定。
    我们在 marker 行之前插入一行决定。
    """
    path = workspace / "PENDING_APPROVALS.md"
    content = _read_text(path)
    if not content:
        return False
    id_re = re.compile(r"(##\s*\[" + re.escape(req_id) + r"\]\s*\S+\s*\(risk:\s*\w+\)\n)")
    if not id_re.search(content):
        return False
    word = "APPROVE" if decision == "approve" else "DENY"
    marker = "=" * 40
    # 定位该请求的块：从 "## [id]" 行开始，到下一个 "## [" 或文件尾
    m = id_re.search(content)
    if not m:
        return False
    block_start = m.end()          # 块头行之后
    rest = content[block_start:]
    next_block = rest.find("## [")
    block = rest if next_block < 0 else rest[:next_block]
    tail = rest[len(block):]
    # 已回复判断：检查块内是否有独立的决定标记行（**APPROVE** / **DENY**）
    if re.search(r"\*\*(APPROVE|DENY)\*\*", block, re.IGNORECASE):
        return True  # 已回复
    # 在块内最后一个 marker 前插入（marker 紧跟在块尾）
    marker_pos = block.rfind(marker)
    if marker_pos < 0:
        # 没有 marker，追加到块尾
        block = block + f"**{word}**\n\n"
    else:
        block = block[:marker_pos] + f"**{word}**\n\n" + block[marker_pos:]
    path.write_text(content[:block_start] + block + tail, encoding="utf-8")
    return True


def _audit_summary(entries: list[dict]) -> dict:
    """audit.jsonl 摘要（与 AuditLogger.summary 同语义）。"""
    actions: dict[str, int] = {}
    failures = 0
    total_cost = 0.0
    for e in entries:
        action = e.get("action", "unknown")
        actions[action] = actions.get(action, 0) + 1
        if e.get("result") in ("failed", "blocked"):
            failures += 1
        total_cost += float(e.get("cost_estimate", 0) or 0)
    return {
        "total_entries": len(entries),
        "by_action": actions,
        "failures": failures,
        "total_cost_estimate": round(total_cost, 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# 子进程管理器（Windows 安全终止）
# ═══════════════════════════════════════════════════════════════════════


class AgentProcessManager:
    """启动/停止 agent 子进程。只认自己 spawn 的进程。"""

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self.started_at: Optional[float] = None
        self.console_path: Optional[Path] = None
        self.start_pos: int = 0

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, project_dir: Path, config_name: str,
              max_cycles: Optional[int] = None) -> int:
        with self._lock:
            cmd = [os.environ.get("AGENT_PYTHON") or sys.executable,
                   "-m", AGENT_MODULE,
                   "--project", str(project_dir), "--config", config_name]
            if max_cycles:
                cmd += ["--max-cycles", str(int(max_cycles))]

            console = project_dir / "agent_console.log"
            console.parent.mkdir(parents=True, exist_ok=True)
            self.console_path = console
            self.start_pos = console.stat().st_size if console.exists() else 0

            # 所有 stdout/stderr 都落盘（append、unbuffered），避免 pipe 死锁
            fh = open(console, "ab", buffering=0)
            flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            # ⚠ 不要 scrub env：agent 主进程需要 API key 来调用 LLM。
            #    API key 的清洗只作用于 agent spawn 的训练/Shell 子进程（execution.py）。
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            self._proc = subprocess.Popen(
                cmd, cwd=REPO_ROOT,
                stdout=fh, stderr=subprocess.STDOUT,
                creationflags=flags,
                env=env,
            )
            self.started_at = time.time()
            return self._proc.pid

    def stop(self, timeout: float = 5.0) -> dict:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return {"ok": True, "already_stopped": True}
            pid = self._proc.pid
            # Windows: terminate() = TerminateProcess（硬杀）
            self._proc.terminate()
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # 整棵进程树一起杀（agent 可能 spawn 了 train.py 等）
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               capture_output=True)
                self._proc.kill()
                try:
                    self._proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pass
            rc = self._proc.poll()
            self._proc = None
            return {"ok": True, "pid": pid, "returncode": rc}


# ═══════════════════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════════════════

app = FastAPI(title="AutoResearcher Dashboard")
_manager = AgentProcessManager()


class _Dashboard:
    """按 --project 解析一次路径，提供各数据 reader。"""

    def __init__(self, project_dir: Path, config_name: str = DEFAULT_CONFIG):
        self.project_dir = project_dir.resolve()
        self.config_name = config_name
        self.config = _load_config(self.project_dir)
        ws = (self.config.get("project", {}) or {}).get("workspace") or DEFAULT_WORKSPACE
        self.workspace = self.project_dir / ws
        self.project_name = ((self.config.get("project", {}) or {}).get("name")
                             or self.project_dir.name)

    # ── 数据读取 ──
    @property
    def audit_entries(self) -> list[dict]:
        return _jsonl(self.workspace / "audit.jsonl")

    @property
    def cost_entries(self) -> list[dict]:
        return _jsonl(self.workspace / "costs.jsonl")

    @property
    def experiment_entries(self) -> list[dict]:
        return _jsonl(self.workspace / "experiments.jsonl")

    def cycle(self) -> int:
        try:
            return int(_read_text(self.workspace / ".cycle_counter").strip() or 0)
        except ValueError:
            return 0

    def directive_pending(self) -> bool:
        text = _read_text(self.workspace / "HUMAN_DIRECTIVE.md")
        return bool(text.strip())

    def total_cost(self) -> float:
        return round(sum(float(e.get("cost_usd", 0) or 0) for e in self.cost_entries), 6)

    # ── 快照 ──
    def snapshot_payload(self) -> dict:
        # 从 state.json 读 agent 实时 phase/detail（nodes.py _update_state 写入）
        state = {}
        sp = self.workspace / "state.json"
        if sp.exists():
            try:
                state = json.loads(sp.read_text(encoding="utf-8", errors="replace"))
            except json.JSONDecodeError:
                state = {}
        running = _manager.is_running()
        # agent 不在跑时，phase 一律归零为 stopped/idle（避免显示上次遗留的 execute/monitor）
        phase = state.get("phase", "idle") if running else "idle"
        return {
            "project": str(self.project_dir),
            "workspace": str(self.workspace),
            "running": running,
            "pid": _manager._proc.pid if running else None,
            "returncode": _manager._proc.returncode if _manager._proc else None,
            "started_at": _manager.started_at,
            "cycle": self.cycle(),
            "phase": phase,
            "phase_next": state.get("next", "") if running else "",
            "phase_detail": state.get("detail", "") if running else "",
            "phase_epoch": state.get("epoch") if running else None,
            "phase_loss": state.get("loss") if running else None,
            "plan_task": state.get("plan_task", "") if running else "",
            "plan_hypothesis": state.get("plan_hypothesis", "") if running else "",
            "plan_agent": state.get("plan_agent", "") if running else "",
            "latest_experiment": (self.experiment_entries[-1] if self.experiment_entries else None),
            "total_cost_usd": self.total_cost(),
            "snapshot_count": len(_snapshots(self.workspace)),
            "directive_pending": self.directive_pending(),
            "checkpoint_count": _checkpoint_count(self.workspace),
            "ts": time.time(),
        }

    def log_sources(self) -> list[tuple[str, Path]]:
        # 训练日志可能叫 training_output.log（launch 重定向）或 train.log
        # （agent 生成脚本自写），都纳入 Live Log。
        return [
            ("agent", self.project_dir / "autoresearcher_nodes.log"),
            ("console", self.project_dir / "agent_console.log"),
            ("train", self.workspace / "training_output.log"),
            ("train", self.workspace / "train.log"),
        ]

    def log_payload(self, lines: int = 200) -> dict:
        sources = {}
        sizes = {}
        for name, path in self.log_sources():
            sources[name] = _tail_lines(path, lines)
            sizes[name] = path.stat().st_size if path.exists() else 0
        return {"sources": sources, "sizes": sizes}

    def merged_log_tail(self, n: int = 100) -> list[str]:
        return _tail_merged(self.log_sources(), n)


_dash: Optional[_Dashboard] = None
_registry: Optional[ProjectRegistry] = None


def _get_dash() -> _Dashboard:
    if _dash is None:
        raise HTTPException(status_code=503, detail="dashboard 未初始化（缺 --project）")
    return _dash


def switch_project(proj_dir: Path) -> dict:
    """切换当前项目：停 agent（若在跑）+ 重建 _dash。"""
    global _dash
    if _manager.is_running():
        _manager.stop()
    _dash = _Dashboard(proj_dir, DEFAULT_CONFIG)
    return _dash.snapshot_payload()


# ═══════════════════════════════════════════════════════════════════════
# 请求体模型
# ═══════════════════════════════════════════════════════════════════════


class DirectiveBody(BaseModel):
    text: str = ""


class StartBody(BaseModel):
    max_cycles: Optional[int] = None


class RollbackBody(BaseModel):
    mode: str = "default"        # default | list | checkpoint | snapshot | epoch | best
    snapshot: Optional[str] = None
    checkpoint: Optional[str] = None   # epoch/best 回退的权重文件名
    max_cycles: Optional[int] = 1


class ProjectBody(BaseModel):
    path: str = ""


class ProjectNewBody(BaseModel):
    name: str = ""
    goal: str = ""
    success_criteria: str = ""
    constraints: str = ""
    dataset_path: str = ""
    model_path: str = ""
    literature: str = ""


class DraftBriefBody(BaseModel):
    goal: str = ""   # 一句话目标
    name: str = ""   # 可选项目名，帮助生成更贴合的草案


class AssetPreviewBody(BaseModel):
    path: str = ""
    kind: str = "dataset"   # "dataset" | "model"


class ApprovalRespondBody(BaseModel):
    decision: str = "approve"   # "approve" | "deny"


class CleanupBody(BaseModel):
    files: list[str] = []   # 要删除的文件名列表


class ProjectDeleteBody(BaseModel):
    path: str = ""
    confirm_name: str = ""   # 输入项目名二次确认


class SettingsBody(BaseModel):
    provider: str = ""
    model: str = ""
    base_url: str = ""
    api_key: str = ""


# ═══════════════════════════════════════════════════════════════════════
# 只读端点
# ═══════════════════════════════════════════════════════════════════════


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


@app.get("/api/settings")
async def api_settings():
    """读取当前 LLM 设置(不回显完整 key,只显示已配置/未配置)。"""
    dash = _get_dash()
    env_path = dash.project_dir / ".env"
    env_vars = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip()
    agent_cfg = {}
    cfg_path = dash.project_dir / DEFAULT_CONFIG
    if cfg_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            agent_cfg = (cfg.get("agent") or {}) or {}
        except Exception:
            pass
    key_env = agent_cfg.get("api_key_env", "") or "OPENAI_API_KEY"
    return {
        "provider": agent_cfg.get("provider", ""),
        "model": agent_cfg.get("model", ""),
        "base_url": agent_cfg.get("base_url", ""),
        "key_env": key_env,
        "key_configured": bool(env_vars.get(key_env) or os.environ.get(key_env)),
    }


@app.post("/api/settings")
async def api_settings_save(body: SettingsBody):
    """保存 LLM 设置:写项目 .env(load_dotenv 已支持,重启生效)
    + 更新 config.yaml 的 agent 段。"""
    dash = _get_dash()
    env_path = dash.project_dir / ".env"
    key_env = "OPENAI_API_KEY"
    existing = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                existing[k.strip()] = v.strip()
    if body.api_key.strip():
        # provider 预设决定默认 key 变量名
        preset_env = {
            "deepseek": "DEEPSEEK_API_KEY", "qwen": "DASHSCOPE_API_KEY",
            "kimi": "MOONSHOT_API_KEY", "glm": "ZHIPUAI_API_KEY",
        }.get(body.provider.strip(), "OPENAI_API_KEY")
        key_env = preset_env
        existing[key_env] = body.api_key.strip()
    lines = [f"{k}={v}" for k, v in sorted(existing.items())]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 更新 config.yaml agent 段(provider/model/base_url)
    cfg_path = dash.project_dir / DEFAULT_CONFIG
    try:
        import yaml
        cfg = {}
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        agent_cfg = cfg.setdefault("agent", {})
        if body.provider.strip():
            agent_cfg["provider"] = body.provider.strip()
        if body.model.strip():
            agent_cfg["model"] = body.model.strip()
        if body.base_url.strip():
            agent_cfg["base_url"] = body.base_url.strip()
        if body.api_key.strip():
            agent_cfg["api_key_env"] = key_env
        cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                            encoding="utf-8")
    except Exception as exc:
        return {"ok": False, "error": f"config.yaml 更新失败: {exc}",
                "env_written": True}
    return {"ok": True,
            "message": "设置已保存(.env + config.yaml)。agent 重启后生效。"}


@app.get("/api/files")
async def api_files(prefix: str = ""):
    """工作区文件树(排除敏感/缓存/二进制;只读浏览用)。"""
    dash = _get_dash()
    ws = dash.workspace
    skip_names = {".env", ".python_env.json", ".python_env.status",
                  "dry_run_log.json", ".last_launch.json", ".crash_context.json",
                  "PENDING_APPROVALS.md", ".agent.lock", ".cycle_counter"}
    skip_dirs = {".git", "__pycache__", ".trainenv", "checkpoints",
                 ".snapshots", "eval", "repos"}
    entries = []
    root = ws / prefix if prefix else ws
    if not root.exists() or not root.is_dir():
        return {"entries": [], "path": prefix}
    try:
        for p in sorted(root.iterdir()):
            if p.is_symlink():
                continue
            rel = p.relative_to(ws).as_posix()
            if p.is_dir():
                if p.name not in skip_dirs:
                    entries.append({"name": p.name, "path": rel, "dir": True})
            else:
                if p.name in skip_names or p.suffix.lower() in (
                        ".pth", ".pt", ".bin", ".db", ".png", ".jpg", ".pyc", ".log"):
                    continue
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                entries.append({"name": p.name, "path": rel, "dir": False,
                                "size": size})
    except OSError:
        pass
    return {"entries": entries, "path": prefix}


@app.get("/api/files/read")
async def api_files_read(path: str = ""):
    """文件内容预览(只读;越界/敏感文件拒绝)。"""
    dash = _get_dash()
    try:
        target = (dash.workspace / path).resolve()
        base = dash.workspace.resolve()
        if base not in target.parents or not target.is_file():
            raise HTTPException(status_code=400, detail="invalid path")
    except (OSError, ValueError):
        raise HTTPException(status_code=400, detail="invalid path")
    if target.name in {".env", "dry_run_log.json", "PENDING_APPROVALS.md",
                       ".python_env.json", ".python_env.status"}:
        raise HTTPException(status_code=403, detail="sensitive file")
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"path": path, "content": text[:100_000]}


@app.get("/api/state")
async def api_state():
    return _get_dash().snapshot_payload()


@app.get("/api/config")
async def api_config():
    """当前实验配置：实际使用的脚本 + 参数 + 优化器 + loss。"""
    dash = _get_dash()
    return _parse_training_config(dash)


@app.get("/api/projects")
async def api_projects():
    if _registry is None:
        raise HTTPException(status_code=503, detail="project registry 未初始化")
    return {
        "current": str(_get_dash().project_dir),
        "projects": _registry.discover(),
        "projects_root": str(_registry.projects_root),
    }


@app.post("/api/project")
async def api_project(body: ProjectBody):
    proj_dir = Path(body.path)
    if not (proj_dir / DEFAULT_CONFIG).exists():
        raise HTTPException(status_code=400, detail="project not found (no config.yaml)")
    return switch_project(proj_dir)


# ── 新项目草案生成（Draft & Confirm）──
@app.post("/api/draft/brief")
async def api_draft_brief(body: DraftBriefBody):
    """LLM 把一句话目标展开为 goal / success_criteria / constraints 草案。

    只返回三个字段，不写任何文件；最终 PROJECT_BRIEF.md 仍由
    /api/project/new → ProjectRegistry.create → _render_brief 生成。
    失败一律降级提示手动填写，不阻塞建项目流程。
    """
    goal = body.goal.strip()
    if not goal:
        raise HTTPException(status_code=400, detail="请先输入一句话研究目标")
    base_url, model, api_key = _draft_llm_settings()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="未配置 LLM API Key（需要 DEEPSEEK_API_KEY），请手动填写下方三个字段")
    try:
        # to_thread：LLM 最长 60s，不阻塞 SSE 事件循环
        raw = await asyncio.to_thread(
            _call_draft_llm, base_url, api_key, model,
            _draft_prompt(goal, body.name.strip()))
    except Exception as exc:
        logger.warning("draft LLM call failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"生成草案失败（LLM 调用出错），请手动填写下方字段: {exc}")
    try:
        parsed = _parse_draft_json(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"生成草案失败（LLM 返回格式异常），请手动填写下方字段: {exc}")
    return {"ok": True, **parsed}


@app.post("/api/project/new")
async def api_project_new(body: ProjectNewBody):
    if _registry is None:
        raise HTTPException(status_code=503, detail="project registry 未初始化")
    if not body.name.strip():
        raise HTTPException(status_code=400, detail="name required")
    try:
        created = _registry.create(body.name, body.goal,
                                   body.success_criteria, body.constraints,
                                   body.dataset_path, body.model_path,
                                   body.literature)
    except (ValueError, FileExistsError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    state = switch_project(Path(created["path"]))
    return {**state, "created": created}


@app.post("/api/project/delete")
async def api_project_delete(body: ProjectDeleteBody):
    """删除项目（危险操作）：需输入项目目录名二次确认。"""
    if _registry is None:
        raise HTTPException(status_code=503, detail="project registry 未初始化")
    if not body.path or not body.confirm_name:
        raise HTTPException(status_code=400, detail="path and confirm_name required")
    proj_dir = Path(body.path).resolve()
    # 项目必须在 projects_root 内
    root = _registry.projects_root.resolve()
    if root not in proj_dir.parents:
        raise HTTPException(status_code=400, detail="project outside projects root")
    if not proj_dir.exists():
        raise HTTPException(status_code=404, detail="project not found")
    if proj_dir.name != body.confirm_name:
        raise HTTPException(status_code=400, detail=f"confirm_name mismatch (expected {proj_dir.name})")
    # 若正在跑该项目的 agent，先停
    if _manager.is_running() and _get_dash().project_dir == proj_dir:
        _manager.stop()
    try:
        shutil.rmtree(proj_dir)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"delete failed: {exc}")
    # 若删的是当前项目，切换到第一个剩余项目
    if _get_dash().project_dir == proj_dir:
        remaining = _registry.discover()
        if remaining:
            switch_project(Path(remaining[0]["path"]))
        else:
            global _dash
            _dash = None
    return {"ok": True, "deleted": proj_dir.name}


@app.post("/api/assets/preview")
async def api_assets_preview(body: AssetPreviewBody):
    """预览外部资产路径（不拷贝）。kind: dataset | model。"""
    if not body.path.strip():
        raise HTTPException(status_code=400, detail="path required")
    return _preview_asset(Path(body.path.strip()), body.kind)


@app.post("/api/assets/upload")
async def api_assets_upload(file: UploadFile = File(...)):
    """上传小文件（数据集/模型），存到 <workspace>/.user_inputs/。"""
    dash = _get_dash()
    in_dir = dash.workspace / ".user_inputs"
    in_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w.\-]", "_", file.filename or "upload.bin")
    dest = in_dir / safe_name
    # 流式写，避免大文件占内存
    size = 0
    with open(dest, "wb") as f:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            size += len(chunk)
    await file.close()
    return {
        "ok": True,
        "saved_path": str(dest),
        "filename": safe_name,
        "size_mb": round(size / 1e6, 2),
    }


@app.get("/api/approvals")
async def api_approvals():
    """列出 PENDING_APPROVALS.md 中的审批请求。"""
    dash = _get_dash()
    reqs = _parse_approvals(dash.workspace)
    return {"approvals": reqs, "pending": sum(1 for r in reqs if not r["decision"])}


@app.post("/api/approvals/{req_id}/respond")
async def api_approvals_respond(req_id: str, body: ApprovalRespondBody):
    """对指定审批请求写入 APPROVE/DENY（与 approval.py 兼容）。"""
    dash = _get_dash()
    if body.decision not in ("approve", "deny"):
        raise HTTPException(status_code=400, detail="decision must be 'approve' or 'deny'")
    ok = _respond_approval(dash.workspace, req_id, body.decision)
    if not ok:
        raise HTTPException(status_code=404, detail=f"approval request [{req_id}] not found")
    return {"ok": True, "id": req_id, "decision": body.decision}


@app.get("/api/cleanup/scan")
async def api_cleanup_scan():
    """扫描 workspace 的可清理辅助文件（用户确认后删除）。"""
    dash = _get_dash()
    return {"candidates": _scan_cleanup_candidates(dash.workspace),
            "protected": sorted(_CLEANUP_PROTECTED)}


@app.post("/api/cleanup")
async def api_cleanup(body: CleanupBody):
    """删除用户选中的辅助文件（二次安全校验：只删匹配模式的）。"""
    dash = _get_dash()
    import fnmatch
    deleted, skipped = [], []
    for fname in body.files:
        p = (dash.workspace / fname).resolve()
        # 安全校验：必须是 workspace 内、文件名匹配辅助模式、非受保护
        if dash.workspace.resolve() not in p.parents or p.parent != dash.workspace.resolve():
            skipped.append({"name": fname, "reason": "outside workspace"})
            continue
        if p.name in _CLEANUP_PROTECTED:
            skipped.append({"name": fname, "reason": "protected"})
            continue
        if not any(fnmatch.fnmatch(p.name, pat) for pat in _CLEANUP_PATTERNS):
            skipped.append({"name": fname, "reason": "not auxiliary pattern"})
            continue
        try:
            p.unlink()
            deleted.append(fname)
        except OSError as exc:
            skipped.append({"name": fname, "reason": str(exc)})
    return {"ok": True, "deleted": deleted, "skipped": skipped}


@app.get("/api/audit")
async def api_audit(n: int = 20):
    dash = _get_dash()
    entries = dash.audit_entries
    return {"entries": entries[-max(0, n):], "summary": _audit_summary(entries)}


@app.get("/api/costs")
async def api_costs(days: int = 7):
    dash = _get_dash()
    entries = dash.cost_entries
    # 与 CostTracker.daily_summary 同语义
    cutoff = time.time() - max(0, days) * 86400
    recent = [e for e in entries if float(e.get("ts", 0)) >= cutoff]
    by_day: dict[str, float] = {}
    by_model: dict[str, float] = {}
    total = 0.0
    for e in recent:
        day = time.strftime("%Y-%m-%d", time.localtime(float(e.get("ts", 0))))
        cost = float(e.get("cost_usd", 0) or 0)
        by_day[day] = by_day.get(day, 0.0) + cost
        by_model[e.get("model", "unknown")] = by_model.get(e.get("model", "unknown"), 0.0) + cost
        total += cost
    return {
        "entries": entries,
        "total_cost_usd": dash.total_cost(),
        "daily": {
            "days": days,
            "total_calls": len(recent),
            "total_cost_usd": round(total, 4),
            "by_day": {k: round(v, 4) for k, v in sorted(by_day.items())},
            "by_model": {k: round(v, 4) for k, v in sorted(by_model.items())},
        },
    }


@app.get("/api/experiments")
async def api_experiments(n: int = 50):
    entries = _get_dash().experiment_entries
    return {"count": len(entries), "entries": entries[-max(0, n):]}


@app.get("/api/memory")
async def api_memory():
    dash = _get_dash()
    return {
        "memory_log": _read_text(dash.workspace / "MEMORY_LOG.md"),
        "insights": _read_text(dash.workspace / "INSIGHTS.md"),
        "dead_ends": _read_text(dash.workspace / "DEAD_ENDS.md"),
        "store": _memory_store(dash.workspace, dash.project_name),
    }


@app.get("/api/snapshots")
async def api_snapshots():
    return {"snapshots": _snapshots(_get_dash().workspace)}


def _parse_training_config(dash: "_Dashboard") -> dict:
    """解析训练脚本源码，提取优化器/loss/argparse 默认参数。

    从 workspace 里最新的 train*.py 解析（框架的 launch 校验要求脚本
    含 checkpoint 逻辑，所以选最新的、含 best_model 的脚本）。
    """
    # 找最新的训练脚本（优先含 checkpoint 逻辑的）
    scripts = [p for p in dash.workspace.glob("train*.py")]
    if not scripts:
        return {"script": None, "optimizer": None, "criterion": None,
                "params": {}, "note": "未找到训练脚本"}
    # 取修改时间最新的
    script = max(scripts, key=lambda p: p.stat().st_mtime)
    src = script.read_text(encoding="utf-8", errors="replace")

    # 优化器：optim.Adam / optim.SGD / optim.AdamW ...
    opt_m = re.search(r"optim\.(\w+)\s*\(", src)
    optimizer = opt_m.group(1) if opt_m else None

    # loss：nn.CrossEntropyLoss / F.mse_loss / ...
    crit_m = re.search(r"criterion\s*=\s*(?:nn\.|F\.)?(\w+)", src)
    criterion = crit_m.group(1) if crit_m else None

    # argparse 默认参数：--epochs/--batch-size/--lr/--dropout
    params = {}
    for flag, key in [("--epochs", "epochs"), ("--batch-size", "batch_size"),
                      ("--lr", "lr"), ("--dropout", "dropout"),
                      ("--device", "device")]:
        m = re.search(rf"'{flag}'.*?default=([^,\s]+)", src)
        if m:
            val = m.group(1).strip().strip("'\"")
            try:
                params[key] = float(val) if "." in val else int(val)
            except ValueError:
                params[key] = val

    # 结合 training_log.json 里实际运行的 config（含 resume 信息）
    log = dash.workspace / "checkpoints" / "training_log.json"
    actual = {}
    if log.exists():
        try:
            d = json.loads(log.read_text(encoding="utf-8", errors="replace"))
            actual = d.get("config", {}) or {}
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "script": script.name,
        "optimizer": optimizer,
        "criterion": criterion,
        "params": {**params, **actual},   # 实际运行参数优先
        "note": "从源码解析（优化器/loss）+ training_log.json（实际参数）",
    }


def _ckpt_params(dash: "_Dashboard") -> dict[int, dict]:
    """读 checkpoints/training_log.json 的每 epoch 参数（epoch → {lr, loss, acc, ...}）。"""
    log = dash.workspace / "checkpoints" / "training_log.json"
    out: dict[int, dict] = {}
    if not log.exists():
        return out
    try:
        data = json.loads(log.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return out
    for h in data.get("history", []) or []:
        ep = int(h.get("epoch", 0))
        if ep:
            out[ep] = {
                "lr": h.get("lr"),
                "train_loss": h.get("train_loss"),
                "train_acc": h.get("train_acc"),
                "test_loss": h.get("test_loss"),
                "test_acc": h.get("test_acc"),
            }
    return out


@app.get("/api/checkpoints")
async def api_checkpoints():
    """列出 workspace/checkpoints/ 下的权重（epoch 列表 + best_model）+ 每 epoch 参数。"""
    dash = _get_dash()
    ckpt_dir = dash.workspace / "checkpoints"
    params = _ckpt_params(dash)
    out = []
    if ckpt_dir.is_dir():
        for p in sorted(ckpt_dir.glob("*.pth")):
            try:
                mb = round(p.stat().st_size / 1e6, 2)
                mtime = p.stat().st_mtime
            except OSError:
                mb, mtime = 0, 0
            name = p.name
            kind = "best" if name == "best_model.pth" else "epoch"
            entry = {"name": name, "kind": kind, "size_mb": mb, "mtime": mtime}
            # 从文件名提取 epoch 号，关联 training_log.json 的参数
            import re as _re
            m = _re.search(r"checkpoint_epoch_(\d+)\.pth", name)
            if m:
                ep = int(m.group(1))
                entry["epoch"] = ep
                entry["params"] = params.get(ep, {})
            elif kind == "best":
                entry["epoch"] = None
                entry["params"] = {"best": True, "note": "训练最优权重"}
            out.append(entry)
    # 按 epoch 排序（best 放最前）
    out.sort(key=lambda e: (e["kind"] != "best", e.get("epoch") or 0))
    return {"checkpoints": out, "count": len(out), "dir": str(ckpt_dir)}


@app.get("/api/checkpoints/{name}/download")
async def api_checkpoints_download(name: str):
    """下载单个权重文件。"""
    dash = _get_dash()
    ckpt_dir = dash.workspace / "checkpoints"
    safe = re.sub(r"[^\w.\-]", "", name)  # 防路径穿越
    path = ckpt_dir / safe
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="checkpoint not found")
    from fastapi.responses import FileResponse
    return FileResponse(path, media_type="application/octet-stream",
                        filename=safe)


@app.get("/api/log")
async def api_log(lines: int = 200):
    return _get_dash().log_payload(lines=min(max(0, lines), 5000))


@app.get("/api/stream")
async def api_stream(request: Request):
    """SSE 实时流：快照轮询 + 事件日志断点续读（Last-Event-ID 重放）。

    - 每 2s 推送一次 state.json 快照（原子写后安全）
    - 同时尾随 events.jsonl：客户端携带 Last-Event-ID（= 上次看到的 seq）
      即从该 seq 之后重放全部事件，不丢事件（对齐 SSE 重放协议）
    - 心跳注释行保持连接活跃
    """
    async def _gen():
        dash = _get_dash()
        # Last-Event-ID: 浏览器 EventSource 断线重连时自动携带
        last_event_id = 0
        header = request.headers.get("Last-Event-ID") or request.headers.get("last-event-id")
        if header and str(header).strip().isdigit():
            last_event_id = int(header)
        try:
            yield "retry: 3000\n\n"
            while True:
                snap = dash.snapshot_payload()
                snap["log_tail"] = dash.merged_log_tail(100)
                yield f": hb {time.time()}\n\n"
                yield f"id: {last_event_id}\n"
                yield f"data: {json.dumps(snap, ensure_ascii=False)}\n\n"

                # 事件日志断点续读（本次连接内滚动）
                journal_path = dash.workspace / "events.jsonl"
                if journal_path.exists():
                    try:
                        journal = EventJournal(journal_path)
                        for ev in journal.read_from(after_seq=last_event_id, limit=50):
                            last_event_id = max(last_event_id, int(ev.get("seq", 0)))
                            yield f"event: journal\n"
                            yield f"id: {last_event_id}\n"
                            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    except Exception:
                        pass  # 事件日志损坏不影响快照流

                await asyncio.sleep(2)
        except asyncio.CancelledError:
            return

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.get("/api/deliverable")
async def api_deliverable(weights: str = "best"):
    """打包项目成果为 tar.gz：模型脚本 + 指标报告 + 权重（可选）。

    weights: best | all | none | <具体权重文件名>
    """
    dash = _get_dash()
    data, filename, summary = _build_deliverable(dash, weights=weights)
    from fastapi.responses import Response
    return Response(
        content=data,
        media_type="application/gzip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"',
                 "X-Deliverable-Summary": json.dumps(summary, ensure_ascii=False)},
    )


@app.get("/api/deliverable/info")
async def api_deliverable_info():
    """交付包内容预览（不下载，方便前端展示）。"""
    dash = _get_dash()
    _, _, summary = _build_deliverable(dash, weights="all")
    # 附上可选权重清单，供前端"选权重"用
    summary["weight_options"] = [
        {"value": "best", "label": "best_model.pth（最优，推荐）"},
        {"value": "all", "label": "全部 checkpoints"},
        {"value": "none", "label": "不含权重"},
    ]
    for c in _list_checkpoints(dash):
        if c["kind"] == "epoch":
            summary["weight_options"].append(
                {"value": c["name"], "label": f"{c['name']}"})
    return summary


# ═══════════════════════════════════════════════════════════════════════
# 控制端点
# ═══════════════════════════════════════════════════════════════════════


@app.post("/api/directive")
async def api_directive(body: DirectiveBody):
    dash = _get_dash()
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    path = dash.workspace / "HUMAN_DIRECTIVE.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {
        "ok": True, "path": str(path), "text": text,
        "note": "consumed by agent at next cycle/supervisor step",
    }


@app.post("/api/start")
async def api_start(body: StartBody):
    dash = _get_dash()
    if _manager.is_running():
        raise HTTPException(status_code=409, detail="agent already running",
                            headers={"X-Pid": str(_manager._proc.pid)})
    cmd = [os.environ.get("AGENT_PYTHON") or sys.executable,
           "-m", AGENT_MODULE,
           "--project", str(dash.project_dir), "--config", dash.config_name]
    if body.max_cycles:
        cmd += ["--max-cycles", str(int(body.max_cycles))]
    try:
        pid = _manager.start(dash.project_dir, dash.config_name,
                             max_cycles=body.max_cycles)
    except Exception as exc:
        return {"ok": False, "detail": "start failed", "error": str(exc)}
    return {"ok": True, "pid": pid, "cmd": cmd}


@app.post("/api/stop")
async def api_stop():
    return _manager.stop()


@app.post("/api/rollback")
async def api_rollback(body: RollbackBody):
    dash = _get_dash()
    mode = body.mode

    if mode == "list":
        return {"ok": True, "mode": "list", "snapshots": _snapshots(dash.workspace)}

    if mode not in ("default", "checkpoint", "snapshot", "epoch", "best"):
        raise HTTPException(status_code=400, detail=f"invalid mode: {mode}")

    if mode == "snapshot":
        snaps = _snapshots(dash.workspace)
        if not body.snapshot or not any(s["name"] == body.snapshot for s in snaps):
            raise HTTPException(status_code=400, detail="snapshot not found")

    # epoch/best 回退：需校验权重文件存在
    ckpt_name = None
    if mode in ("epoch", "best"):
        ckpt_dir = dash.workspace / "checkpoints"
        if mode == "best":
            ckpt_name = "best_model.pth"
        elif body.checkpoint:
            ckpt_name = body.checkpoint
        if not ckpt_name or not (ckpt_dir / ckpt_name).exists():
            raise HTTPException(status_code=400,
                                detail=f"checkpoint not found: {ckpt_name}")

    # ── stop → write → restart → poll ──
    if _manager.is_running():
        _manager.stop()

    if mode == "default":
        directive_text = "rollback"
    elif mode == "checkpoint":
        directive_text = "rollback --checkpoint"
    elif mode in ("epoch", "best"):
        # 用最新的 post 快照（含权重）作为回退目标；resume 指令随后由
        # RollbackHandler 写入 RESUME_DIRECTIVE.md
        snaps = _snapshots(dash.workspace)
        post_snaps = [s for s in snaps if "_post" in s["name"]]
        target_snap = (post_snaps[0] if post_snaps else snaps[0])["name"] if snaps else None
        if not target_snap:
            raise HTTPException(status_code=400, detail="no snapshot available for rollback")
        ckpt = ckpt_name or "best_model.pth"
        directive_text = (f"rollback --snapshot {target_snap}\n"
                          f"resume checkpoints/{ckpt}")
    else:
        directive_text = f"rollback --snapshot {body.snapshot}"
    (dash.workspace / "HUMAN_DIRECTIVE.md").write_text(directive_text, encoding="utf-8")

    console = dash.project_dir / "agent_console.log"
    start_pos = console.stat().st_size if console.exists() else 0

    try:
        pid = _manager.start(dash.project_dir, dash.config_name, max_cycles=1)
    except Exception as exc:
        return {"ok": False, "detail": "rollback restart failed", "error": str(exc)}

    manifest = dash.workspace / ".rollback_manifest.json"
    manifest_ts = manifest.stat().st_mtime if manifest.exists() else 0

    # 轮询回退结果：agent 启动即打印 "[rollback] <result>" 到 console
    # 异步端点内不得用 time.sleep 阻塞事件循环 → to_thread 委托线程池
    import asyncio as _asyncio
    result, polled = await _asyncio.to_thread(
        _poll_rollback_result, console, start_pos, manifest, manifest_ts, mode)

    if result is None:
        return {
            "ok": True, "mode": mode, "status": "pending",
            "agent_pid": pid, "polled_seconds": polled,
            "note": "回退结果未在 45s 内出现 — 请到 /api/log 查看",
        }
    return {
        "ok": True, "mode": mode,
        "result": result,
        "manifest": result if isinstance(result, dict) else None,
        "agent_pid": pid, "polled_seconds": polled,
    }


# ═══════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════


def main():
    global _dash, _registry
    parser = argparse.ArgumentParser(description="AutoResearcher Dashboard")
    parser.add_argument("--project", type=str, default=None, help="项目目录路径（缺省时自动发现 projects_dir 下第一个项目）")
    parser.add_argument("--projects-dir", type=str, default=str(REPO_ROOT / "examples"),
                        help="项目根目录（扫描/新建项目用）")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="配置文件（相对 project 目录）")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    _registry = ProjectRegistry(Path(args.projects_dir))
    if args.project:
        _dash = _Dashboard(Path(args.project), args.config)
    else:
        found = _registry.discover()
        if not found:
            raise SystemExit(
                f"projects_dir '{_registry.projects_root}' 下没有可用项目。"
                f"先 --project 指定一个，或用 UI 的「＋新建」创建。")
        _dash = _Dashboard(Path(found[0]["path"]), DEFAULT_CONFIG)

    import uvicorn
    print(f"AutoResearcher Dashboard → http://{args.host}:{args.port}")
    print(f"  project   = {_dash.project_dir}")
    print(f"  workspace = {_dash.workspace}")
    print(f"  projects_dir = {_registry.projects_root}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


# ═══════════════════════════════════════════════════════════════════════
# 前端（内联 HTML，零构建）
# ═══════════════════════════════════════════════════════════════════════

INDEX_HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AutoResearcher Dashboard</title>
<style>
  :root{
    --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#c9d1d9;
    --muted:#8b949e; --accent:#58a6ff; --ok:#3fb950; --err:#f85149; --warn:#d29922;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font:14px/1.5 ui-monospace,"Cascadia Mono",Consolas,monospace;padding-bottom:60px}
  .topbar{position:sticky;top:0;background:var(--panel);border-bottom:1px solid var(--border);padding:10px 16px;display:flex;gap:18px;align-items:center;flex-wrap:wrap;z-index:10}
  .topbar .title{font-weight:700;color:var(--accent)}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--err)}
  .dot.run{background:var(--ok);animation:blink 1.2s infinite}
  @keyframes blink{50%{opacity:.35}}
  .wrap{max-width:1100px;margin:0 auto;padding:16px}
  .panel{background:var(--panel);border:1px solid var(--border);border-radius:8px;margin:14px 0;padding:14px}
  .panel h2{font-size:13px;color:var(--accent);text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;border-bottom:1px solid var(--border);padding-bottom:6px}
  .kpi{display:flex;gap:24px;flex-wrap:wrap;font-size:13px}
  .kpi b{color:var(--text);font-weight:600}
  label{color:var(--muted);font-size:12px}
  textarea,input[type=text],input[type=number]{background:var(--bg);border:1px solid var(--border);border-radius:6px;color:var(--text);padding:8px;font:inherit;width:100%}
  textarea{min-height:64px;resize:vertical}
  .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
  button{background:#21262d;border:1px solid var(--border);border-radius:6px;color:var(--text);padding:8px 14px;font:inherit;cursor:pointer}
  button:hover:not(:disabled){border-color:var(--accent)}
  button:disabled{opacity:.45;cursor:not-allowed}
  button.ok{background:#1f3a22;border-color:var(--ok)}
  button.warn{background:#3a2b1f;border-color:var(--warn)}
  button.danger{background:#3a1f1f;border-color:var(--err)}
  button.small{padding:3px 8px;font-size:12px}
  pre{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:10px;font:12px/1.5 ui-monospace,Consolas,monospace;overflow:auto;max-height:420px;white-space:pre-wrap;word-break:break-all}
  table{width:100%;border-collapse:collapse;font-size:12.5px}
  th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--border);vertical-align:top}
  th{color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--panel)}
  .ok-c{color:var(--ok)} .err-c{color:var(--err)} .warn-c{color:var(--warn)} .mut{color:var(--muted)}
  #toast{position:fixed;bottom:16px;right:16px;max-width:480px;background:#21262d;border:1px solid var(--border);border-radius:8px;padding:10px 14px;font-size:12.5px;opacity:0;transition:opacity .25s;z-index:20}
  #toast.show{opacity:1}
  #toast.err{border-color:var(--err)}
  .stack{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  @media(max-width:760px){.stack{grid-template-columns:1fr}}
  .mt{margin-top:8px}
  select{background:var(--bg);border:1px solid var(--border);border-radius:6px;color:var(--text);padding:5px 8px;font:inherit}
  .modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;z-index:40}
  .modal-box{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:20px;width:min(520px,92vw);max-height:88vh;overflow:auto}
  .modal-box h2{color:var(--accent);font-size:14px;margin-bottom:12px}
  .modal-box label{display:block;margin-top:10px}
  .dropzone{border:1px dashed var(--border);border-radius:6px;padding:8px;margin-top:6px;text-align:center;color:var(--muted);font-size:12px;cursor:pointer;transition:border-color .2s}
  .dropzone.drag{border-color:var(--accent);color:var(--accent)}
  .asset-row{display:flex;gap:8px;align-items:center;margin-top:6px}
  .asset-row input{flex:1}
  .asset-pv{font-size:11.5px;margin-top:4px;color:var(--muted);word-break:break-all}
</style>
</head>
<body>
<div class="topbar">
  <span class="title">▚ AutoResearcher <span id="appr-badge" style="display:none;color:var(--warn)">⏳1</span></span>
  <span><label>Project</label>
    <select id="proj-select" style="width:auto"></select>
    <button class="small" id="btn-newproj">＋ 新建</button>
    <button class="small danger" id="btn-delproj">🗑 删除</button>
  </span>
  <span><span id="st-dot" class="dot"></span> <span id="st-status">—</span></span>
  <span><label>PID</label> <span id="st-pid">—</span></span>
  <span><label>Cycle</label> <span id="st-cycle">—</span></span>
  <span><label>Cost $</label> <span id="st-cost">—</span></span>
  <span><label>Snapshots</label> <span id="st-snaps">—</span></span>
  <span id="st-time" class="mut"></span>
</div>
<div class="topbar" id="plan-bar" style="display:none">
  <span class="title">📋 最新计划</span>
  <span id="plan-agent" class="mut"></span>
  <span id="plan-task"></span>
  <span id="plan-hyp" class="mut"></span>
</div>

<div class="wrap">

  <div class="panel">
    <h2>Control</h2>
    <label>Human Directive（一句话指令，下个 cycle 以最高优先级执行）</label>
    <textarea id="dtext" placeholder="例如：Try label smoothing 0.1 / 距离目标还差0.79%，试试调大epoch"></textarea>
    <div class="row mt">
      <button id="btn-send" class="ok">Send Directive</button>
      <span style="flex:1"></span>
      <label>max_cycles</label>
      <input type="number" id="maxcycles" value="1" min="1" style="width:70px">
      <button id="btn-start">▶ Start Agent</button>
      <button id="btn-stop" class="danger">■ Stop</button>
      <button id="btn-rb" class="warn">↩ Rollback</button>
      <button id="btn-rbcp" class="warn">↩ Checkpoint</button>
      <button id="btn-rblist">List Snapshots</button>
      <span style="flex:1"></span>
      <label>权重</label>
      <select id="dl-weights" style="width:auto">
        <option value="best">best_model.pth（最优）</option>
        <option value="all">全部 checkpoints</option>
        <option value="none">不含权重</option>
      </select>
      <button id="btn-dl">⬇ 下载交付包</button>
      <span id="dl-info" class="mut"></span>
    </div>
    <div class="mut mt" id="pending-note"></div>
  </div>

  <div class="panel" id="approvals-panel" style="display:none">
    <h2>⏳ 待审批 <span class="mut" id="approval-meta"></span></h2>
    <div id="approval-list"></div>
  </div>

  <div class="panel">
    <h2>📁 工作区文件 <span class="mut" id="files-meta"></span></h2>
    <div id="file-tree" class="mut">（点击刷新加载）</div>
    <button class="small mt" onclick="loadFiles()">刷新文件树</button>
    <pre id="file-preview" class="mut" style="max-height:260px;overflow:auto;white-space:pre-wrap;font-size:12px"></pre>
  </div>

  <div class="panel">
    <h2>⚙️ LLM 设置 <span class="mut" id="settings-meta"></span></h2>
    <div class="row"><label>Provider</label>
      <select id="set-provider" style="width:auto">
        <option value="deepseek">deepseek</option>
        <option value="openai">openai</option>
        <option value="qwen">qwen</option>
        <option value="kimi">kimi</option>
        <option value="glm">glm</option>
      </select>
      <label>Model</label><input id="set-model" style="width:150px" placeholder="deepseek-chat">
    </div>
    <div class="row mt"><label>Base URL</label><input id="set-baseurl" style="flex:1" placeholder="(默认用预设)"></div>
    <div class="row mt"><label>API Key</label><input id="set-key" type="password" style="flex:1" placeholder="(留空则不修改)"></div>
    <div class="row mt">
      <button class="ok" onclick="saveSettings()">保存设置</button>
      <span class="mut" id="set-status"></span>
    </div>
  </div>

  <div class="panel">
    <h2>🧹 Workspace 清理 <span class="mut" id="cleanup-meta"></span></h2>
    <div class="row">
      <button id="btn-cleanup-scan">扫描辅助文件</button>
      <button id="btn-cleanup-del" class="danger" disabled>删除选中的辅助文件</button>
    </div>
    <div id="cleanup-list" class="mut" style="margin-top:8px">点击扫描查看可清理的辅助文件</div>
  </div>

  <div id="newproj-modal" style="display:none">
    <div class="modal-overlay">
      <div class="modal-box">
        <h2>新建研究项目</h2>
        <label>项目名（目录名，自动转安全字符）</label>
        <input type="text" id="np-name" placeholder="如 mnist-vit">
        <div class="row mt">
          <input type="text" id="np-draft-goal" style="flex:1"
                 placeholder="一句话目标，如：在 CIFAR-100 上训练 ViT 达到 85% 准确率">
          <button class="small" id="btn-np-draft">✨ 生成草案</button>
        </div>
        <label>研究目标（草案生成后可再修改）</label>
        <textarea id="np-goal" placeholder="Train a ViT on CIFAR-100 to 85% accuracy"></textarea>
        <label>成功标准</label>
        <textarea id="np-success" placeholder="Test accuracy > 85%, training completes without errors"></textarea>
        <label>约束</label>
        <textarea id="np-constraints" placeholder="PyTorch, max 50 epochs, GPU 0"></textarea>
        <label>数据集（可选：本地路径或拖拽上传）</label>
        <div class="asset-row">
          <input type="text" id="np-dataset-path" placeholder="D:/data/cifar100">
          <button class="small" id="np-dataset-preview">预览</button>
        </div>
        <div class="dropzone" id="dz-dataset">拖拽文件到此处上传，或点击选择</div>
        <div class="asset-pv" id="np-dataset-pv"></div>
        <label>模型框架（可选：用户提供的模型代码/权重，agent 基于它迭代；支持 GitHub clone 目录或 .pth）</label>
        <div class="asset-row">
          <input type="text" id="np-model-path" placeholder="D:/my_model_project 或 D:/models/base.pth">
          <button class="small" id="np-model-preview">预览</button>
        </div>
        <div class="dropzone" id="dz-model">拖拽文件到此处上传，或点击选择</div>
        <div class="asset-pv" id="np-model-pv"></div>
        <label>文献（可选：每行一篇论文标题/arXiv ID/URL，idea_agent 直接分析，绕过外部搜索限流）</label>
        <textarea id="np-literature" placeholder="每行一篇，如：&#10;Batch Normalization: Accelerating Deep Network Training (arXiv:1502.03167)&#10;ResNet: Deep Residual Learning (arXiv:1512.03385)"></textarea>
        <div class="row mt">
          <button id="btn-np-create" class="ok">创建并切换到该项目</button>
          <button id="btn-np-cancel">取消</button>
        </div>
      </div>
    </div>
  </div>

  <div class="panel">
    <h2>实验配置 <span class="mut" id="cfg-meta"></span></h2>
    <div id="cfg-body" class="mut">（加载中…）</div>
  </div>

  <div class="panel">
    <h2>Live Log <span class="mut" id="log-meta"></span></h2>
    <pre id="log">（连接 SSE…）</pre>
  </div>

  <div class="panel">
    <h2>Decision Chain <span class="mut" id="audit-meta"></span></h2>
    <table>
      <thead><tr><th>Time</th><th>Actor</th><th>Action</th><th>Target</th><th>Result</th></tr></thead>
      <tbody id="audit-tbody"></tbody>
    </table>
  </div>

  <div class="stack">
    <div class="panel">
      <h2>Memory</h2>
      <label>MEMORY_LOG</label>
      <pre id="mem-log">（加载中…）</pre>
      <label class="mt">INSIGHTS</label>
      <pre id="mem-insights">（加载中…）</pre>
      <label class="mt">DEAD_ENDS</label>
      <pre id="mem-deadends">（加载中…）</pre>
    </div>
    <div class="panel">
      <h2>Store Insights <span class="mut" id="store-meta"></span></h2>
      <div id="store-list" class="mut">（加载中…）</div>
    </div>
  </div>

  <div class="stack">
    <div class="panel">
      <h2>Costs <span class="mut" id="cost-meta"></span></h2>
      <div class="kpi">
        <span>Total <b id="cost-total">—</b></span>
        <span>Today <b id="cost-today">—</b></span>
        <span>Calls <b id="cost-calls">—</b></span>
      </div>
      <table class="mt">
        <thead><tr><th>Time</th><th>Model</th><th>In/Out</th><th>Cost</th><th>Actor</th></tr></thead>
        <tbody id="cost-tbody"></tbody>
      </table>
    </div>
    <div class="panel">
      <h2>Experiments</h2>
      <table>
        <thead><tr><th>Cycle</th><th>Status</th><th>Hypothesis</th><th>Metrics</th><th>Conclusion</th></tr></thead>
        <tbody id="exp-tbody"></tbody>
      </table>
    </div>
  </div>

  <div class="panel">
    <h2>Snapshots <span class="mut" id="snap-meta"></span></h2>
    <table>
      <thead><tr><th>Name</th><th>Cycle</th><th>Size</th><th>Models</th><th>Created</th><th>Action</th></tr></thead>
      <tbody id="snap-tbody"></tbody>
    </table>
  </div>

  <div class="panel">
    <h2>Checkpoints <span class="mut" id="ckpt-meta"></span></h2>
    <table>
      <thead><tr><th>Weight</th><th>Kind</th><th>Size</th><th>Updated</th><th>Action</th></tr></thead>
      <tbody id="ckpt-tbody"></tbody>
    </table>
  </div>

</div>

<div id="toast"></div>

<script>
const $ = id => document.getElementById(id);
let toastTimer = null;
function toast(msg, isErr){
  const t = $('toast');
  t.textContent = msg;
  t.classList.toggle('err', !!isErr);
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(()=>t.classList.remove('show'), 4000);
}
async function api(path, opts){
  const r = await fetch(path, opts);
  let data = null;
  try{ data = await r.json(); }catch(e){}
  if(!r.ok){
    const d = (data && data.detail) ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)) : ('HTTP ' + r.status);
    throw new Error(d);
  }
  return data || {};
}
const fmtTime = ts => ts ? new Date(ts*1000).toLocaleString() : '—';
const esc = s => String(s ?? '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

// ── SSE 实时 ──
let lastDotRun = false;
function updateTopbar(s){
  $('st-dot').classList.toggle('run', !!s.running);
  let status = s.running ? 'RUNNING' : 'IDLE';
  if(s.running){
    status += ' · ' + (s.phase || '?');
    if(s.phase === 'supervisor' && s.phase_next) status += ' → 下一步: ' + s.phase_next;
    if(s.phase === 'monitor'){
      const parts = [];
      if(s.phase_epoch) parts.push('epoch '+s.phase_epoch);
      if(s.phase_loss !== undefined && s.phase_loss !== null) parts.push('loss '+parseFloat(s.phase_loss).toFixed(4));
      if(parts.length) status += ' (' + parts.join(', ') + ')';
    }
    if(s.pid) status += ' · pid='+s.pid;
  }else{
    status += ' · ' + (s.phase || 'idle');
  }
  $('st-status').textContent = status;
  $('st-pid').textContent = s.pid ?? '—';
  $('st-cycle').textContent = s.cycle ?? '—';
  $('st-cost').textContent = (s.total_cost_usd ?? 0).toFixed(5);
  $('st-snaps').textContent = s.snapshot_count ?? 0;
  $('st-time').textContent = fmtTime(s.ts);
  $('pending-note').textContent = s.directive_pending ? '⚠ 有未消费的 HUMAN_DIRECTIVE（agent 会在下一个 cycle 读取）' : '';
  // 最新 plan 显示
  const planBar = $('plan-bar');
  if(s.plan_task){
    planBar.style.display = '';
    $('plan-agent').textContent = '→ ' + (s.plan_agent || '?');
    $('plan-task').textContent = s.plan_task;
    $('plan-hyp').textContent = s.plan_hypothesis ? '（' + s.plan_hypothesis + '）' : '';
  } else {
    planBar.style.display = 'none';
  }
}
function connectSSE(){
  const es = new EventSource('/api/stream');
  es.onmessage = ev => {
    try{
      const s = JSON.parse(ev.data);
      updateTopbar(s);
      if(s.log_tail){
        const logEl = $('log');
        const atBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 40;
        logEl.textContent = s.log_tail.join('\n');
        $('log-meta').textContent = s.log_tail.length + ' lines';
        if(atBottom) logEl.scrollTop = logEl.scrollHeight;
      }
    }catch(e){}
  };
  es.onerror = () => { /* EventSource 自动重连 */ };
}

// ── 只读数据加载 ──
async function loadAudit(){
  try{
    const d = await api('/api/audit?n=30');
    $('audit-meta').textContent = (d.summary.total_entries||0)+' total · '+(d.summary.failures||0)+' failed';
    $('audit-tbody').innerHTML = d.entries.map(e =>
      `<tr><td class="mut">${fmtTime(e.timestamp)}</td>
           <td>${esc(e.actor)}</td><td>${esc(e.action)}</td>
           <td>${esc(String(e.target||'').slice(0,80))}</td>
           <td class="${e.result==='failed'?'err-c':e.result==='blocked'?'warn-c':'ok-c'}">${esc(e.result)}</td></tr>`
    ).join('') || '<tr><td colspan=5 class="mut">无记录</td></tr>';
  }catch(e){ toast('audit: '+e.message, true); }
}
async function loadCosts(){
  try{
    const d = await api('/api/costs?days=7');
    const today = d.daily.by_day ? Object.keys(d.daily.by_day).pop() : null;
    $('cost-total').textContent = '$'+d.total_cost_usd.toFixed(5);
    $('cost-today').textContent = today ? '$'+d.daily.by_day[today].toFixed(5) : '$0';
    $('cost-calls').textContent = d.daily.total_calls;
    $('cost-meta').textContent = d.daily.by_model ? Object.entries(d.daily.by_model).map(([m,c])=>m+' $'+c.toFixed(4)).join(' · ') : '';
    $('cost-tbody').innerHTML = d.entries.slice(-15).reverse().map(e =>
      `<tr><td class="mut">${fmtTime(e.ts)}</td><td>${esc(e.model)}</td>
           <td>${e.input_tokens}/${e.output_tokens}</td>
           <td>$${(e.cost_usd||0).toFixed(6)}</td><td class="mut">${esc(e.actor||'')} ${esc(e.action||'')}</td></tr>`
    ).join('') || '<tr><td colspan=5 class="mut">无记录</td></tr>';
  }catch(e){ toast('costs: '+e.message, true); }
}
async function loadExperiments(){
  try{
    const d = await api('/api/experiments?n=30');
    $('exp-tbody').innerHTML = d.entries.slice().reverse().map(e =>
      `<tr><td>${e.cycle}</td>
           <td class="${e.status==='failed'?'err-c':'ok-c'}">${esc(e.status)}</td>
           <td>${esc(String(e.hypothesis||'').slice(0,120))}</td>
           <td>${esc(JSON.stringify(e.metrics||{}))}</td>
           <td>${esc(String(e.conclusion||'').slice(0,120))}</td></tr>`
    ).join('') || '<tr><td colspan=5 class="mut">无记录</td></tr>';
  }catch(e){ toast('experiments: '+e.message, true); }
}
let lastRefreshTs = null;
async function loadMemory(){
  try{
    const d = await api('/api/memory');
    $('mem-log').textContent = d.memory_log || '（空）';
    $('mem-insights').textContent = d.insights || '（空）';
    $('mem-deadends').textContent = d.dead_ends || '（空）';
    const st = d.store || {};
    $('store-meta').textContent = (st.stats && st.stats.total_entries) ? st.stats.total_entries+' entries · '+(st.stats.db_size_kb||0)+' KB' : '';
    $('store-list').innerHTML = (st.insights||[]).map(i =>
      `<div style="padding:6px 0;border-bottom:1px solid var(--border)">${esc(i.text)} <span class="mut">· ${fmtTime(i.created_at)}</span></div>`
    ).join('') || '<div>（无跨项目语义记忆）</div>';
  }catch(e){ toast('memory: '+e.message, true); }
}
async function loadConfig(){
  try{
    const d = await api('/api/config');
    $('cfg-meta').textContent = d.note || '';
    if(!d.script){
      $('cfg-body').textContent = '（未找到训练脚本）';
      return;
    }
    const rows = [];
    rows.push(`<b>训练脚本</b>: <code>${esc(d.script)}</code>`);
    if(d.optimizer) rows.push(`<b>优化器</b>: ${esc(d.optimizer)}`);
    if(d.criterion) rows.push(`<b>Loss</b>: ${esc(d.criterion)}`);
    const p = d.params || {};
    const shown = ['epochs','batch_size','lr','dropout','device','resume_from','resume_epoch'];
    const pv = shown.filter(k => p[k] !== undefined)
      .map(k => `${k}=${esc(String(p[k]))}`).join(' · ');
    if(pv) rows.push(`<b>参数</b>: ${pv}`);
    $('cfg-body').innerHTML = rows.map(r => `<div style="padding:2px 0">${r}</div>`).join('');
  }catch(e){ $('cfg-body').textContent = '配置加载失败'; }
}
async function loadCheckpoints(){
  try{
    const d = await api('/api/checkpoints');
    $('ckpt-meta').textContent = d.count ? d.count+' weights · '+d.dir.split(/[\\/]/).pop() : '';
    $('ckpt-tbody').innerHTML = d.checkpoints.map(c => {
      const p = c.params || {};
      let paramTxt = '';
      if(c.kind === 'best'){
        paramTxt = '<span class="ok-c">★ 训练最优</span>';
      } else if(p.test_acc !== undefined){
        paramTxt = `acc ${p.test_acc} · loss ${p.train_loss !== undefined ? p.train_loss : '—'}`;
        if(p.lr) paramTxt += ` · lr ${p.lr}`;
      } else {
        paramTxt = '—';
      }
      return `<tr><td>${esc(c.name)}${c.kind==='best'?' <span class="ok-c">★best</span>':''}</td>
           <td>${c.kind}${c.epoch? ' '+c.epoch : ''}</td><td>${paramTxt}</td>
           <td>${c.size_mb} MB</td>
           <td class="mut">${fmtTime(c.mtime)}</td>
           <td>
             <button class="small warn ckpt-rb" data-name="${esc(c.name)}">↩ 回退</button>
             <button class="small ckpt-dl" data-name="${esc(c.name)}">⬇ 下载</button>
           </td></tr>`;
    }).join('') || '<tr><td colspan=6 class="mut">无权重（agent 训练后会保存到 checkpoints/）</td></tr>';
    document.querySelectorAll('.ckpt-rb').forEach(b => b.onclick = () => doRollbackCheckpoint(b.dataset.name));
    document.querySelectorAll('.ckpt-dl').forEach(b => b.onclick = () => {
      const a = document.createElement('a');
      a.href = '/api/checkpoints/' + encodeURIComponent(b.dataset.name) + '/download';
      a.download = ''; document.body.appendChild(a); a.click(); a.remove();
      toast('⬇ 正在下载 '+b.dataset.name);
    });
  }catch(e){ /* 非关键 */ }
}
async function doRollbackCheckpoint(name){
  if(!confirm('确认回退到权重 '+name+'？\n（将停止 agent → 恢复快照 → 从该权重续训）')) return;
  toast('↩ 回退到权重 '+name+'（停止→恢复→续训，最长 45s）…');
  try{
    const d = await api('/api/rollback', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({mode: name==='best_model.pth' ? 'best' : 'epoch', checkpoint: name})});
    if(d.status==='pending'){ toast('回退 pending: '+d.note, true); return; }
    const r = typeof d.result === 'string' ? d.result : JSON.stringify(d.result||'').slice(0,200);
    toast('✓ 已回退并从权重续训: '+r);
    setTimeout(refreshAll, 1500);
  }catch(e){ toast('rollback: '+e.message, true); }
}
async function loadDeliverableInfo(){
  try{
    const d = await api('/api/deliverable/info');
    const models = (d.models||[]).length;
    $('dl-info').textContent = models
      ? `${d.files.length} 文件 · ${models} 个权重 · ${d.total_mb} MB`
      : `${d.files.length} 文件 · 无权重(需在 BRIEF 要求保存)`;
    // 填充权重下拉（含分 epoch 权重）
    if(d.weight_options){
      const sel = $('dl-weights');
      const cur = sel.value;
      const epochOpts = d.weight_options.filter(o => !['best','all','none'].includes(o.value));
      const baseOpts = d.weight_options.filter(o => ['best','all','none'].includes(o.value));
      sel.innerHTML = baseOpts.map(o => `<option value="${o.value}">${o.label}</option>`).join('') +
        epochOpts.map(o => `<option value="${o.value}">${o.label}</option>`).join('');
      sel.value = cur && epochOpts.some(o=>o.value===cur) ? cur : 'best';
    }
  }catch(e){ /* 非关键，忽略 */ }
}
async function downloadDeliverable(){
  try{
    const w = $('dl-weights').value || 'best';
    const a = document.createElement('a');
    a.href = '/api/deliverable?weights=' + encodeURIComponent(w);
    a.download = '';
    document.body.appendChild(a);
    a.click();
    a.remove();
    toast('⬇ 交付包已开始下载（权重: '+w+'）');
  }catch(e){ toast('download: '+e.message, true); }
}
async function loadSnapshots(){
  try{
    const d = await api('/api/snapshots');
    $('snap-meta').textContent = d.snapshots.length + ' snapshots';
    $('snap-tbody').innerHTML = d.snapshots.map(s =>
      `<tr><td>${esc(s.name)}${s.latest?' <span class="ok-c">◀latest</span>':''}</td>
           <td>${s.cycle}</td><td>${(s.size_bytes/1024).toFixed(1)} KB</td>
           <td class="${s.models_total && s.models_ok<s.models_total ? 'err-c':'ok-c'}">${s.models_ok}/${s.models_total}${s.model_files.length? ' found':''}</td>
           <td class="mut">${s.created_at}</td>
           <td><button class="small warn rb-snap" data-name="${esc(s.name)}">↩ Rollback</button></td></tr>`
    ).join('') || '<tr><td colspan=6 class="mut">无快照（agent 每次 execute 前自动创建）</td></tr>';
    document.querySelectorAll('.rb-snap').forEach(b => b.onclick = () => doRollback('snapshot', b.dataset.name));
  }catch(e){ toast('snapshots: '+e.message, true); }
}
async function loadApprovals(){
  try{
    const d = await api('/api/approvals');
    const pending = d.approvals.filter(a => !a.decision);
    $('approval-meta').textContent = pending.length ? pending.length+' 个待审批' : '';
    $('appr-badge').style.display = pending.length ? '' : 'none';
    $('appr-badge').textContent = '⏳'+pending.length;
    $('approvals-panel').style.display = pending.length ? '' : 'none';
    $('approval-list').innerHTML = d.approvals.map(a => {
      const status = a.decision ? (a.decision==='approved' ? '<span class="ok-c">已批准</span>' : '<span class="err-c">已拒绝</span>') : '<span class="warn-c">待审批</span>';
      return `<div style="border:1px solid var(--border);border-radius:6px;padding:10px;margin-bottom:8px">
        <div><b>[${a.id}]</b> ${esc(a.action)} <span class="mut">risk: ${esc(a.risk)}</span> ${status}</div>
        <div class="mut">cost: $${a.cost||'0'} · ${esc(a.detail||'')}</div>
        <div class="mut">${esc(a.time||'')}</div>
        ${!a.decision ? `<div class="row mt">
          <button class="small ok" data-appr="${a.id}" data-dec="approve">✓ 批准</button>
          <button class="small danger" data-appr="${a.id}" data-dec="deny">✗ 拒绝</button>
        </div>` : ''}
      </div>`;
    }).join('') || '<div class="mut">无审批请求</div>';
    document.querySelectorAll('[data-appr]').forEach(btn =>
      btn.onclick = () => respondApproval(btn.dataset.appr, btn.dataset.dec));
  }catch(e){ /* 非关键 */ }
}
async function respondApproval(id, decision){
  try{
    await api(`/api/approvals/${id}/respond`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({decision})});
    toast(decision==='approve' ? '✓ 已批准 '+id : '✗ 已拒绝 '+id);
    await loadApprovals();
  }catch(e){ toast('approval: '+e.message, true); }
}

// ── 工作区文件树(只读浏览)──
let filePrefix = '';
async function loadFiles(){
  try{
    const d = await api('/api/files?prefix='+encodeURIComponent(filePrefix));
    const tree = $('file-tree');
    tree.innerHTML = '';
    if(filePrefix){
      const parent = filePrefix.split('/').slice(0,-1).join('/');
      tree.innerHTML += `<div style="cursor:pointer" onclick="filePrefix='${parent}';loadFiles()">⬆ ..</div>`;
    }
    d.entries.forEach(e => {
      if(e.dir){
        tree.innerHTML += `<div style="cursor:pointer" onclick="filePrefix='${e.path}';loadFiles()">📁 ${esc(e.name)}/</div>`;
      }else{
        tree.innerHTML += `<div style="cursor:pointer" data-file="${esc(e.path)}">📄 ${esc(e.name)} <span class="mut">(${e.size}B)</span></div>`;
      }
    });
    document.querySelectorAll('[data-file]').forEach(el =>
      el.onclick = () => previewFile(el.dataset.file));
    $('files-meta').textContent = '· ' + filePrefix || '';
  }catch(e){ /* 非关键 */ }
}
async function previewFile(path){
  try{
    const d = await api('/api/files/read?path='+encodeURIComponent(path));
    $('file-preview').textContent = d.content.slice(0, 4000);
  }catch(e){ $('file-preview').textContent = '读取失败: '+e.message; }
}
// ── LLM 设置 ──
async function loadSettings(){
  try{
    const d = await api('/api/settings');
    if(d.provider) $('set-provider').value = d.provider;
    if(d.model) $('set-model').value = d.model;
    if(d.base_url) $('set-baseurl').value = d.base_url;
    $('set-status').textContent = d.key_configured ? '✓ Key 已配置('+d.key_env+')' : '⚠ Key 未配置('+d.key_env+')';
  }catch(e){ /* 非关键 */ }
}
async function saveSettings(){
  try{
    const body = {
      provider: $('set-provider').value,
      model: $('set-model').value.trim(),
      base_url: $('set-baseurl').value.trim(),
      api_key: $('set-key').value.trim(),
    };
    const d = await api('/api/settings', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    toast(d.message || '设置已保存');
    $('set-key').value = '';
    await loadSettings();
  }catch(e){ toast('settings: '+e.message, true); }
}
async function refreshAll(){
  lastRefreshTs = new Date();
  await Promise.allSettled([loadConfig(), loadAudit(), loadCosts(), loadExperiments(), loadMemory(), loadSnapshots(), loadCheckpoints(), loadDeliverableInfo(), loadApprovals(), loadFiles(), loadSettings()]);
  // 各区块 meta 追加刷新时间戳，让用户看到数据是实时拉取的
  const stamp = '· ' + lastRefreshTs.toLocaleTimeString();
  ['cfg-meta','audit-meta','cost-meta','store-meta','snap-meta','ckpt-meta','approval-meta'].forEach(id => {
    const el = $(id);
    if(el) el.textContent = (el.textContent || '') + ' ' + stamp;
  });
}

// ── 控制动作 ──
async function sendDirective(){
  const text = $('dtext').value.trim();
  if(!text) return toast('请输入指令', true);
  $('btn-send').disabled = true;
  try{
    const d = await api('/api/directive', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})});
    toast('✓ 指令已写入: '+d.text);
    $('dtext').value = '';
  }catch(e){ toast('directive: '+e.message, true); }
  finally{ $('btn-send').disabled = false; }
}
async function doStart(){
  const mc = parseInt($('maxcycles').value, 10) || null;
  $('btn-start').disabled = true;
  try{
    const d = await api('/api/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({max_cycles:mc})});
    toast('✓ Agent 已启动 pid='+d.pid+' max_cycles='+(mc??'unlimited'));
  }catch(e){ toast('start: '+e.message, true); }
  finally{ $('btn-start').disabled = false; }
}
async function doStop(){
  $('btn-stop').disabled = true;
  try{
    const d = await api('/api/stop', {method:'POST'});
    toast(d.already_stopped ? 'Agent 本就没有运行' : '✓ Agent 已停止 (pid='+d.pid+')');
  }catch(e){ toast('stop: '+e.message, true); }
  finally{ $('btn-stop').disabled = false; }
}
async function doRollback(mode, snapName){
  const label = mode==='checkpoint' ? 'checkpoint' : (mode==='snapshot' ? snapName : 'default');
  if(!confirm('确认回退？模式: '+label+'\n（将停止当前 agent → 写入回退指令 → 重启 1 cycle）')) return;
  toast('↩ 回退进行中（停止→写指令→重启→轮询，最长 45s）…');
  try{
    const d = await api('/api/rollback', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({mode, snapshot:snapName||null})});
    if(d.status==='pending'){ toast('回退 pending: '+d.note, true); return; }
    const r = typeof d.result === 'string' ? d.result : (d.result ? JSON.stringify(d.result).slice(0,200) : '');
    toast('✓ 回退完成 ('+d.mode+'): '+r);
    setTimeout(refreshAll, 1500);
  }catch(e){ toast('rollback: '+e.message, true); }
}

// ── 项目切换 / 新建 ──
async function loadProjects(selectedPath){
  try{
    const d = await api('/api/projects');
    const sel = $('proj-select');
    sel.innerHTML = d.projects.map(p =>
      `<option value="${esc(p.path)}">${esc(p.name)}${p.has_workspace ? '' : ' ✱'}</option>`
    ).join('');
    const cur = selectedPath || d.current;
    sel.value = cur;
    $('proj-select').dataset.root = d.projects_root;
    return d;
  }catch(e){ toast('projects: '+e.message, true); return null; }
}
// ── workspace 清理 ──
let cleanupCandidates = [];
async function scanCleanup(){
  try{
    const d = await api('/api/cleanup/scan');
    cleanupCandidates = d.candidates || [];
    $('cleanup-meta').textContent = cleanupCandidates.length+' 个辅助文件 · 受保护 '+d.protected.length+' 项';
    $('cleanup-list').innerHTML = cleanupCandidates.length
      ? cleanupCandidates.map(c => `<label style="display:block;padding:2px 0"><input type="checkbox" data-cleanup="${esc(c.name)}"> ${esc(c.name)} <span class="mut">(${c.size_kb} KB)</span></label>`).join('')
      : '（没有可清理的辅助文件）';
    $('btn-cleanup-del').disabled = cleanupCandidates.length === 0;
    document.querySelectorAll('[data-cleanup]').forEach(cb => cb.onchange = () => {
      $('btn-cleanup-del').disabled = !document.querySelectorAll('[data-cleanup]:checked').length;
    });
  }catch(e){ toast('cleanup scan: '+e.message, true); }
}
$('btn-cleanup-scan').onclick = scanCleanup;
$('btn-cleanup-del').onclick = async () => {
  const files = [...document.querySelectorAll('[data-cleanup]:checked')].map(cb => cb.dataset.cleanup);
  if(!files.length) return toast('请选择要删除的文件', true);
  if(!confirm(`确认删除 ${files.length} 个辅助文件？\n${files.join('\n')}`)) return;
  try{
    const d = await api('/api/cleanup', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({files})});
    toast('✓ 已删除 '+d.deleted.length+' 个'+(d.skipped.length? '，跳过 '+d.skipped.length+' 个':''));
    await scanCleanup();
  }catch(e){ toast('cleanup: '+e.message, true); }
};

// ── 删除项目（二次确认）──
$('btn-delproj').onclick = async () => {
  const path = $('proj-select').value;
  const name = path ? path.split(/[\\/]/).pop() : '';
  if(!name) return toast('无当前项目', true);
  const typed = prompt(`⚠️ 危险操作！确认删除项目「${name}」？\n此操作将永久删除该项目目录（含权重/日志/记忆）。\n\n请输入项目名确认: ${name}`);
  if(typed === null) return;
  if(typed.trim() !== name) return toast('项目名不匹配，已取消', true);
  try{
    const d = await api('/api/project/delete', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({path, confirm_name:name})});
    toast('🗑 已删除项目: '+d.deleted);
    await loadProjects(); await refreshAll();
  }catch(e){ toast('delete project: '+e.message, true); }
};

$('proj-select').onchange = async () => {
  const path = $('proj-select').value;
  if(!path) return;
  toast('切换到项目…');
  try{
    await api('/api/project', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({path})});
    toast('✓ 已切换项目');
    await refreshAll();
  }catch(e){ toast('switch: '+e.message, true); await loadProjects(); }
};
function showNewProject(show){
  $('newproj-modal').style.display = show ? 'block' : 'none';
}
$('btn-newproj').onclick = () => showNewProject(true);
$('btn-np-cancel').onclick = () => showNewProject(false);
$('btn-np-draft').onclick = async () => {
  const goal = $('np-draft-goal').value.trim();
  if(!goal) return toast('请先输入一句话目标', true);
  const btn = $('btn-np-draft');
  btn.disabled = true; btn.textContent = '生成中…';
  try{
    const d = await api('/api/draft/brief', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({goal, name: $('np-name').value.trim()})});
    $('np-goal').value = d.goal;
    $('np-success').value = d.success_criteria;
    $('np-constraints').value = d.constraints;
    toast('✓ 草案已生成，请检查/修改后点击「创建」');
  }catch(e){
    toast('生成草案失败：'+e.message+'（可手动填写下方字段）', true);
  }finally{
    btn.disabled = false; btn.textContent = '✨ 生成草案';
  }
};
$('btn-np-create').onclick = async () => {
  const body = {
    name: $('np-name').value.trim(),
    goal: $('np-goal').value.trim(),
    success_criteria: $('np-success').value.trim(),
    constraints: $('np-constraints').value.trim(),
    dataset_path: $('np-dataset-path').value.trim(),
    model_path: $('np-model-path').value.trim(),
    literature: $('np-literature').value.trim(),
  };
  if(!body.name) return toast('请填写项目名', true);
  $('btn-np-create').disabled = true;
  try{
    const d = await api('/api/project/new', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    toast('✓ 已创建并切换到项目: '+d.created.name);
    showNewProject(false);
    ['np-name','np-draft-goal','np-goal','np-success','np-constraints','np-dataset-path','np-model-path','np-literature'].forEach(id => $(id).value = '');
    ['np-dataset-pv','np-model-pv'].forEach(id => $(id).textContent = '');
    await loadProjects();
    await refreshAll();
  }catch(e){ toast('new project: '+e.message, true); }
  finally{ $('btn-np-create').disabled = false; }
};

// ── 资产预览 / 上传 ──
async function previewAsset(kind){
  const input = $(kind==='dataset' ? 'np-dataset-path' : 'np-model-path');
  const out = $(kind==='dataset' ? 'np-dataset-pv' : 'np-model-pv');
  const path = input.value.trim();
  if(!path) return out.textContent = '（请先填路径或上传文件）';
  out.textContent = '预览中…';
  try{
    const d = await api('/api/assets/preview', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({path, kind})});
    if(d.kind==='dir') out.textContent = `目录 · ${d.entry_count} 项 · ${d.size_mb} MB · 前5个: ${d.sample.join(', ')}`;
    else out.textContent = `${d.name} · ${d.size_mb} MB${d.note ? ' · ' + d.note : ''}`;
  }catch(e){ out.textContent = '❌ '+e.message; }
}
function setupDropzone(dzId, inputId, outId, kind){
  const dz = $(dzId), input = $(inputId), out = $(outId);
  dz.onclick = () => { const fi = document.createElement('input'); fi.type='file'; fi.onchange = () => fi.files[0] && uploadTo(fi.files[0], kind, input, out); fi.click(); };
  ['dragover','dragenter'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave','drop'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('drag'); }));
  dz.addEventListener('drop', e => { const f = e.dataTransfer.files[0]; if(f) uploadTo(f, kind, input, out); });
}
async function uploadTo(file, kind, input, out){
  out.textContent = '上传中…';
  const fd = new FormData(); fd.append('file', file);
  try{
    const d = await api('/api/assets/upload', {method:'POST', body:fd});
    input.value = d.saved_path;
    out.textContent = `✓ 已上传 ${d.filename} (${d.size_mb} MB) → ${d.saved_path}`;
  }catch(e){ out.textContent = '❌ 上传失败: '+e.message; }
}
$('np-dataset-preview').onclick = () => previewAsset('dataset');
$('np-model-preview').onclick = () => previewAsset('model');
setupDropzone('dz-dataset','np-dataset-path','np-dataset-pv','dataset');
setupDropzone('dz-model','np-model-path','np-model-pv','model');

// ── 初始化 ──
$('btn-send').onclick = sendDirective;
$('btn-start').onclick = doStart;
$('btn-stop').onclick = doStop;
$('btn-rb').onclick = () => doRollback('default');
$('btn-rbcp').onclick = () => doRollback('checkpoint');
$('btn-rblist').onclick = loadSnapshots;
$('btn-dl').onclick = downloadDeliverable;
loadProjects();
connectSSE();
refreshAll();
setInterval(refreshAll, 15000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
