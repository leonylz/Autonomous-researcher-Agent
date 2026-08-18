"""
LangGraph Store 的 SQLite 持久化实现。

对齐 LangGraph 生产部署最佳实践：
  - 完整实现 BaseStore 抽象接口（batch / abatch 为核心，put/get/search/
    list_namespaces/delete 由基类 wrapper 驱动，签名与语义自动对齐）
  - SQLite 作为单机持久化后端（对标 PostgresStore，适用于单机部署）
  - 四类命名空间：episodic / semantic / procedural / preference
  - 重启不丢数据（替代 InMemoryStore）

设计要点（基于 langgraph-checkpoint 4.x BaseStore 源码逐项对齐）：
  - batch() 单事务、按序处理；Get/Search/List 在 Put 之前基于旧数据求值
    （与 InMemoryStore 语义一致），Put 最后应用；出错整体回滚
  - namespace 以 JSON 编码存储（无 '/' 分隔符歧义），前缀/后代匹配在
    Python 侧完成（数据量小，正确性优先）
  - Item / SearchItem 使用 datetime(UTC) 时间戳（当前 BaseStore 要求）
  - filter 支持精确匹配、嵌套字段与 $eq/$ne/$gt/$gte/$lt/$lte 操作符
  - WAL + busy_timeout + check_same_thread=False + RLock：
    LangGraph sync 节点在线程池中执行，单连接会触发 ProgrammingError
  - PutOp.value=None 表示删除（对齐 BaseStore.delete 语义，返回 None）

用法:
    from .persistent_store import SqliteStore
    store = SqliteStore(workspace / "langgraph_store.db")
    store.setup()
    with store:                       # __enter__ = setup，__exit__ = close
        ...
    graph = builder.compile(store=store)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger("autoresearcher.persistent_store")

# ── LangGraph BaseStore 接口（必须可用，否则本模块无意义）──
try:
    from langgraph.store.base import (  # type: ignore[import-untyped]
        BaseStore,
        GetOp,
        Item,
        ListNamespacesOp,
        MatchCondition,
        PutOp,
        SearchItem,
        SearchOp,
    )
    _HAS_LG_STORE = True
except ImportError:
    _HAS_LG_STORE = False

    # 极简兼容桩：仅当 langgraph 未安装时用于静态导入；运行时行为不受保证。
    class BaseStore:  # type: ignore[no-redef]
        supports_ttl = False

    class Item:  # type: ignore[no-redef]
        def __init__(self, value, key, namespace, created_at, updated_at):
            self.value = value
            self.key = key
            self.namespace = namespace
            self.created_at = created_at
            self.updated_at = updated_at

    class SearchItem(Item):  # type: ignore[no-redef]
        def __init__(self, namespace, key, value, created_at, updated_at, score=None):
            super().__init__(value, key, namespace, created_at, updated_at)
            self.score = score

    class GetOp:  # type: ignore[no-redef]
        def __init__(self, namespace, key, refresh_ttl=True):
            self.namespace, self.key, self.refresh_ttl = namespace, key, refresh_ttl

    class SearchOp:  # type: ignore[no-redef]
        def __init__(self, namespace_prefix, filter=None, limit=10, offset=0,
                     query=None, refresh_ttl=True):
            self.namespace_prefix, self.filter = namespace_prefix, filter
            self.limit, self.offset, self.query = limit, offset, query

    class PutOp:  # type: ignore[no-redef]
        def __init__(self, namespace, key, value, index=None, ttl=None):
            self.namespace, self.key, self.value = namespace, key, value
            self.index, self.ttl = index, ttl

    class ListNamespacesOp:  # type: ignore[no-redef]
        def __init__(self, match_conditions=None, max_depth=None, limit=100, offset=0):
            self.match_conditions, self.max_depth = match_conditions, limit
            self.limit, self.offset = limit, offset

    class MatchCondition:  # type: ignore[no-redef]
        def __init__(self, match_type, path):
            self.match_type, self.path = match_type, path


def _encode_namespace(namespace: tuple[str, ...]) -> str:
    """JSON 编码命名空间。无分隔符歧义：('a/b',) 与 ('a','b') 不碰撞。"""
    return json.dumps(list(namespace), ensure_ascii=False)


def _decode_namespace(encoded: str) -> tuple[str, ...]:
    try:
        data = json.loads(encoded)
        return tuple(str(x) for x in data) if isinstance(data, list) else ()
    except (json.JSONDecodeError, TypeError):
        return ()


def _is_prefix(prefix: tuple[str, ...], full: tuple[str, ...]) -> bool:
    """prefix 是 full 的前缀（后代匹配，含相等）。"""
    if len(prefix) > len(full):
        return False
    return all(p == f for p, f in zip(prefix, full))


def _ns_matches(match_type: str, pattern: tuple, ns: tuple) -> bool:
    """namespace 匹配规则：prefix / suffix，pattern 元素支持 '*' 通配。"""
    def _seg_match(p: str, s: str) -> bool:
        return p == "*" or p == s

    if match_type == "prefix":
        if len(pattern) > len(ns):
            return False
        return all(_seg_match(p, s) for p, s in zip(pattern, ns))
    if match_type == "suffix":
        if len(pattern) > len(ns):
            return False
        return all(_seg_match(p, s) for p, s in zip(pattern, reversed(ns)))
    return False


def _dt_from_epoch(epoch: float) -> datetime:
    """epoch → UTC datetime（BaseStore Item 要求 datetime）。"""
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def _get_path(value: Any, path: str) -> Any:
    """按 'a.b.c' 路径取嵌套字段。"""
    cur = value
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
    return cur


def _matches_filter(value: dict, filter: Optional[dict]) -> bool:
    """filter 语义（对齐 InMemoryStore）：AND 条件，支持 $eq/$ne/$gt/$gte/$lt/$lte。"""
    if not filter:
        return True
    for key, cond in filter.items():
        actual = _get_path(value, key)
        if isinstance(cond, dict) and any(
            k in cond for k in ("$eq", "$ne", "$gt", "$gte", "$lt", "$lte")
        ):
            for op, target in cond.items():
                if op == "$eq" and not (actual == target):
                    return False
                if op == "$ne" and not (actual != target):
                    return False
                if op == "$gt" and not (actual is not None and actual > target):
                    return False
                if op == "$gte" and not (actual is not None and actual >= target):
                    return False
                if op == "$lt" and not (actual is not None and actual < target):
                    return False
                if op == "$lte" and not (actual is not None and actual <= target):
                    return False
        else:
            # 直接值 → 精确匹配
            if actual != cond:
                return False
    return True


def _query_hit(value: dict, key: str, query: str) -> bool:
    """无向量索引的 query 兜底：value/key 子串匹配（Python 侧，无 SQL 通配符问题）。"""
    q = query.lower()
    if q in key.lower():
        return True
    return q in json.dumps(value, ensure_ascii=False).lower()


class SqliteStore(BaseStore if _HAS_LG_STORE else object):
    """SQLite 持久化 LangGraph Store。

    支持四类语义记忆命名空间：
      - ("project", <name>, "episodes")    — 经验记忆（历史实验、决策链）
      - ("project", <name>, "semantic")    — 语义记忆（领域知识、论文洞察）
      - ("procedural",)                    — 程序性记忆（成功配置）
      - ("preferences",)                   — 偏好记忆（用户画像）
    """

    # BaseStore 不支持 TTL（对齐 PostgresStore 之外的简化实现）
    supports_ttl = False

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()

    # ── 生命周期 ──
    def setup(self):
        """建表 + WAL。对齐 PostgresSaver.setup() 约定：首次使用前必须调用。"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS store_items (
                        namespace   TEXT    NOT NULL,   -- JSON 编码，无分隔符歧义
                        key         TEXT    NOT NULL,
                        value_json  TEXT    NOT NULL DEFAULT '{}',
                        created_at  REAL    NOT NULL,   -- epoch（秒）
                        updated_at  REAL    NOT NULL,
                        PRIMARY KEY (namespace, key)
                    )
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_store_updated "
                    "ON store_items(updated_at)"
                )
                conn.commit()
            finally:
                conn.close()
        logger.info("SqliteStore setup complete: %s", self.db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,  # LangGraph 节点跑在线程池
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def close(self):
        """关闭所有连接（幂等）。"""
        # 连接为每操作创建，无长期持有的连接需要关闭；保留接口以对齐生命周期约定。
        logger.debug("SqliteStore close (per-op connections, nothing to release)")

    def __enter__(self) -> "SqliteStore":
        self.setup()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── BaseStore 核心抽象方法（wrapper 的 put/get/search/delete 都走这里）──
    def batch(self, ops: Iterable) -> list:
        """按序执行一批操作，单事务，出错回滚。

        对齐 InMemoryStore 语义：Get/Search/List 基于 Put 之前的存储快照求值，
        Put 最后统一应用。
        """
        ops = list(ops)
        with self._lock:
            conn = self._connect()
            try:
                results: list = []
                pending_puts: list = []
                for op in ops:
                    if isinstance(op, GetOp):
                        results.append(self._get(conn, op.namespace, op.key))
                    elif isinstance(op, SearchOp):
                        results.append(self._search(conn, op))
                    elif isinstance(op, ListNamespacesOp):
                        results.append(self._list_namespaces(conn, op))
                    elif isinstance(op, PutOp):
                        results.append(None)  # InMemoryStore: Put 结果恒为 None
                        pending_puts.append(op)
                    else:
                        raise TypeError(f"unsupported store op: {type(op).__name__}")

                # Put 最后应用（单事务）
                for op in pending_puts:
                    self._apply_put(conn, op)
                conn.commit()
                return results
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    async def abatch(self, ops: Iterable) -> list:
        """异步批处理：委托线程池执行同步 batch，不阻塞事件循环。"""
        return await asyncio.to_thread(self.batch, ops)

    # ── 内部实现 ──
    @staticmethod
    def _row_to_item(row: sqlite3.Row, namespace: tuple[str, ...]) -> Item:
        return Item(
            value=json.loads(row["value_json"]),
            key=row["key"],
            namespace=namespace,
            created_at=_dt_from_epoch(row["created_at"]),
            updated_at=_dt_from_epoch(row["updated_at"]),
        )

    def _get(self, conn: sqlite3.Connection, namespace: tuple[str, ...],
             key: str) -> Optional[Item]:
        ns_enc = _encode_namespace(namespace)
        row = conn.execute(
            "SELECT namespace, key, value_json, created_at, updated_at "
            "FROM store_items WHERE namespace=? AND key=?",
            (ns_enc, key),
        ).fetchone()
        return self._row_to_item(row, namespace) if row else None

    def _search(self, conn: sqlite3.Connection, op: SearchOp) -> list[SearchItem]:
        rows = conn.execute(
            "SELECT namespace, key, value_json, created_at, updated_at "
            "FROM store_items ORDER BY updated_at DESC"
        ).fetchall()

        out: list[SearchItem] = []
        for row in rows:
            ns = _decode_namespace(row["namespace"])
            if not _is_prefix(op.namespace_prefix, ns):
                continue
            value = json.loads(row["value_json"])
            if op.query and not _query_hit(value, row["key"], op.query):
                continue
            if not _matches_filter(value, op.filter):
                continue
            out.append(SearchItem(
                namespace=ns,
                key=row["key"],
                value=value,
                created_at=_dt_from_epoch(row["created_at"]),
                updated_at=_dt_from_epoch(row["updated_at"]),
                score=None,  # 无向量索引；query 为子串匹配
            ))
            if len(out) >= op.offset + op.limit:
                break
        return out[op.offset:op.offset + op.limit]

    def _list_namespaces(self, conn: sqlite3.Connection,
                         op: ListNamespacesOp) -> list[tuple[str, ...]]:
        seen: dict[tuple[str, ...], None] = {}
        for row in conn.execute(
            "SELECT DISTINCT namespace FROM store_items"
        ).fetchall():
            ns = _decode_namespace(row["namespace"])
            for cond in (op.match_conditions or ()):
                if not _ns_matches(cond.match_type, cond.path, ns):
                    break
            else:
                seen[ns] = None

        nss = sorted(seen.keys())
        if op.max_depth is not None:
            nss = [ns[:op.max_depth] for ns in nss]
            # 截断后去重
            nss = list(dict.fromkeys(nss))
        return nss[op.offset:op.offset + op.limit]

    def _apply_put(self, conn: sqlite3.Connection, op: PutOp):
        if op.value is None:
            # PutOp.value=None → 删除（BaseStore.delete 语义）
            conn.execute(
                "DELETE FROM store_items WHERE namespace=? AND key=?",
                (_encode_namespace(op.namespace), op.key),
            )
            return
        ns_enc = _encode_namespace(op.namespace)
        now = time.time()
        conn.execute(
            """INSERT INTO store_items (namespace, key, value_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(namespace, key) DO UPDATE SET
                   value_json=excluded.value_json,
                   updated_at=excluded.updated_at
               """,  # created_at 保留首次写入时间（对齐 InMemoryStore 语义）
            (ns_enc, op.key, json.dumps(op.value, ensure_ascii=False), now, now),
        )

    # ── 诊断 ──
    def stats(self) -> dict:
        with self._lock:
            conn = self._connect()
            try:
                total = conn.execute("SELECT COUNT(*) FROM store_items").fetchone()[0]
                by_ns_rows = conn.execute(
                    "SELECT namespace, COUNT(*) AS cnt FROM store_items GROUP BY namespace"
                ).fetchall()
            finally:
                conn.close()
        by_ns = {_decode_namespace(r["namespace"]): r["cnt"] for r in by_ns_rows}
        return {
            "db_path": str(self.db_path),
            "db_size_kb": round(self.db_path.stat().st_size / 1024, 1)
            if self.db_path.exists() else 0,
            "total_items": total,
            "by_namespace": by_ns,
        }
