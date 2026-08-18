"""SqliteStore（LangGraph BaseStore SQLite 适配层）的单元测试。

覆盖审查发现的全部接口/语义问题：
  - 抽象方法 batch/abatch 已实现 → 可实例化
  - Item/SearchItem 用 datetime 构造，search 返回 SearchItem
  - namespace 前缀/后代匹配 + JSON 编码无分隔符碰撞
  - filter 操作符（$eq/$ne/$gt/$gte/$lt/$lte + 嵌套字段）
  - list_namespaces 全参数（prefix/suffix/max_depth/limit/offset）
  - batch 顺序语义（Get/Search 在 Put 前求值）+ 单事务回滚
  - 重启持久化 + 线程并发 + 上下文管理器
"""
import json
import shutil
import sqlite3
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.persistent_store import (
    SqliteStore,
    _encode_namespace,
    _decode_namespace,
    _is_prefix,
)


@pytest.fixture()
def store_dir(tmp_path: Path):
    yield tmp_path
    # WAL 文件可能残留，统一清理
    for suffix in ("-wal", "-shm"):
        p = tmp_path / f"test.db{suffix}"
        if p.exists():
            p.unlink()


@pytest.fixture()
def store(store_dir):
    s = SqliteStore(store_dir / "test.db")
    s.setup()
    yield s


# ── 实例化与生命周期 ──

def test_instantiates_not_abstract(store_dir):
    """回归：之前缺 batch/abatch → TypeError 抽象类无法实例化。"""
    s = SqliteStore(store_dir / "test.db")
    s.setup()  # 不应抛 TypeError
    assert not hasattr(s, "__abstractmethods__") or not s.__abstractmethods__


def test_context_manager(store_dir):
    with SqliteStore(store_dir / "ctx.db") as s:
        s.put(("episodic",), "k1", {"a": 1})
    # 退出后重新打开仍可读（持久化）
    s2 = SqliteStore(store_dir / "ctx.db")
    item = s2.get(("episodic",), "k1")
    assert item is not None and item.value == {"a": 1}


# ── CRUD 与类型 ──

def test_put_get_roundtrip(store):
    store.put(("episodic",), "exp_001", {"result": "acc 85%", "cycle": 1})
    item = store.get(("episodic",), "exp_001")
    assert item is not None
    assert item.value["result"] == "acc 85%"
    # Item 时间戳是 datetime（当前 BaseStore 契约）
    assert isinstance(item.created_at, datetime)
    assert item.created_at.tzinfo is not None


def test_get_missing_returns_none(store):
    assert store.get(("episodic",), "nope") is None


def test_put_preserves_created_at(store):
    store.put(("s",), "k", {"v": 1})
    first = store.get(("s",), "k")
    import time as _t
    _t.sleep(0.01)
    store.put(("s",), "k", {"v": 2})
    second = store.get(("s",), "k")
    assert second.value == {"v": 2}
    assert second.created_at == first.created_at  # 首写时间保留
    assert second.updated_at >= first.updated_at


def test_delete_returns_none_and_removes(store):
    store.put(("s",), "k", {"v": 1})
    result = store.delete(("s",), "k")
    assert result is None  # BaseStore 契约：delete 返回 None
    assert store.get(("s",), "k") is None
    store.delete(("s",), "k")  # 删除不存在的条目不抛错


# ── search 语义 ──

def test_search_returns_search_items(store):
    store.put(("project", "p1", "episodes"), "c1", {"cycle": 1})
    results = store.search(("project", "p1", "episodes"), limit=10)
    assert results and hasattr(results[0], "score")


def test_search_prefix_descendants(store):
    """BaseStore 契约：namespace_prefix 匹配后代。"""
    store.put(("project", "p1", "episodes"), "c1", {"cycle": 1})
    store.put(("project", "p1", "semantic"), "i1", {"text": "insight"})
    store.put(("project", "p2", "episodes"), "c1", {"cycle": 2})
    # 前缀 ("project", "p1") 应匹配两个后代 namespace
    results = store.search(("project", "p1"), limit=10)
    nss = {r.namespace for r in results}
    assert ("project", "p1", "episodes") in nss
    assert ("project", "p1", "semantic") in nss
    assert not any("p2" in ns for ns in nss)
    # 空前缀搜索全部
    assert len(store.search((), limit=10)) == 3


def test_search_query_substring(store):
    store.put(("semantic",), "a", {"text": "learning rate 0.001 helps"})
    store.put(("semantic",), "b", {"text": "batch size matters"})
    hits = store.search(("semantic",), query="learning rate", limit=10)
    assert [h.key for h in hits] == ["a"]


def test_search_filter_operators(store):
    store.put(("s",), "low", {"score": 1.0, "meta": {"tier": "x"}})
    store.put(("s",), "mid", {"score": 5.0, "meta": {"tier": "y"}})
    store.put(("s",), "high", {"score": 9.0, "meta": {"tier": "x"}})
    # 精确匹配（嵌套字段）
    hits = store.search(("s",), filter={"meta.tier": "x"}, limit=10)
    assert {h.key for h in hits} == {"low", "high"}
    # 比较操作符
    hits = store.search(("s",), filter={"score": {"$gte": 5.0}}, limit=10)
    assert {h.key for h in hits} == {"mid", "high"}
    hits = store.search(("s",), filter={"score": {"$lt": 5.0}}, limit=10)
    assert {h.key for h in hits} == {"low"}
    hits = store.search(("s",), filter={"score": {"$ne": 5.0}}, limit=10)
    assert {h.key for h in hits} == {"low", "high"}


def test_search_limit_offset(store):
    for i in range(10):
        store.put(("s",), f"k{i}", {"i": i})
    first = store.search(("s",), limit=3)
    assert len(first) == 3
    second = store.search(("s",), limit=3, offset=3)
    assert len(second) == 3
    assert first[0].key != second[0].key


# ── namespace 编码 ──

def test_namespace_encode_collision_free(store):
    """('a/b',) 与 ('a','b') 不得碰撞（JSON 编码无分隔符歧义）。"""
    store.put(("a/b",), "k", {"which": "single-label"})
    store.put(("a", "b"), "k", {"which": "two-labels"})
    assert store.get(("a/b",), "k").value["which"] == "single-label"
    assert store.get(("a", "b"), "k").value["which"] == "two-labels"
    # 搜索 ("a",) 前缀不得命中 ("a/b",)
    assert store.search(("a",), limit=10)[0].namespace == ("a", "b")


def test_encode_decode_roundtrip():
    assert _decode_namespace(_encode_namespace(("project", "p1", "episodes"))) == \
        ("project", "p1", "episodes")
    assert _decode_namespace(_encode_namespace(())) == ()


def test_is_prefix():
    assert _is_prefix(("a",), ("a", "b"))
    assert _is_prefix(("a", "b"), ("a", "b"))
    assert not _is_prefix(("a", "b"), ("a",))
    assert not _is_prefix(("a",), ("ab",))


# ── list_namespaces ──

def test_list_namespaces_prefix_suffix(store):
    store.put(("project", "p1", "episodes"), "c1", {"x": 1})
    store.put(("project", "p2", "semantic"), "i1", {"x": 1})
    store.put(("preferences",), "u1", {"x": 1})
    nss = store.list_namespaces(prefix=("project",))
    assert set(nss) == {("project", "p1", "episodes"), ("project", "p2", "semantic")}
    nss = store.list_namespaces(suffix=("episodes",))
    assert nss == [("project", "p1", "episodes")]


def test_list_namespaces_max_depth(store):
    store.put(("project", "p1", "episodes"), "c1", {"x": 1})
    store.put(("project", "p2", "semantic"), "i1", {"x": 1})
    nss = store.list_namespaces(prefix=("project",), max_depth=2)
    assert set(nss) == {("project", "p1"), ("project", "p2")}


def test_list_namespaces_pagination(store):
    for i in range(5):
        store.put(("project", f"p{i}"), "k", {"x": 1})
    page1 = store.list_namespaces(limit=2)
    page2 = store.list_namespaces(limit=2, offset=2)
    assert len(page1) == 2 and len(page2) == 2
    assert not set(page1) & set(page2)


# ── batch 语义 ──

def test_batch_mixed_ops_order(store):
    """对齐 InMemoryStore：batch 内所有 Get/Search 在 Put 应用之前基于旧快照求值。"""
    from langgraph.store.base import GetOp, PutOp, SearchOp
    ops = [
        GetOp(("s",), "k"),           # put 前 → None
        PutOp(("s",), "k", {"v": 1}),  # 结果恒为 None
        GetOp(("s",), "k"),           # 仍在 put 前求值 → None（快照语义）
        SearchOp(("s",), limit=10),   # 同样在 put 前 → 空
    ]
    results = store.batch(ops)
    assert results[0] is None
    assert results[1] is None
    assert results[2] is None
    assert results[3] == []
    # batch 结束后 put 已生效
    assert store.get(("s",), "k").value == {"v": 1}


def test_batch_rollback_on_error(store):
    """单事务：中途出错整体回滚，不留半提交状态。"""
    from langgraph.store.base import PutOp
    ops = [
        PutOp(("s",), "good", {"v": 1}),
        PutOp(("s",), "bad", object()),  # 不可 JSON 序列化 → 抛错
    ]
    with pytest.raises(TypeError):
        store.batch(ops)
    assert store.get(("s",), "good") is None  # 已回滚


def test_abatch_async(store):
    import asyncio
    from langgraph.store.base import PutOp
    results = asyncio.run(store.abatch([
        PutOp(("s",), "k", {"v": 1}),
    ]))
    assert results == [None]
    assert store.get(("s",), "k").value == {"v": 1}


# ── 持久化与并发 ──

def test_restart_persistence(store_dir):
    s1 = SqliteStore(store_dir / "test.db")
    s1.setup()
    s1.put(("semantic",), "insight_1", {"text": "persisted"})
    s2 = SqliteStore(store_dir / "test.db")
    s2.setup()
    item = s2.get(("semantic",), "insight_1")
    assert item is not None and item.value["text"] == "persisted"


def test_threaded_access(store):
    """LangGraph 节点在线程池执行 → 多线程并发读写不崩（check_same_thread=False）。"""
    errors = []

    def writer(i):
        try:
            for j in range(5):
                store.put(("threads",), f"t{i}_j{j}", {"i": i, "j": j})
                store.search(("threads",), limit=50)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(store.search(("threads",), limit=100)) == 40


def test_stats(store):
    store.put(("s1",), "a", {"x": 1})
    store.put(("s2",), "b", {"x": 1})
    stats = store.stats()
    assert stats["total_items"] == 2
    assert stats["by_namespace"][("s1",)] == 1
