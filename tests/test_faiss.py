"""CrossProjectStore FAISS 加速路径测试。

覆盖：
  - 无 faiss 环境 → 自动 fallback 暴力搜索（行为不回归）
  - mock faiss → build_index / search 走 FAISS 快路径，结果正确
  - 向量维度不匹配 → 跳过该行
"""
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.cross_project_memory import CrossProjectStore, EmbeddingProvider


class _FakeFaissIndex:
    """最小 IndexFlatIP 假实现（真实余弦语义，用于集成测试）。"""

    def __init__(self, dim: int):
        self.dim = dim
        self._vectors = []

    def add(self, matrix: np.ndarray):
        self._vectors = [row.copy() for row in matrix]

    def search(self, q: np.ndarray, k: int):
        qv = q[0]
        scores = []
        for v in self._vectors:
            scores.append(float(np.dot(qv, v)))  # 已归一化 → 余弦
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return np.array([[scores[i] for i in idxs]], dtype="float32"), \
            np.array([idxs], dtype="int64")


class _FakeFaissModule:
    IndexFlatIP = _FakeFaissIndex


class _TinyEmbedder(EmbeddingProvider):
    """固定维度确定性 embedder（FAISS 需要 dim 一致）。"""

    def __init__(self):
        self._model = None

    @property
    def dim(self) -> int:
        return 4

    def encode(self, text: str) -> list:
        v = np.zeros(4)
        for i, ch in enumerate(text[:4]):
            v[i] = ord(ch) / 1000.0
        norm = float(np.linalg.norm(v)) or 1.0
        return (v / norm).tolist()

    def encode_batch(self, texts: list) -> list:
        return [self.encode(t) for t in texts]


def _make_store(tmp_path: Path) -> CrossProjectStore:
    return CrossProjectStore(tmp_path / "mem.db", embedder=_TinyEmbedder())


def _seed(store: CrossProjectStore, text: str, project: str = "p1") -> None:
    store.add(text=text, project=project, namespace="semantic")


def test_fallback_without_faiss(tmp_path: Path):
    """无 faiss（当前环境）→ 暴力搜索路径正常，行为不回归。"""
    store = _make_store(tmp_path)
    assert store._faiss is None
    _seed(store, "learning rate tuning helped accuracy")
    _seed(store, "batch size experiments")
    results = store.search("learning rate", limit=5)
    assert results and "learning rate" in results[0]["text"]


def test_build_index_with_mock_faiss(tmp_path: Path):
    store = _make_store(tmp_path)
    _seed(store, "learning rate tuning helped accuracy")
    _seed(store, "batch size experiments")
    with patch.dict(sys.modules, {"faiss": _FakeFaissModule()}):
        # 模拟 import 成功后的状态
        store._faiss = _FakeFaissModule()
        ok = store.build_index()
    assert ok
    assert store._faiss_index is not None
    assert len(store._faiss_ids) == 2


def test_search_uses_faiss_path(tmp_path: Path):
    store = _make_store(tmp_path)
    _seed(store, "learning rate tuning helped accuracy")
    _seed(store, "batch size experiments")
    store._faiss = _FakeFaissModule()
    store.build_index()
    store._faiss_row_count = 2  # 防止 _ensure_index 重建
    results = store.search("learning rate", limit=1)
    assert len(results) == 1
    assert "learning rate" in results[0]["text"]
    assert results[0]["similarity"] > 0.9  # 强相似


def test_faiss_project_filter(tmp_path: Path):
    store = _make_store(tmp_path)
    _seed(store, "learning rate tuning helped accuracy", project="p1")
    _seed(store, "learning rate in pytorch", project="p2")
    store._faiss = _FakeFaissModule()
    store.build_index()
    store._faiss_row_count = 2
    # project 过滤：只返回 p1
    results = store.search("learning rate", limit=5, project="p1")
    assert len(results) == 1
    assert results[0]["project"] == "p1"


def test_dim_mismatch_row_skipped(tmp_path: Path):
    store = _make_store(tmp_path)
    store.add(text="good", project="p1", namespace="semantic")
    # 手工插入一条维度错误的 embedding（损坏数据）
    import sqlite3
    with sqlite3.connect(str(store.db_path)) as conn:
        conn.execute(
            "INSERT INTO memories (id, project, namespace, text, embedding, "
            "metadata, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            ("bad", "p1", "semantic", "bad vec",
             store._pack_embedding([1.0, 2.0]), "{}", 1.0, 1.0),
        )
    store._faiss = _FakeFaissModule()
    ok = store.build_index()
    assert ok
    assert len(store._faiss_ids) == 1  # 只有维度正确的行入索引
    assert store._faiss_ids[0] != "bad"


def test_rebuild_when_row_count_changes(tmp_path: Path):
    store = _make_store(tmp_path)
    _seed(store, "first")
    store._faiss = _FakeFaissModule()
    store.build_index()
    assert store._faiss_row_count == 1
    _seed(store, "second")  # 行数变化
    store._ensure_index()
    assert store._faiss_row_count == 2
    assert len(store._faiss_ids) == 2
