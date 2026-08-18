"""
Cross-project semantic memory with persistent storage.

Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings,
SQLite for persistence, cosine similarity for retrieval.
Graceful fallback to TF-IDF-like encoding if sentence-transformers
is not installed.

面试价值：RAG 三大组件（embedding + vector store + retrieval）的完整实现。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger("autoresearcher.cross_project")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingProvider:
    """Embedding provider with graceful fallback.

    Tries sentence-transformers first; falls back to a lightweight
    hash-based bag-of-words if the package isn't installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info("Loaded sentence-transformer: %s", self.model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; "
                "falling back to hash-based bag-of-words. "
                "Install with: pip install sentence-transformers"
            )
        except Exception:
            logger.warning(
                "Failed to load %s; using hash-based fallback.",
                self.model_name,
            )

    @property
    def dim(self) -> int:
        if self._model:
            return self._model.get_sentence_embedding_dimension()
        return 256  # fallback dimension

    def encode(self, text: str) -> list[float]:
        """Encode a single text to an embedding vector."""
        if self._model:
            return self._model.encode(
                text, normalize_embeddings=True
            ).tolist()
        return self._fallback_encode(text)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts."""
        if self._model:
            return self._model.encode(
                texts, normalize_embeddings=True
            ).tolist()
        return [self._fallback_encode(t) for t in texts]

    # ------------------------------------------------------------------
    # Fallback: lightweight hash-based bag-of-words (no dependencies)
    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_encode(text: str) -> list[float]:
        """Character trigram hashing → fixed-size vector.

        More overlap-friendly than word-level hashing: "learning_rate"
        and "learning rate" share the trigram "lea"/"ear"/"arn"/"rni"/"nin"/"ing".
        """
        import hashlib

        text_lower = text.lower()
        # Character trigrams (robust to spacing/punctuation differences)
        trigrams = set()
        for i in range(len(text_lower) - 2):
            trigrams.add(text_lower[i:i + 3])
        vec = [0.0] * 256
        for tg in trigrams:
            idx = int(hashlib.md5(tg.encode()).hexdigest(), 16) % 256
            vec[idx] += 1.0
        # L2-normalize
        norm = (sum(x * x for x in vec)) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


# ═══════════════════════════════════════════════════════════════════
# Persistent SQLite store
# ═══════════════════════════════════════════════════════════════════

class CrossProjectStore:
    """Persistent semantic memory with cross-project search.

    Each memory entry is stored as:
      - id, project, namespace, text, embedding (BLOB), metadata (JSON), timestamps

    Search flow（FAISS 优先，暴力搜索兜底）:
      1. Encode query → query_vec
      2. FAISS 可用 → ANN 检索 top-K（IndexFlatIP，精确余弦）
      3. FAISS 不可用 → 加载候选行，Python 逐条余弦（fallback 文化）
      4. 按 project / namespace 过滤后返回 top-K

    FAISS 索引惰性重建：记录构建时的行数，DB 行数变化才重建（写入低频，
    行数对比 O(1)）。faiss 为可选依赖（pip install faiss-cpu）。
    """

    def __init__(self, db_path: Path, embedder: Optional[EmbeddingProvider] = None):
        self.db_path = db_path
        self.embedder = embedder or EmbeddingProvider()
        self._init_db()

        # ── FAISS 可选加速（缺失 → 自动 fallback 暴力搜索）──
        self._faiss = None
        self._faiss_index = None
        self._faiss_ids: list[str] = []      # index 位置 → 行 id
        self._faiss_row_count = -1           # 构建时的 DB 行数（惰性重建判据）
        try:
            import faiss  # type: ignore[import-untyped]
            self._faiss = faiss
        except ImportError:
            logger.info("faiss 未安装（pip install faiss-cpu 可启用 ANN 检索），"
                        "使用暴力余弦 fallback")

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id         TEXT PRIMARY KEY,
                    project    TEXT    NOT NULL,
                    namespace  TEXT    NOT NULL DEFAULT 'semantic',
                    text       TEXT    NOT NULL,
                    embedding  BLOB    NOT NULL,
                    metadata   TEXT    DEFAULT '{}',
                    created_at REAL    NOT NULL,
                    updated_at REAL    NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_proj "
                "ON memories(project)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_ns "
                "ON memories(namespace)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_ts "
                "ON memories(created_at)"
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pack_embedding(vec: list[float]) -> bytes:
        return struct.pack(f"{len(vec)}f", *vec)

    @staticmethod
    def _unpack_embedding(blob: bytes) -> list[float]:
        n = len(blob) // 4
        return list(struct.unpack(f"{n}f", blob))

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add(self, text: str, project: str, namespace: str = "semantic",
            metadata: Optional[dict] = None,
            entry_id: Optional[str] = None) -> str:
        """Embed *text* and persist it.  Returns the entry id."""
        entry_id = entry_id or str(uuid.uuid4())[:12]
        now = time.time()
        embedding = self.embedder.encode(text)
        packed = self._pack_embedding(embedding)
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO memories
                   (id, project, namespace, text, embedding, metadata,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (entry_id, project, namespace, text, packed, meta_json,
                 now, now),
            )
            conn.commit()

        logger.debug("CrossProjectStore add [%s] %s/%s: %.80s",
                     entry_id, project, namespace, text)
        return entry_id

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # FAISS ANN 检索（可选加速）
    # ------------------------------------------------------------------

    def build_index(self) -> bool:
        """构建 FAISS 索引（IndexFlatIP，embedding 已 L2-normalize → 内积=余弦）。

        惰性重建：仅在 DB 行数变化时由 search() 触发。返回是否成功构建。
        """
        if self._faiss is None:
            return False
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                rows = conn.execute(
                    "SELECT id, embedding FROM memories"
                ).fetchall()
            if not rows:
                self._faiss_index = None
                self._faiss_ids = []
                self._faiss_row_count = 0
                return False

            dim = self.embedder.dim
            matrix = []
            ids = []
            for row in rows:
                try:
                    vec = self._unpack_embedding(row[1])
                except Exception:
                    continue
                if len(vec) != dim:
                    continue
                matrix.append(vec)
                ids.append(row[0])
            if not matrix:
                return False

            index = self._faiss.IndexFlatIP(dim)
            import numpy as np
            index.add(np.ascontiguousarray(matrix, dtype="float32"))
            self._faiss_index = index
            self._faiss_ids = ids
            self._faiss_row_count = len(rows)
            logger.info("FAISS index built: %d vectors, dim=%d", len(ids), dim)
            return True
        except Exception as exc:
            logger.warning("FAISS build failed (%s); using brute-force fallback", exc)
            self._faiss_index = None
            return False

    def _ensure_index(self) -> None:
        """行数变化时惰性重建索引。"""
        if self._faiss is None:
            return
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        except Exception:
            return
        if count != self._faiss_row_count:
            self.build_index()

    def _search_faiss(self, query_vec: list[float], limit: int) -> list[tuple[str, float]]:
        """FAISS ANN 检索 → [(row_id, score)]。索引未构建时返回 None。"""
        if self._faiss_index is None:
            return []
        import numpy as np
        q = np.ascontiguousarray([query_vec], dtype="float32")
        scores, idxs = self._faiss_index.search(q, min(limit * 3, len(self._faiss_ids)))
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self._faiss_ids):
                continue
            out.append((self._faiss_ids[idx], float(score)))
        return out

    def search(self, query: str, limit: int = 5,
               project: Optional[str] = None,
               namespace: Optional[str] = None) -> list[dict]:
        """Semantic search across memories.

        Parameters
        ----------
        query : str
            Natural-language query.
        limit : int
            Max results to return.
        project : str or None
            If set, restrict to one project; if None, search all projects.
        namespace : str or None
            If set, restrict to one namespace (e.g. "semantic").

        Returns
        -------
        list[dict]
            Ranked by cosine similarity descending.
        """
        query_vec = self.embedder.encode(query)

        # ── FAISS 快路径：先 ANN 候选，再 DB 过滤 project/namespace ──
        self._ensure_index()
        if self._faiss_index is not None:
            try:
                candidates = self._search_faiss(query_vec, limit)
                if candidates:
                    out = []
                    id_to_row = {}
                    ids = [cid for cid, _score in candidates]
                    placeholders = ",".join("?" * len(ids))
                    with sqlite3.connect(str(self.db_path)) as conn:
                        rows = conn.execute(
                            f"SELECT id, project, namespace, text, metadata, created_at "
                            f"FROM memories WHERE id IN ({placeholders})",
                            ids,
                        ).fetchall()
                    for r in rows:
                        id_to_row[r[0]] = r
                    for cid, score in candidates:
                        r = id_to_row.get(cid)
                        if r is None:
                            continue
                        if project and r[1] != project:
                            continue
                        if namespace and r[2] != namespace:
                            continue
                        out.append({
                            "id": r[0], "project": r[1], "namespace": r[2],
                            "text": r[3],
                            "similarity": round(score, 4),
                            "metadata": json.loads(r[4]) if r[4] else {},
                            "created_at": r[5],
                        })
                        if len(out) >= limit:
                            break
                    if out:
                        return out
            except Exception as exc:
                logger.debug("FAISS search failed (%s); falling back", exc)

        conditions = []
        params: list = []
        if project:
            conditions.append("project = ?")
            params.append(project)
        if namespace:
            conditions.append("namespace = ?")
            params.append(namespace)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = (
            f"SELECT id, project, namespace, text, embedding, metadata, created_at "
            f"FROM memories {where} "
            f"ORDER BY created_at DESC LIMIT 500"
        )

        candidates = []
        with sqlite3.connect(str(self.db_path)) as conn:
            for row in conn.execute(sql, params):
                try:
                    emb = self._unpack_embedding(row[4])
                    sim = _cosine_similarity(query_vec, emb)
                    candidates.append({
                        "id": row[0],
                        "project": row[1],
                        "namespace": row[2],
                        "text": row[3],
                        "similarity": round(sim, 4),
                        "metadata": json.loads(row[5]) if row[5] else {},
                        "created_at": row[6],
                    })
                except Exception:
                    logger.debug("Skipping un-decodable embedding for %s", row[0])

        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:limit]

    def search_cross_project(self, query: str, exclude_project: str = "",
                             limit: int = 5) -> list[dict]:
        """Search ALL projects, optionally excluding the current one.

        Useful for surfacing transferable insights:
        "What worked in other projects that might apply here?"
        """
        results = self.search(query, limit=limit * 2)
        if exclude_project:
            results = [r for r in results if r["project"] != exclude_project]
        return results[:limit]

    def get_project_insights(self, project: str, limit: int = 20) -> list[dict]:
        """Recent insights for one project (recency-sorted, no semantic filter)."""
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                """SELECT id, project, namespace, text, metadata, created_at
                   FROM memories WHERE project = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (project, limit),
            ).fetchall()

        return [
            {
                "id": r[0], "project": r[1], "namespace": r[2],
                "text": r[3],
                "metadata": json.loads(r[4]) if r[4] else {},
                "created_at": r[5],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def delete_old(self, days: int = 90) -> int:
        """Purge entries older than *days*.  Returns count deleted."""
        cutoff = time.time() - days * 86400
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.execute(
                "DELETE FROM memories WHERE created_at < ?", (cutoff,)
            )
            conn.commit()
            return cur.rowcount

    def stats(self) -> dict:
        """Return summary statistics for monitoring / dashboards."""
        if not self.db_path.exists():
            return {"total_entries": 0, "db_path": str(self.db_path)}

        with sqlite3.connect(str(self.db_path)) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM memories"
            ).fetchone()[0]
            by_project = dict(
                conn.execute(
                    "SELECT project, COUNT(*) FROM memories "
                    "GROUP BY project ORDER BY COUNT(*) DESC"
                ).fetchall()
            )
            by_namespace = dict(
                conn.execute(
                    "SELECT namespace, COUNT(*) FROM memories "
                    "GROUP BY namespace ORDER BY COUNT(*) DESC"
                ).fetchall()
            )

        size_kb = round(self.db_path.stat().st_size / 1024, 1)

        return {
            "total_entries": total,
            "by_project": by_project,
            "by_namespace": by_namespace,
            "db_path": str(self.db_path),
            "db_size_kb": size_kb,
        }
