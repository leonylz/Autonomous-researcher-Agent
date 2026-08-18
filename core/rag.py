"""
论文/文档知识库 RAG：摄取 → 分块 → 向量检索 → 注入。

复用 CrossProjectStore（core/cross_project_memory.py）做存储与检索：
  - embedding：sentence-transformers（缺失自动 fallback hash 编码）
  - 向量检索：FAISS 优先，缺失 fallback 暴力余弦
  - namespace="rag" 与记忆命名空间隔离

分块策略（轻量版智能分块器）：
  - 按段落（\n\n）切分，超长段落再滑动窗口（chunk_size / overlap）
  - 每块保留来源元数据（source + 块序号），注入时可溯源

用法:
    from .cross_project_memory import CrossProjectStore
    from .rag import RagKnowledgeBase
    kb = RagKnowledgeBase(CrossProjectStore(workspace / "memory.db"))
    kb.add_document(lit_text, source="USER_LITERATURE.md")
    hits = kb.retrieve("learning rate tuning", top_k=3)
"""

from __future__ import annotations

import logging
from typing import Optional

from .cross_project_memory import CrossProjectStore

logger = logging.getLogger("autoresearcher.rag")

RAG_NAMESPACE = "rag"


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> list[str]:
    """轻量智能分块：段落优先，超长段落滑动窗口 + 重叠。

    - 按 \n\n 分段（Markdown/论文摘要天然段落结构）
    - 单段 ≤ chunk_size → 整段一块
    - 单段 > chunk_size → 滑动窗口切（overlap 保留上下文）
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            chunks.append(para)
            continue
        # 滑动窗口
        step = max(1, chunk_size - overlap)
        start = 0
        while start < len(para):
            chunks.append(para[start:start + chunk_size])
            start += step
            if start >= len(para):
                break
    return chunks


class RagKnowledgeBase:
    """基于 CrossProjectStore 的文档知识库（摄取/检索）。"""

    def __init__(self, cross_store: CrossProjectStore,
                 chunk_size: int = 800, overlap: int = 200,
                 project: str = "rag_kb"):
        self.cross_store = cross_store
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.project = project

    # ── 摄取 ──

    def add_document(self, text: str, source: str,
                     metadata: Optional[dict] = None) -> int:
        """分块摄取一篇文档。返回摄取块数。"""
        chunks = _chunk_text(text, self.chunk_size, self.overlap)
        for i, chunk in enumerate(chunks):
            self.cross_store.add(
                text=chunk,
                project=self.project,
                namespace=RAG_NAMESPACE,
                metadata={
                    "source": source,
                    "chunk": i,
                    "total_chunks": len(chunks),
                    **(metadata or {}),
                },
            )
        if chunks:
            logger.info("RAG ingested %d chunks from %s", len(chunks), source)
        return len(chunks)

    def add_paper(self, title: str, abstract: str = "", methods: str = "") -> int:
        """结构化摄取一篇论文（标题 + 摘要 + 方法/实验）。

        幂等：同 source 且该段内容 hash 未变化 → 跳过（防止反复 ingest
        同一论文产生重复 chunk —— 论文库去重的第一道闸）。
        方法/实验段带 section="methods" 元数据 —— 检索注入时可优先展示
        「怎么做」而不是「讲了什么」(idea agent 决策假设靠方法细节)。
        """
        import hashlib
        source = f"paper:{title}"
        existing = self.retrieve("", top_k=100, source_filter=source)
        seen = {(e.get("metadata") or {}).get("content_hash")
                for e in existing}
        total = 0
        if abstract:
            h = hashlib.md5(("abstract:" + abstract).encode("utf-8")).hexdigest()[:12]
            if h not in seen:
                total += self.add_document(
                    f"# {title}\n\n## Abstract\n{abstract}",
                    source=source,
                    metadata={"section": "abstract", "content_hash": h})
        if methods:
            h = hashlib.md5(("methods:" + methods).encode("utf-8")).hexdigest()[:12]
            if h not in seen:
                total += self.add_document(
                    f"# {title}\n\n## Methods\n{methods}",
                    source=source,
                    metadata={"section": "methods", "content_hash": h})
        if not abstract and not methods:
            total += self.add_document(f"# {title}", source=source)
        return total

    # ── 检索 ──

    def retrieve(self, query: str, top_k: int = 3,
                 source_filter: Optional[str] = None) -> list[dict]:
        """向量检索 Top-K 块。返回 [{text, source, chunk, similarity, metadata}]。"""
        results = self.cross_store.search(
            query, limit=top_k, project=self.project, namespace=RAG_NAMESPACE)
        out = []
        for r in results:
            meta = r.get("metadata", {}) or {}
            if source_filter and meta.get("source") != source_filter:
                continue
            out.append({
                "text": r["text"],
                "source": meta.get("source", ""),
                "chunk": meta.get("chunk", 0),
                "similarity": r.get("similarity", 0),
                "metadata": meta,
            })
        return out

    def stats(self) -> dict:
        """知识库统计（总块数 + 来源分布）。"""
        entries = self.cross_store.search("", limit=500,
                                          project=self.project,
                                          namespace=RAG_NAMESPACE)
        by_source: dict[str, int] = {}
        for e in entries:
            src = (e.get("metadata") or {}).get("source", "?")
            by_source[src] = by_source.get(src, 0) + 1
        return {"total_chunks": len(entries), "by_source": by_source}
