"""论文知识库 RAG 单元测试：分块、摄取、检索、注入。"""
from pathlib import Path

from core.cross_project_memory import CrossProjectStore
from core.rag import RagKnowledgeBase, _chunk_text


def _make_kb(tmp_path: Path) -> RagKnowledgeBase:
    return RagKnowledgeBase(CrossProjectStore(tmp_path / "mem.db"))


class TestChunkText:
    def test_empty(self):
        assert _chunk_text("") == []
        assert _chunk_text("   ") == []

    def test_short_paragraph_single_chunk(self):
        chunks = _chunk_text("short text")
        assert chunks == ["short text"]

    def test_paragraphs_split(self):
        chunks = _chunk_text("para one\n\npara two")
        assert chunks == ["para one", "para two"]

    def test_long_paragraph_sliding_window(self):
        text = "word " * 500  # 2500 chars > 800
        chunks = _chunk_text(text, chunk_size=800, overlap=200)
        assert len(chunks) > 1
        assert all(len(c) <= 800 for c in chunks)
        # 重叠：相邻块有共享内容
        assert chunks[0][-100:] in chunks[1] or chunks[1][:100] in chunks[0] \
            or len(chunks[0]) == 800


class TestRagKnowledgeBase:
    def test_add_document_and_retrieve(self, tmp_path: Path):
        kb = _make_kb(tmp_path)
        n = kb.add_document(
            "learning rate scheduling with cosine annealing improves convergence.\n\n"
            "batch size 512 with gradient accumulation.", source="paperA")
        assert n == 2
        hits = kb.retrieve("cosine annealing learning rate", top_k=3)
        assert hits
        assert hits[0]["source"] == "paperA"
        assert "cosine annealing" in hits[0]["text"]

    def test_add_paper_structured(self, tmp_path: Path):
        kb = _make_kb(tmp_path)
        kb.add_paper("Mixup: Beyond Empirical Risk Minimization",
                     abstract="Mixup trains on convex combinations.",
                     methods="lambda sampled from Beta distribution.")
        hits = kb.retrieve("mixup training", top_k=3)
        assert hits
        assert hits[0]["source"] == "paper:Mixup: Beyond Empirical Risk Minimization"

    def test_retrieve_source_filter(self, tmp_path: Path):
        kb = _make_kb(tmp_path)
        kb.add_document("alpha method", source="paperA")
        kb.add_document("alpha method", source="paperB")
        hits = kb.retrieve("alpha", top_k=10, source_filter="paperA")
        assert hits and all(h["source"] == "paperA" for h in hits)

    def test_retrieve_empty_kb(self, tmp_path: Path):
        kb = _make_kb(tmp_path)
        assert kb.retrieve("anything") == []

    def test_stats(self, tmp_path: Path):
        kb = _make_kb(tmp_path)
        kb.add_document("doc one", source="s1")
        kb.add_document("doc two", source="s2")
        st = kb.stats()
        assert st["total_chunks"] == 2
        assert st["by_source"] == {"s1": 1, "s2": 1}

    def test_metadata_carries_source_and_chunk(self, tmp_path: Path):
        kb = _make_kb(tmp_path)
        kb.add_document("one\n\ntwo\n\nthree", source="doc")
        hits = kb.retrieve("", top_k=10)
        assert len(hits) == 3
        # 检索按更新时间排序，块顺序不保证 → 验证 chunk 集合完整
        assert sorted(h["chunk"] for h in hits) == [0, 1, 2]
        assert all(h["metadata"]["total_chunks"] == 3 for h in hits)


class TestAddPaperIdempotent:
    def test_same_paper_twice_is_skipped(self, tmp_path: Path):
        kb = _make_kb(tmp_path)
        n1 = kb.add_paper("P [arXiv:1]", abstract="abs", methods="methods")
        n2 = kb.add_paper("P [arXiv:1]", abstract="abs", methods="methods")
        # 每段 2 chunks(标题块 + 内容块):abstract 2 + methods 2
        assert n1 == 4
        assert n2 == 0  # 幂等:完全相同的论文不再入库
        assert kb.stats()["total_chunks"] == 4

    def test_methods_change_only_adds_methods_chunk(self, tmp_path: Path):
        kb = _make_kb(tmp_path)
        kb.add_paper("P [arXiv:1]", abstract="abs", methods="v1")
        n = kb.add_paper("P [arXiv:1]", abstract="abs", methods="v2")
        assert n == 2  # 只有 methods 段变化 → 只补 methods 段(2 chunks)
        hits = kb.retrieve("v2", top_k=10)
        assert hits and "v2" in hits[0]["text"]

    def test_section_metadata_carried(self, tmp_path: Path):
        kb = _make_kb(tmp_path)
        kb.add_paper("P [arXiv:1]", abstract="a", methods="m")
        hits = kb.retrieve("", top_k=10)
        sections = {(h.get("metadata") or {}).get("section") for h in hits}
        assert sections == {"abstract", "methods"}
        assert all(h["source"].startswith("paper:P") for h in hits)
