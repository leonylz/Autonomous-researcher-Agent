"""ar5iv 全文解析测试:HTML 标题分节 → 方法/实验段优先入库;注入优先级验证。"""
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.cross_project_memory import CrossProjectStore  # noqa: E402
from core.nodes import ResearchGraph  # noqa: E402
from core.rag import RagKnowledgeBase  # noqa: E402
from scripts.ingest_papers import fetch_ar5iv_sections, ingest_arxiv  # noqa: E402

SAMPLE_HTML = """<html><body>
<h1>Mixup: Beyond Empirical Risk Minimization</h1>
<p>We propose mixup, a simple learning principle.</p>
<h2>Abstract</h2>
<p>Mixup trains on convex combinations of pairs of examples.</p>
<h2>Method</h2>
<p>For each batch, sample lambda from Beta(alpha, alpha).</p>
<p>Construct virtual examples: x = lambda*x_i + (1-lambda)*x_j.</p>
<h2>Experiments</h2>
<p>On CIFAR-10 mixup reduces error to 3.9%.</p>
<h2>Related Work</h2>
<p>Prior augmentation methods.</p>
<h2>Conclusion</h2>
<p>We conclude.</p>
</body></html>"""


class _FakeResp:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return SAMPLE_HTML.encode("utf-8")


class Ar5ivParseTests:
    def test_sections_classified_by_heading(self):
        with patch("urllib.request.urlopen", return_value=_FakeResp()):
            sections = fetch_ar5iv_sections("1710.09412")
        assert sections is not None
        # 方法/实验段被提取(决策假设的关键内容)
        assert "Beta(alpha" in sections["methods"]
        assert "CIFAR-10" in sections["methods"]
        # 摘要被提取
        assert "convex combinations" in sections["abstract"]
        # intro/related/conclusion 被丢弃
        assert "Prior augmentation" not in sections["methods"]
        assert "We conclude" not in sections["abstract"]

    def test_network_failure_returns_none(self):
        with patch("urllib.request.urlopen", side_effect=OSError("boom")):
            assert fetch_ar5iv_sections("1710.09412") is None


class SectionAwareInjectionTests:
    def _graph(self, tmp_path: Path) -> ResearchGraph:
        g = object.__new__(ResearchGraph)
        g.workspace = tmp_path
        g._store_project = "proj"
        g._rag_enabled = True
        g.store = type("S", (), {"search": lambda self, ns, limit=8: []})()
        g.rag = RagKnowledgeBase(CrossProjectStore(tmp_path / "mem.db"),
                                 project="rag_proj")
        return g

    def test_methods_chunk_gets_longer_budget_and_priority(self, tmp_path: Path):
        g = self._graph(tmp_path)
        # 同来源:方法段(长文本) + 摘要段
        g.rag.add_paper(
            title="P [arXiv:1710.09412]",
            abstract="A short abstract about mixing.",
            methods="Method detail: " + "x" * 400 + " end.",
        )
        ctx = g._recall_from_store({"task": "implement mixup"})
        rag_text = ctx.get("📚 RAG Knowledge", "")
        # 方法段(400 字符)获得 300 字符额度 → 完整出现,未被 150 截断
        assert ("Method detail: " + "x" * 400) in rag_text
        assert "Method detail: " in rag_text

    def test_abstract_chunk_still_injected(self, tmp_path: Path):
        g = self._graph(tmp_path)
        g.rag.add_paper(title="P [arXiv:1]", abstract="mixing abstract",
                        methods="")
        ctx = g._recall_from_store({"task": "mixup"})
        assert "mixing abstract" in ctx.get("📚 RAG Knowledge", "")

    def test_fulltext_ingest_writes_methods_section(self, tmp_path: Path):
        """ingest_arxiv fulltext 模式:方法段入库且带 section=methods 元数据。"""
        import json
        from core.nodes import set_tool_context  # noqa: F401

        kb = RagKnowledgeBase(CrossProjectStore(tmp_path / "mem.db"),
                              project="rag_t")
        with patch("urllib.request.urlopen",
                   side_effect=[_FakeResp(), _FakeResp()]):
            # 第一次调用抓 arXiv 元数据,第二次抓 ar5iv 全文
            total = ingest_arxiv(kb, ["1710.09412"], fulltext=True)
        assert total >= 2  # abstract 段 + methods 段
        hits = kb.retrieve("beta distribution sampling", top_k=10)
        methods_hits = [h for h in hits
                        if (h.get("metadata") or {}).get("section") == "methods"]
        assert methods_hits, "方法段应入库且带 section 元数据"
        assert "Beta(alpha" in methods_hits[0]["text"]
