"""ingest_papers.py 冒烟测试 — 离线(mock arXiv API),验证摄取逻辑。"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.cross_project_memory import CrossProjectStore  # noqa: E402
from core.rag import RagKnowledgeBase  # noqa: E402
from scripts import ingest_papers  # noqa: E402

ARXIV_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1708.04552v2</id>
    <title>mixup: Beyond Empirical Risk Minimization</title>
    <summary>Mixup trains on convex combinations of pairs of examples.</summary>
  </entry>
</feed>
"""


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return ARXIV_FEED


def test_ingest_arxiv_pulls_metadata_and_chunks(tmp_path: Path):
    kb = RagKnowledgeBase(CrossProjectStore(tmp_path / "mem.db"), project="rag_t")
    with patch("urllib.request.urlopen", return_value=_FakeResponse()):
        total = ingest_papers.ingest_arxiv(kb, ["1708.04552"])

    assert total >= 1
    hits = kb.retrieve("mixup convex combinations", top_k=5)
    assert hits
    assert "1708.04552" in hits[0]["source"]


def test_ingest_dir_scans_markdown(tmp_path: Path):
    lit = tmp_path / "literature"
    lit.mkdir()
    (lit / "notes.md").write_text(
        "# Cosine annealing\nCosine schedule with warmup helps late training.\n\n"
        "Use with SGD momentum.\n", encoding="utf-8")

    kb = RagKnowledgeBase(CrossProjectStore(tmp_path / "mem.db"), project="rag_t")
    total = ingest_papers.ingest_dir(kb, lit)

    assert total >= 1
    hits = kb.retrieve("cosine warmup", top_k=5)
    assert hits
    assert hits[0]["source"] == "notes.md"


def test_ingest_dir_skips_non_docs(tmp_path: Path):
    lit = tmp_path / "literature"
    lit.mkdir()
    (lit / "weights.bin").write_bytes(b"\x00\x01")
    (lit / "notes.md").write_text("useful", encoding="utf-8")

    kb = RagKnowledgeBase(CrossProjectStore(tmp_path / "mem.db"), project="rag_t")
    total = ingest_papers.ingest_dir(kb, lit)
    assert total == 1  # 只有 .md 被摄取
