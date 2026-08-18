"""文献工具回归测试 — LangGraph 引擎 (core/nodes.py)。

从旧引擎 (core/tools.py ToolRegistry) 迁移:工具改为直接调用 @tool .func,
返回值统一是字符串(结构化工具返回 JSON 文本)。
"""

import json
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from core.nodes import TOOL_FUNCTIONS, get_paper, search_arxiv, search_papers


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._payload


ARXIV_ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.01234v1</id>
    <title>A Great Paper</title>
    <published>2026-05-30T00:00:00Z</published>
    <summary>We do something    novel.</summary>
    <author><name>Jane Doe</name></author>
    <author><name>John Roe</name></author>
  </entry>
</feed>
"""

PAPER_JSON = json.dumps(
    {
        "title": "A Great Paper",
        "abstract": "We do something novel.",
        "year": 2026,
        "citationCount": 5,
        "references": [{"title": f"Ref {i}"} for i in range(40)],
        "citations": [{"title": f"Cite {i}"} for i in range(40)],
    }
).encode()


class LiteratureToolTests(unittest.TestCase):
    def test_idea_agent_exposes_literature_tools(self):
        names = {name for name, _ in TOOL_FUNCTIONS["idea"]}
        self.assertIn("search_arxiv", names)
        self.assertIn("get_paper", names)
        self.assertIn("search_papers", names)

    def test_code_agent_exposes_repo_reading_tools(self):
        names = {name for name, _ in TOOL_FUNCTIONS["code"]}
        self.assertIn("list_tree", names)
        self.assertIn("search_code", names)
        self.assertIn("launch_experiment", names)

    def test_search_arxiv_parses_entries(self):
        with patch("urllib.request.urlopen", return_value=_FakeResponse(ARXIV_ATOM)):
            result = json.loads(search_arxiv.func(query="diffusion"))
        self.assertEqual(len(result["papers"]), 1)
        paper = result["papers"][0]
        self.assertEqual(paper["arxiv_id"], "2401.01234v1")
        self.assertEqual(paper["title"], "A Great Paper")
        self.assertEqual(paper["authors"], ["Jane Doe", "John Roe"])
        self.assertEqual(paper["abstract"], "We do something novel.")

    def test_get_paper_trims_references_and_citations(self):
        with patch("urllib.request.urlopen", return_value=_FakeResponse(PAPER_JSON)):
            result = json.loads(get_paper.func(paper_id="arXiv:2401.01234"))
        self.assertEqual(result["title"], "A Great Paper")
        self.assertEqual(len(result["references"]), 25)
        self.assertEqual(len(result["citations"]), 25)

    def test_get_paper_empty_id_errors_without_network(self):
        result = json.loads(get_paper.func(paper_id="  "))
        self.assertIn("error", result)

    def test_search_arxiv_network_failure_is_graceful(self):
        with patch("urllib.request.urlopen", side_effect=OSError("boom")):
            result = json.loads(search_arxiv.func(query="x"))
        self.assertIn("error", result)
        self.assertIn("arXiv search failed", result["error"])

    def test_search_papers_network_failure_is_graceful(self):
        with patch("urllib.request.urlopen", side_effect=OSError("boom")):
            result = json.loads(search_papers.func(query="x"))
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
