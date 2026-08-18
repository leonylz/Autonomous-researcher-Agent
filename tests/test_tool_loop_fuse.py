"""0.5-A 工具循环熔断测试:连续同工具同参数调用 → 中断并提示。"""
import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from core.nodes import ResearchGraph


class _FakeLLM:
    """按脚本返回 tool_call 序列的假 LLM(bind_tools 透传)。"""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = 0

    def invoke(self, messages, **kwargs):
        self.calls += 1
        return self.turns.pop(0) if self.turns else AIMessage(content="done")

    def bind_tools(self, tools):
        return self


def _mk_graph(llm) -> ResearchGraph:
    g = object.__new__(ResearchGraph)
    g._llm_worker = None
    g.llm = llm
    g._manage_context_window = lambda msgs, agent_type, max_tokens=8000: msgs
    g._emit_event = lambda *a, **k: None
    g._tool_loop_fuse = 3
    g._tool_log = []
    return g


def _tool_call(name, args, cid):
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": cid, "type": "tool_call"}])


def _patch_tool():
    """TOOL_FUNCTIONS 持有模块级对象引用 → patch 对象属性(.func)才生效。"""
    return patch("core.nodes.search_arxiv.func",
                 return_value='{"papers": []}')


class ToolLoopFuseTests(unittest.TestCase):
    def test_identical_calls_trigger_fuse(self):
        llm = _FakeLLM([
            _tool_call("search_arxiv", {"query": "diffusion"}, "1"),
            _tool_call("search_arxiv", {"query": "diffusion"}, "2"),
            _tool_call("search_arxiv", {"query": "diffusion"}, "3"),
            _tool_call("search_arxiv", {"query": "diffusion"}, "4"),
        ])
        g = _mk_graph(llm)
        with _patch_tool() as mock_tool:
            result = g._run_worker_single_step("idea", "find papers")

        # 相同调用第 3 次被熔断:工具只真执行 2 次
        self.assertEqual(mock_tool.call_count, 2)
        self.assertIn("fuse", result["response"])

    def test_varied_calls_do_not_trigger_fuse(self):
        llm = _FakeLLM([
            _tool_call("search_arxiv", {"query": "a"}, "1"),
            _tool_call("search_arxiv", {"query": "b"}, "2"),
            _tool_call("search_arxiv", {"query": "a"}, "3"),
            AIMessage(content="done"),
        ])
        g = _mk_graph(llm)
        with _patch_tool() as mock_tool:
            result = g._run_worker_single_step("idea", "find papers")
        self.assertEqual(mock_tool.call_count, 3)  # 无熔断,全部执行
        self.assertNotIn("fuse", result["response"])

    def test_fuse_disabled_with_zero(self):
        llm = _FakeLLM([
            _tool_call("search_arxiv", {"query": "x"}, "1"),
            _tool_call("search_arxiv", {"query": "x"}, "2"),
            _tool_call("search_arxiv", {"query": "x"}, "3"),
            _tool_call("search_arxiv", {"query": "x"}, "4"),
            AIMessage(content="done"),
        ])
        g = _mk_graph(llm)
        g._tool_loop_fuse = 0  # 关闭熔断
        with _patch_tool() as mock_tool:
            g._run_worker_single_step("idea", "find papers")
        self.assertEqual(mock_tool.call_count, 4)  # 全部执行(由 max_turns 兜底)


if __name__ == "__main__":
    unittest.main()
