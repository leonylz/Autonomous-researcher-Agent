"""滑动窗口上下文压缩的单元测试。

验证核心不变量：压缩后 ToolMessage ↔ AIMessage 配对完整，
不会产生 "ToolMessage with id X not found" 的孤立消息。
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from core.nodes import ResearchGraph


def _ai_with_tool(call_id: str, text: str = "x") -> AIMessage:
    return AIMessage(
        content=text,
        tool_calls=[{"name": "run_shell", "args": {"command": "echo hi"},
                     "id": call_id, "type": "tool_call"}],
    )


def _tool(call_id: str, payload: str = "ok") -> ToolMessage:
    return ToolMessage(content=payload, tool_call_id=call_id)


def _assert_no_orphan(messages: list) -> None:
    """不变量：任何 ToolMessage 之前（在本窗口内）必须有配对的 AIMessage。"""
    seen_tool_call_ids = set()
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in (m.tool_calls or []):
                seen_tool_call_ids.add(tc.get("id"))
        elif isinstance(m, ToolMessage):
            assert m.tool_call_id in seen_tool_call_ids, \
                f"orphan ToolMessage: {m.tool_call_id!r}"


def _small_tokens(msgs: list) -> int:
    """用极小的 token 估算强制触发压缩（避免依赖真实长度）。"""
    return sum(len(str(getattr(m, 'content', ''))) for m in msgs) // 4


class TestRoundSafeStart:
    def test_skips_leading_tool_messages(self):
        msgs = [_tool("t1"), _ai_with_tool("t2"), _tool("t2")]
        # start 0 但 msgs[0] 是孤立 ToolMessage → 跳到 1
        assert ResearchGraph._round_safe_start(msgs, 0) == 1

    def test_normal_start_unchanged(self):
        msgs = [_ai_with_tool("t1"), _tool("t1")]
        assert ResearchGraph._round_safe_start(msgs, 0) == 0

    def test_start_at_non_tool_mid(self):
        msgs = [_ai_with_tool("a"), _tool("a"), _ai_with_tool("b"), _tool("b")]
        # index 1 是 ToolMessage（配对的 AIMessage 在 0）→ 跳到 2
        assert ResearchGraph._round_safe_start(msgs, 1) == 2
        assert ResearchGraph._round_safe_start(msgs, 2) == 2


class TestManageContextWindow:
    def _build_long_convo(self, rounds: int = 6) -> list:
        """SystemMessage + HumanMessage + N 个工具轮。"""
        msgs = [SystemMessage(content="sys"), HumanMessage(content="task")]
        for i in range(rounds):
            msgs.append(_ai_with_tool(f"c{i}"))
            msgs.append(_tool(f"c{i}", "out" * 40))  # 每个 tool result 80 字符
        return msgs

    def test_under_budget_returns_unchanged(self):
        msgs = self._build_long_convo(rounds=2)
        result = ResearchGraph._manage_context_window(
            msgs, "code", max_tokens=10_000_000)
        assert result is msgs  # 原对象

    def test_over_budget_triggers_compression(self):
        msgs = self._build_long_convo(rounds=6)
        # 给一个比总量小的预算，触发压缩
        total = _small_tokens(msgs)
        result = ResearchGraph._manage_context_window(
            msgs, "code", max_tokens=max(2, total - 1))
        assert _small_tokens(result) <= max(2, total - 1)
        _assert_no_orphan(result)

    def test_level3_hard_truncation_preserves_pairing(self):
        """关键回归：窗口起点落在 ToolMessage 上时必须跳过，保证配对。"""
        # 构造一个较长会话，让最后 8 条的起点恰好在 ToolMessage 上
        msgs = [SystemMessage(content="sys"), HumanMessage(content="task")]
        for i in range(10):
            msgs.append(_ai_with_tool(f"c{i}"))
            msgs.append(_tool(f"c{i}", "out" * 40))
        # 用极小预算强制走第 3 级硬截断
        result = ResearchGraph._manage_context_window(
            msgs, "code", max_tokens=1)
        _assert_no_orphan(result)
        assert isinstance(result[0], SystemMessage)

    def test_level2_summary_preserves_pairing(self):
        """第 2 级摘要后 late 窗口不产生孤立 ToolMessage。"""
        msgs = [SystemMessage(content="sys"), HumanMessage(content="task")]
        for i in range(6):
            msgs.append(_ai_with_tool(f"c{i}"))
            msgs.append(_tool(f"c{i}", "out" * 40))
        # 预算介于 level1 和 level2 之间 → 触发 level2
        l1 = ResearchGraph._estimate_tokens(msgs)
        # 先截断 tool 结果后再算，找 level2 触发区间
        result = ResearchGraph._manage_context_window(
            msgs, "code", max_tokens=max(1, l1 // 3))
        _assert_no_orphan(result)
        _assert_no_orphan(result)

    def test_level1_truncates_long_tool_results(self):
        msgs = [SystemMessage(content="sys"), HumanMessage(content="task"),
                _ai_with_tool("c1"), _tool("c1", "z" * 2000)]
        total = ResearchGraph._estimate_tokens(msgs)
        result = ResearchGraph._manage_context_window(
            msgs, "code", max_tokens=max(1, total - 1), tool_result_max_chars=100)
        # level-1 只截断 tool result 内容，不删消息
        assert len(result) == len(msgs)
        for m in result:
            if isinstance(m, ToolMessage):
                assert len(str(m.content)) < 500
        _assert_no_orphan(result)
