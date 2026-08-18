"""工具级 prompt（路由约束）与工具结果摘要测试。

对照 Claude Code 源码分析（cc18 第五层工具 prompt / 第六层 tool summary）：
- 工具 description 必须包含路由约束（何时用/禁止用）——纯指令，无元注释
- 工具结果回灌前即时代总结（超长截断保留关键信息）
"""
import json

import core.nodes as N


class TestToolRoutingPrompts:
    def _desc(self, tool):
        return tool.description

    def test_launch_experiment_handoff(self):
        """launch 交接语义在工具描述中（模型可见）。"""
        desc = self._desc(N.launch_experiment)
        assert "ONLY entry" in desc
        assert "run_shell" in desc
        assert "monitor" in desc
        assert "do not" in desc.lower()

    def test_run_shell_dedicated_tools(self):
        """run_shell 路由约束：有dedicated tools时不用 Shell（对齐 BashTool prompt）。"""
        desc = self._desc(N.run_shell)
        assert "dedicated tools" in desc
        assert "read_file" in desc and "launch_experiment" in desc

    def test_read_file_dedicated(self):
        desc = self._desc(N.read_file)
        assert "run_shell" in desc
        assert "2000" in desc
        assert "re-read" in desc

    def test_no_meta_comments_in_description(self):
        """description 不含元注释（"对齐/参考/思路"等废话）。"""
        for tool in (N.launch_experiment, N.run_shell, N.read_file):
            assert "对齐" not in tool.description
            assert "Claude Code" not in tool.description


class TestToolResultSummarization:
    def test_long_output_summarized(self):
        """超长工具结果 → 截断保留头部 + 截断提示。"""
        long_out = "x" * 3000 + "\nfinal line with key info"
        s = N._summarize_tool_output(long_out, "run_shell")
        assert len(s) < 1200
        assert "truncated" in s
        assert "final line with key info" in s  # 尾部关键行保留
        assert "x" * 3000 not in s

    def test_short_output_unchanged_logic(self):
        """短输出 → 截断阈值判断在调用处（函数本身对短输入也返回带提示）。"""
        s = N._summarize_tool_output("short", "read_file")
        assert "short" in s

    def test_threshold_constant(self):
        assert N._TOOL_RESULT_SUMMARY_CHARS == 800

    def test_read_file_gets_whole_file_visibility(self):
        """冒烟实测修复:read_file 结果被 800 字符截断 → agent 被迫用
        shell dump文件分块读,60 轮全耗在侦察上。read_file 有独立大阈值,
        训练脚本(≤16K 字符)应整体可见。"""
        # 5K 字符的脚本:默认阈值会截断,read_file 阈值不截断
        script = "# line\n" * 200  # 约 1400 行注释 ≈ 9K 字符
        s = N._summarize_tool_output(script, "read_file",
                                     head_chars=N._READ_FILE_SUMMARY_CHARS)
        assert "truncated" not in s
        # 超过 read_file 阈值仍截断(防上下文爆炸)
        huge = "x" * (N._READ_FILE_SUMMARY_CHARS + 5000)
        s2 = N._summarize_tool_output(huge, "read_file",
                                      head_chars=N._READ_FILE_SUMMARY_CHARS)
        assert "truncated" in s2

    def test_run_shell_forbids_file_dump(self):
        desc = N.run_shell.description
        assert "dump" in desc and "read_file" in desc
