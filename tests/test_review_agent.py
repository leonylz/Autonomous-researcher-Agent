"""Review Agent 单元测试：审查逻辑 + 工具权限边界。"""
import json

from core.nodes import (
    TOOL_FUNCTIONS,
    WORKER_MAX_TURNS,
    _WORKER_PROMPTS,
    _is_training_task,
    _parse_json_response,
)


class TestReviewAgentRegistration:
    def test_tools_are_read_only(self):
        """review 只能读/查/语法检查，绝不能有 launch/write 权限。"""
        tools = dict(TOOL_FUNCTIONS.get("review", []))
        assert "launch_experiment" not in tools
        assert "write_file" not in tools
        assert "run_shell" in tools  # 用于 py_compile
        assert "read_file" in tools
        assert "search_code" in tools

    def test_max_turns_configured(self):
        assert WORKER_MAX_TURNS.get("review", 0) == 15

    def test_prompt_exists(self):
        prompt = _WORKER_PROMPTS.get("review", "")
        assert "review agent" in prompt or "review_agent" in prompt
        assert "approved" in prompt  # JSON 输出格式

    def test_prompt_requires_issues_on_reject(self):
        """拒绝必须列问题（端到端实测暴露：review 曾返回 approved=False + issues=[]）。"""
        prompt = _WORKER_PROMPTS.get("review", "")
        assert "at least one" in prompt or "至少包含" in prompt


class TestIsTrainingTask:
    def test_training_keywords(self):
        assert _is_training_task("训练 CNN 模型")
        assert _is_training_task("train the model")
        assert _is_training_task("launch experiment")
        assert _is_training_task("跑模型验证 accuracy")

    def test_non_training_task(self):
        assert not _is_training_task("查看目录结构")
        assert not _is_training_task("整理实验日志文件")
        assert not _is_training_task("")


class TestReviewDecisionParsing:
    def test_parse_approved(self):
        out = json.dumps({"approved": True, "issues": []})
        parsed = _parse_json_response(out)
        assert parsed.get("approved") is True

    def test_parse_rejected_with_issues(self):
        out = json.dumps({"approved": False, "issues": [
            {"severity": "high", "file": "train.py",
             "message": "data path not found"}]})
        parsed = _parse_json_response(out)
        assert parsed["approved"] is False
        assert parsed["issues"][0]["severity"] == "high"

    def test_parse_prose_fallback(self):
        """非 JSON 输出 → 兜底字典无 approved 键 → 视为不通过（fail-safe）。"""
        parsed = _parse_json_response("this script looks fine")
        # 兜底返回决策字典（action/agent/task），不含 approved 键
        assert bool(parsed.get("approved", False)) is False
