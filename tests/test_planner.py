"""多步实验规划器（Plan-then-Execute with Replan）单元测试。"""
import json

from core.nodes import ResearchGraph


def _plan(*steps):
    """构造计划列表。"""
    return [{"step_id": s[0], "title": s[1], "agent": s[2], "status": s[3], "result": ""}
            for s in steps]


class TestParsePlan:
    def test_empty(self):
        assert ResearchGraph._parse_plan("") == []
        assert ResearchGraph._parse_plan(None) == []

    def test_invalid_json(self):
        assert ResearchGraph._parse_plan("not json") == []

    def test_valid(self):
        raw = json.dumps(_plan(("s1", "baseline", "code", "pending")))
        assert len(ResearchGraph._parse_plan(raw)) == 1

    def test_filters_bad_steps(self):
        raw = json.dumps([
            {"step_id": "s1", "status": "pending"},
            {"title": "no id", "agent": "code"},
            {"step_id": "s2", "status": "done"},
        ])
        steps = ResearchGraph._parse_plan(raw)
        assert len(steps) == 2  # 缺 step_id 或 status 的过滤掉


class TestFormatPlan:
    def test_empty(self):
        assert ResearchGraph._format_plan([]) == ""

    def test_renders_status_markers(self):
        plan = _plan(("s1", "baseline", "code", "done"),
                     ("s2", "tune lr", "code", "pending"))
        text = ResearchGraph._format_plan(plan)
        assert "✅" in text and "baseline" in text
        assert "⬜" in text and "tune lr" in text


class TestMergePlan:
    def test_first_plan_adopted_with_running(self):
        """无 plan + Leader 给出 → 采纳，首个步骤标 running。"""
        leader = [{"step_id": "s1", "title": "baseline", "agent": "code"},
                  {"step_id": "s2", "title": "tune lr", "agent": "code"}]
        merged = ResearchGraph._merge_plan("", leader)
        steps = json.loads(merged)
        assert steps[0]["status"] == "running"
        assert steps[1]["status"] == "pending"

    def test_existing_plan_preserved(self):
        """已有 plan → Leader 返回 [] 时沿用，且下一个 pending 标 running。"""
        current = json.dumps(_plan(("s1", "baseline", "code", "done"),
                                   ("s2", "tune lr", "code", "pending")))
        merged = ResearchGraph._merge_plan(current, [])
        steps = json.loads(merged)
        assert steps[0]["status"] == "done"
        assert steps[1]["status"] == "running"

    def test_leader_empty_and_no_plan(self):
        assert ResearchGraph._merge_plan("", []) == ""


class TestReplan:
    def test_running_step_marked_done_on_success(self):
        plan = _plan(("s1", "baseline", "code", "running"))
        exec_data = {"experiment_status": "completed",
                     "final_metrics": {"accuracy": "95.2%"}}
        revised = ResearchGraph._replan(plan, exec_data, {})
        assert revised == []  # 单步计划全部 done → 清空，下轮重新规划

    def test_partial_plan_keeps_done_marker(self):
        """多步计划：当前步骤标 done + 附结果，后续步骤保持 pending。"""
        plan = _plan(("s1", "baseline", "code", "running"),
                     ("s2", "tune lr", "code", "pending"))
        exec_data = {"experiment_status": "completed",
                     "final_metrics": {"accuracy": "95.2%"}}
        revised = ResearchGraph._replan(plan, exec_data, {})
        assert revised[0]["status"] == "done"
        assert "95.2%" in revised[0]["result"]
        assert revised[1]["status"] == "pending"

    def test_running_step_marked_failed(self):
        plan = _plan(("s1", "baseline", "code", "running"),
                     ("s2", "tune lr", "code", "pending"))
        exec_data = {"experiment_status": "failed", "terminal_state": "OUT_OF_MEMORY"}
        revised = ResearchGraph._replan(plan, exec_data, {})
        assert revised[0]["status"] == "failed"
        assert "OUT_OF_MEMORY" in revised[0]["result"]
        assert revised[1]["status"] == "pending"

    def test_leader_replan_replaces(self):
        plan = _plan(("s1", "baseline", "code", "running"))
        reflect = {"plan": [{"step_id": "new1", "title": "new direction", "agent": "idea"}]}
        revised = ResearchGraph._replan(plan, {}, reflect)
        assert len(revised) == 1
        assert revised[0]["step_id"] == "new1"
        assert revised[0]["status"] == "pending"

    def test_no_plan_returns_empty(self):
        assert ResearchGraph._replan([], {}, {}) == []
