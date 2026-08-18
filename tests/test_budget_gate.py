"""预算封顶测试:cost.daily_budget 启用后,超限终止循环、80% 注入警告。"""
import json
import tempfile
import time
import unittest
from pathlib import Path

from core.cost_tracker import CostTracker
from core.nodes import ResearchGraph


def _mk_graph(workspace: Path, daily_budget: float, costs_usd: float) -> ResearchGraph:
    g = object.__new__(ResearchGraph)
    g.workspace = workspace
    g.cost_tracker = CostTracker(workspace)
    g._daily_budget = daily_budget
    # 伪造今天的成本记录(直接写 costs.jsonl,走真实读取路径)
    entry = {
        "ts": time.time(), "model": "test", "input_tokens": 1, "output_tokens": 1,
        "cost_usd": costs_usd, "actor": "test", "action": "test",
    }
    (workspace / "costs.jsonl").write_text(
        json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")
    return g


class BudgetVerdictTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name)
        self.workspace.mkdir(exist_ok=True)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_disabled_when_budget_zero(self):
        g = _mk_graph(self.workspace, daily_budget=0, costs_usd=100.0)
        self.assertEqual(g._budget_verdict(), "")

    def test_normal_under_budget(self):
        g = _mk_graph(self.workspace, daily_budget=1.0, costs_usd=0.05)
        self.assertEqual(g._budget_verdict(), "")

    def test_warning_at_80_percent(self):
        g = _mk_graph(self.workspace, daily_budget=1.0, costs_usd=0.85)
        self.assertEqual(g._budget_verdict(), "warning")

    def test_exceeded_at_100_percent(self):
        g = _mk_graph(self.workspace, daily_budget=1.0, costs_usd=1.20)
        self.assertEqual(g._budget_verdict(), "exceeded")


if __name__ == "__main__":
    unittest.main()
