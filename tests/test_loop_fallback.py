"""反卡死(no-progress fallback)回归测试 — LangGraph 引擎。

从旧引擎 (core/loop.py) 迁移到 ResearchGraph。原 ResearchLoop.run() 的
「重置 leader 历史」测试对应新引擎的 StateGraph 生命周期,不再有等价物,
已由 tests/test_routing.py 的路由语义测试覆盖。
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from core.journal import ResearchJournal
from core.nodes import ResearchGraph


def _mk_graph(workspace: Path, threshold: int = 2) -> ResearchGraph:
    g = object.__new__(ResearchGraph)
    g.workspace = workspace
    g.no_progress_fallback_threshold = threshold
    g._no_progress_streak = 0
    g._last_no_progress_signature = ""
    g.memory = Mock()  # log_decision 记录即可,不落盘
    g.journal = ResearchJournal(workspace, max_chars=4000)
    return g


class NoProgressFallbackTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name) / "workspace"
        self.workspace.mkdir()
        self.graph = _mk_graph(self.workspace)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_repeated_no_progress_plan_triggers_wait_fallback(self):
        plan = {
            "action": "experiment",
            "agent": "code",
            "task": "Retry the same broken command",
            "hypothesis": "It might work this time",
        }
        execute_result = {"experiment_launched": False}
        reflect_result = {}

        self.graph._record_cycle_outcome(plan, execute_result, reflect_result)
        self.graph._record_cycle_outcome(plan, execute_result, reflect_result)

        fallback = self.graph._apply_no_progress_fallback(plan, directive=None)

        self.assertEqual(fallback["action"], "wait")
        self.assertIn("Fallback triggered", fallback["reason"])
        # 失败方向被记为 dead end,后续 REFLECT 会看到
        self.assertIn("no progress", self.graph.journal.dead_ends_tail(2000))

    def test_progress_resets_streak(self):
        plan = {"action": "experiment", "agent": "code", "task": "t",
                "hypothesis": "h"}
        self.graph._record_cycle_outcome(plan, {"experiment_launched": False}, {})
        self.assertEqual(self.graph._no_progress_streak, 1)
        # 一旦启动过实验,计数器清零
        self.graph._record_cycle_outcome(
            plan, {"experiment_launched": True, "pid": 1}, {}
        )
        self.assertEqual(self.graph._no_progress_streak, 0)

    def test_directive_bypasses_wait_but_still_allows_execution(self):
        plan = {"action": "experiment", "agent": "code", "task": "t",
                "hypothesis": "h"}
        for _ in range(3):
            self.graph._record_cycle_outcome(plan, {"experiment_launched": False}, {})

        # 有用户指令时不强制 wait,避免掐断指令
        result = self.graph._apply_no_progress_fallback(plan, directive="try lr=1e-4")
        self.assertEqual(result["action"], "experiment")

    def test_threshold_zero_disables_fallback(self):
        g = _mk_graph(self.workspace, threshold=0)
        plan = {"action": "experiment", "agent": "code", "task": "t", "hypothesis": "h"}
        result = g._apply_no_progress_fallback(plan, directive=None)
        self.assertEqual(result["action"], "experiment")


if __name__ == "__main__":
    unittest.main()
