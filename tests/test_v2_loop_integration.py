"""v2 自主层回归测试 — LangGraph 引擎 (core/nodes.py.ResearchGraph)。

从旧引擎 (core/loop.py.ResearchLoop) 迁移。采用与 tests/test_routing.py
一致的 object.__new__ 轻量构造模式(只挂被测方法需要的属性,不跑完整 __init__,
避免 SQLite/信号/环境探测等副作用)。
"""

import tempfile
import unittest
from pathlib import Path

from core.journal import ResearchJournal
from core.ledger import ExperimentLedger
from core.nodes import ResearchGraph


def _mk_graph(workspace: Path, **overrides) -> ResearchGraph:
    g = object.__new__(ResearchGraph)
    g.workspace = workspace
    g.state_path = workspace / "state.json"
    g._ledger_cfg = {
        "metric_key": "acc", "metric_direction": "higher_better",
        "recent_in_context": 5,
    }
    g._stagnation_cfg = {"enabled": True, "threshold_cycles": 2, "min_delta": 0.0}
    g._gates_cfg = {"enabled": True, "threshold": 0.8, "direction": "higher_better"}
    g._journal_cfg = {"tail_in_context": 1500}
    g._safety_cfg = {"enabled": True, "fail_threshold": 3, "stale_state_hours": 6}
    g._no_progress_streak = 0
    g._last_no_progress_signature = ""
    g.ledger = ExperimentLedger(workspace)
    g.journal = ResearchJournal(workspace, max_chars=4000)
    g.memory = None
    for k, v in overrides.items():
        setattr(g, k, v)
    return g


class V2ContextEnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name) / "workspace"
        self.workspace.mkdir()
        self.graph = _mk_graph(self.workspace)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_enrich_context_populates_all_signals(self):
        for i, acc in enumerate([0.70, 0.71, 0.71, 0.71]):
            self.graph.ledger.record(cycle=i, hypothesis=f"exp {i}", status="launched",
                                     metrics={"acc": acc}, ts=float(i))
        self.graph.journal.append_dead_end("SGD without warmup diverges")
        self.graph.journal.append_insight("cosine schedule helps late training")

        context = {}
        self.graph._enrich_context(context)

        self.assertIn("Recent Experiments", context)
        self.assertIn("Progress Signal", context)
        self.assertIn("STAGNATING", context["Progress Signal"])
        self.assertIn("Phase Gate", context)
        self.assertIn("NOT met", context["Phase Gate"])  # best acc 0.71 < 0.8
        self.assertIn("Dead Ends (do NOT retry these)", context)
        self.assertIn("SGD without warmup", context["Dead Ends (do NOT retry these)"])
        self.assertIn("Durable Insights", context)
        self.assertIn("cosine", context["Durable Insights"])

    def test_violation_surfaces_on_repeated_no_progress(self):
        self.graph._no_progress_streak = 3
        context = {}
        self.graph._enrich_context(context)
        self.assertIn("Active Violations", context)

    def test_record_to_ledger_from_cycle_results(self):
        think = {"action": "experiment", "hypothesis": "try dropout"}
        execute = {"experiment_launched": True, "pid": 42, "log_file": "logs/a.log",
                   "final_metrics": {"acc": 0.77}}
        reflect = {"milestone": "best acc so far 0.77"}
        self.graph._record_to_ledger(1, think, execute, reflect)

        entries = self.graph.ledger.all()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["metrics"]["acc"], 0.77)
        self.assertEqual(entries[0]["pid"], 42)
        # milestone captured as a durable insight
        self.assertIn("best acc so far", self.graph.journal.insights_tail(2000))

    def test_record_to_ledger_marks_failed_with_terminal_state(self):
        # A failed experiment is recorded as "failed" (not "launched"), and the
        # sacct terminal state is prefixed onto the conclusion.
        think = {"action": "experiment", "hypothesis": "lr=10"}
        execute = {"experiment_launched": True, "experiment_status": "failed",
                   "terminal_state": "TIMEOUT", "pid": 9, "log_file": "logs/a.log",
                   "final_metrics": {}}
        reflect = {"decision": "retry with lower lr"}
        self.graph._record_to_ledger(2, think, execute, reflect)

        entry = self.graph.ledger.all()[0]
        self.assertEqual(entry["status"], "failed")
        self.assertTrue(entry["conclusion"].startswith("[TIMEOUT] "))


class V2ThrottleTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name) / "workspace"
        self.workspace.mkdir()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_throttle_disabled_is_noop_and_writes_nothing(self):
        g = _mk_graph(self.workspace, max_cycles_per_hour=0,
                      _cycle_times_path=self.workspace / ".cycle_times",
                      _running=True)
        g._throttle_if_needed()
        self.assertFalse(g._cycle_times_path.exists())

    def test_throttle_enabled_records_cycle_time_when_under_budget(self):
        g = _mk_graph(self.workspace, max_cycles_per_hour=6,
                      _cycle_times_path=self.workspace / ".cycle_times",
                      _running=True)
        g._throttle_if_needed()  # under budget -> no sleep, but records a timestamp
        self.assertTrue(g._cycle_times_path.exists())
        self.assertEqual(len(g._load_cycle_times()), 1)


if __name__ == "__main__":
    unittest.main()
