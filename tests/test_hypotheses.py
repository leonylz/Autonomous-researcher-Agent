"""G1 HypothesisStore 测试:状态流转/去重/持久化/上下文注入/reflect 结算。"""
import tempfile
import unittest
from pathlib import Path

from core.hypotheses import HypothesisStore
from core.nodes import ResearchGraph


def _mk_store(tmp: Path) -> HypothesisStore:
    return HypothesisStore(tmp / "hypotheses.db")


class HypothesisStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = _mk_store(Path(self.tempdir.name))

    def tearDown(self):
        import gc
        del self.store
        gc.collect()
        self.tempdir.cleanup()

    def test_add_and_dedup(self):
        h1 = self.store.add("mixup improves accuracy")
        h2 = self.store.add("mixup improves accuracy")  # 去重
        self.assertEqual(h1, h2)
        self.assertEqual(len(self.store.all()), 1)

    def test_lifecycle_flow(self):
        hid = self.store.add("cosine schedule helps")
        self.store.mark_testing(hid, experiment_id="exp1")
        self.assertEqual(self.store.get(hid)["status"], "testing")
        self.store.resolve(hid, "confirmed", experiment_id="exp1",
                           evidence="acc 0.88 -> 0.90")
        entry = self.store.get(hid)
        self.assertEqual(entry["status"], "confirmed")
        self.assertEqual(entry["experiment_id"], "exp1")

    def test_invalid_outcome_rejected(self):
        hid = self.store.add("x")
        with self.assertRaises(ValueError):
            self.store.resolve(hid, "bogus")

    def test_pending_and_refuted_queries(self):
        a = self.store.add("hyp A")
        b = self.store.add("hyp B")
        self.store.resolve(a, "refuted", evidence="failed")
        self.assertEqual(len(self.store.pending()), 1)
        self.assertEqual(self.store.pending()[0]["id"], b)
        self.assertEqual(self.store.refuted()[0]["id"], a)

    def test_to_context_renders_sections(self):
        self.store.add("待验证假设 A")
        r = self.store.add("已否证假设 B")
        self.store.resolve(r, "refuted", evidence="实验失败")
        ctx = self.store.to_context()
        self.assertIn("待验证假设", ctx)
        self.assertIn("已否证假设", ctx)
        self.assertIn("❌", ctx)


class HypothesisNodesIntegrationTests(unittest.TestCase):
    """reflect 结算 + think 注入(nodes 层)。"""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.ws = Path(self.tempdir.name)
        self.ws.mkdir(exist_ok=True)
        self.graph = object.__new__(ResearchGraph)
        self.graph.workspace = self.ws
        self.graph.hypotheses = _mk_store(self.ws)

    def tearDown(self):
        import gc
        del self.graph
        gc.collect()
        self.tempdir.cleanup()

    def test_failed_experiment_refutes_hypothesis(self):
        think = {"action": "experiment", "hypothesis": "lr=10 works"}
        execute = {"experiment_status": "failed", "pid": 1}
        reflect = {"decision": "lr too high"}
        self.graph._settle_hypothesis(think, execute, reflect)
        h = self.graph.hypotheses.find_by_text("lr=10 works")
        self.assertEqual(h["status"], "refuted")

    def test_milestone_confirms_hypothesis(self):
        think = {"action": "experiment", "hypothesis": "mixup helps"}
        execute = {"experiment_status": "completed", "pid": 2}
        reflect = {"milestone": "acc 0.95 best so far"}
        self.graph._settle_hypothesis(think, execute, reflect)
        h = self.graph.hypotheses.find_by_text("mixup helps")
        self.assertEqual(h["status"], "confirmed")

    def test_no_milestone_is_inconclusive(self):
        think = {"action": "experiment", "hypothesis": "dropout 0.5"}
        execute = {"experiment_status": "completed", "pid": 3}
        reflect = {}
        self.graph._settle_hypothesis(think, execute, reflect)
        h = self.graph.hypotheses.find_by_text("dropout 0.5")
        self.assertEqual(h["status"], "inconclusive")

    def test_missing_hypothesis_is_noop(self):
        self.graph._settle_hypothesis({"action": "experiment"},
                                      {"experiment_status": "failed"}, {})
        self.assertEqual(self.graph.hypotheses.all(), [])

    def test_meta_statement_not_stored_as_hypothesis(self):
        """冒烟实测:think 在收尾轮把「无需新假设,目标已达成。」写进
        hypothesis 字段 —— 元陈述不是可验证假设,不入账本。"""
        think = {"action": "report", "hypothesis": "无需新假设，目标已达成。"}
        self.graph._settle_hypothesis(think, {"experiment_status": "report"}, {})
        self.assertEqual(self.graph.hypotheses.all(), [])

    def test_report_cycle_does_not_clobber_confirmed(self):
        """冒烟实测修复:确认过的假设被后续无实验信号轮(如报告轮)
        的 else 分支 mark_testing 打回 testing,结论丢失。"""
        think = {"action": "experiment", "hypothesis": "warmup helps"}
        execute = {"experiment_status": "completed", "pid": 7}
        reflect = {"milestone": "acc 0.97"}
        self.graph._settle_hypothesis(think, execute, reflect)
        h = self.graph.hypotheses.find_by_text("warmup helps")
        self.assertEqual(h["status"], "confirmed")

        # 下一轮是报告轮(无实验)→ 同一假设去重复用,else 分支 mark_testing
        think2 = {"action": "report", "hypothesis": "warmup helps"}
        execute2 = {"experiment_status": "report"}
        self.graph._settle_hypothesis(think2, execute2, {})
        h2 = self.graph.hypotheses.find_by_text("warmup helps")
        self.assertEqual(h2["status"], "confirmed",
                         "无信息量更新不得降级已有结论")
        self.assertEqual(h2["experiment_id"], "7")


class HypothesisStoreGuardTests(unittest.TestCase):
    """store 层守卫:mark_testing 不降级终态结论。"""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = _mk_store(Path(self.tempdir.name))

    def tearDown(self):
        import gc
        del self.store
        gc.collect()
        self.tempdir.cleanup()

    def test_mark_testing_keeps_terminal_status(self):
        hid = self.store.add("hyp")
        self.store.resolve(hid, "refuted", evidence="failed")
        self.store.mark_testing(hid, experiment_id="exp9")
        h = self.store.get(hid)
        self.assertEqual(h["status"], "refuted")
        self.assertEqual(h["evidence"], "failed")


class EvidenceAwareSettleTests(unittest.TestCase):
    """用户审查修复:结算看指标增量,单次小幅负结果 ≠ 否证。

    T6 实测:加宽通道 -0.6pp 被 settled confirmed(证据却说 refuted);
    基线/mixup 被发散误判 failed 沉淀成 refuted。"""
    from core.ledger import ExperimentLedger

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.ws = Path(self.tempdir.name)
        self.ws.mkdir(exist_ok=True)
        self.graph = object.__new__(ResearchGraph)
        self.graph.workspace = self.ws
        self.graph.hypotheses = _mk_store(self.ws)
        self.graph.ledger = self.ExperimentLedger(self.ws)
        self.graph._ledger_cfg = {"metric_key": "test_acc"}
        # 前置账本:历史最佳 0.7453(mixup+SGDR 轮)
        self.graph.ledger.record(cycle=3, action="experiment",
                                 status="completed", hypothesis="prev",
                                 metrics={"test_acc": 0.7453})

    def tearDown(self):
        import gc
        del self.graph
        gc.collect()
        self.tempdir.cleanup()

    def _settle(self, best, milestone="a result", status="completed"):
        think = {"action": "experiment", "hypothesis": "widening helps"}
        execute = {"experiment_status": status, "pid": 9,
                   "final_metrics": {"test_acc": best, "accuracy": best}}
        reflect = {"milestone": milestone, "decision": "the run finished"}
        self.graph._settle_hypothesis(think, execute, reflect, cycle=4)
        return self.graph.hypotheses.find_by_text("widening helps")

    def test_small_negative_delta_is_inconclusive_not_refuted(self):
        """-0.86pp(0.7367 vs 0.7453):单次小幅负结果 → inconclusive,
        证据注明架构/超参混淆可能(用户批评的'否得太容易'修复)。"""
        h = self._settle(0.7367)
        self.assertEqual(h["status"], "inconclusive")
        self.assertIn("confound", h["evidence"])

    def test_clear_negative_delta_refutes(self):
        """明确下降(≤ -1pp)才否证。"""
        h = self._settle(0.7300)
        self.assertEqual(h["status"], "refuted")
        self.assertIn("clear drop", h["evidence"])

    def test_improvement_confirms(self):
        h = self._settle(0.7500)
        self.assertEqual(h["status"], "confirmed")

    def test_failed_refutes_with_truthful_terminal_state(self):
        """失败 → refuted,但证据必须带真实 terminal_state(发散误判可见)。"""
        think = {"action": "experiment", "hypothesis": "x helps"}
        execute = {"experiment_status": "failed", "pid": 9,
                   "terminal_state": "diverged:loss_rising"}
        reflect = {"decision": "run crashed"}
        self.graph._settle_hypothesis(think, execute, reflect, cycle=4)
        h = self.graph.hypotheses.find_by_text("x helps")
        self.assertEqual(h["status"], "refuted")
        self.assertIn("diverged:loss_rising", h["evidence"])


if __name__ == "__main__":
    unittest.main()
