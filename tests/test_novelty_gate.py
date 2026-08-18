"""创新路由测试(去硬约束后):任务级契约兜底 + 自适应 min_delta + 计划创新点检测。

用户审查:行为类硬约束(收益递减/连续低创新/同维度门)冗余且不自然,已删除,
路由交由提示词(Tune vs. Innovate + few-shot)驱动。本文件只覆盖保留下来的
最小代码面:任务要求创新时的契约兜底、咨询级自适应阈值、创新点关键词检测。
"""
import tempfile
import unittest
from pathlib import Path

from core.ledger import ExperimentLedger
from core.nodes import ResearchGraph


class AdaptiveMinDeltaTests(unittest.TestCase):
    """自适应停滞阈值(咨询级,注入 think 上下文):max(0.3pp, 40%×离目标距离),
    且不超剩余距离。"""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        ws = Path(self.tempdir.name)
        self.graph = object.__new__(ResearchGraph)
        self.graph.ledger = ExperimentLedger(ws)
        self.graph._ledger_cfg = {"metric_key": "test_acc",
                                  "metric_direction": "higher_better"}
        self.graph._stagnation_cfg = {"target": 0.997,
                                      "min_delta_ratio": 0.4,
                                      "min_delta_floor": 0.003}

    def tearDown(self):
        import gc
        del self.graph
        gc.collect()
        self.tempdir.cleanup()

    def _best(self, acc):
        self.graph.ledger.record(cycle=1, action="experiment",
                                 status="completed", hypothesis="h",
                                 metrics={"test_acc": acc})

    def test_far_from_target_uses_floor(self):
        # 离目标 0.007(0.7pp):40% = 0.0028 < floor 0.003 → 用 floor
        self._best(0.990)
        self.assertEqual(self.graph._stagnation_min_delta("test_acc"), 0.003)

    def test_mid_distance_ratio_exceeds_floor(self):
        # 离目标 0.02(2pp):40% = 0.008 > floor 0.003 → 用 0.008
        self._best(0.977)
        self.assertAlmostEqual(
            self.graph._stagnation_min_delta("test_acc"), 0.008, places=6)

    def test_near_target_capped_by_remaining_distance(self):
        # 离目标 0.0016(0.16pp):40% = 0.00064 < floor → 0.003,但
        # 上限=剩余距离 0.0016 → 0.0016(达标那一跳永远算进展)
        self._best(0.9954)
        self.assertAlmostEqual(
            self.graph._stagnation_min_delta("test_acc"), 0.0016, places=6)

    def test_no_target_falls_back_to_fixed(self):
        self.graph._stagnation_cfg = {"min_delta": 0.001}
        self._best(0.99)
        self.assertEqual(self.graph._stagnation_min_delta("test_acc"), 0.001)


class InnovationPointDetectionTests(unittest.TestCase):
    def test_plan_has_innovation_keywords(self):
        g = object.__new__(ResearchGraph)
        self.assertTrue(g._plan_has_innovation("implement mixup data mixing"))
        self.assertTrue(g._plan_has_innovation("借鉴论文方法"))
        self.assertFalse(g._plan_has_innovation("把通道加宽到 128"))
        self.assertFalse(g._plan_has_innovation("dropout 0.3 + cosine lr"))


class ContractBackstopTests(unittest.TestCase):
    """任务级契约兜底:requires_innovation 且无 IDEA_NOTES 且计划无创新点 → idea。"""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        ws = Path(self.tempdir.name)
        self.graph = object.__new__(ResearchGraph)
        self.graph.workspace = ws
        self.graph.ledger = ExperimentLedger(ws)
        self.graph._ledger_cfg = {"metric_key": "test_acc"}
        self.graph._stagnation_cfg = {"innovation_required": True,
                                      "target": 0.83,
                                      "min_delta_floor": 0.003}
        self.graph._direction_forced = False
        self.graph._last_innovation = ("", "")
        self.graph._current_directive = ""

    def tearDown(self):
        import gc
        del self.graph
        gc.collect()
        self.tempdir.cleanup()

    def _record(self, cycle, hypothesis, acc):
        self.graph.ledger.record(cycle=cycle, action="experiment",
                                 status="completed", hypothesis=hypothesis,
                                 metrics={"test_acc": acc})

    def test_innovation_required_plan_without_innovation_escalates(self):
        """任务要求创新但计划无创新点 → 升级 idea(用户澄清:
        '当前没有提供创新点 → 就去找创新点')。"""
        self._record(1, "加深卷积层", 0.75)
        self._record(2, "加深卷积层", 0.76)
        self._record(3, "加深卷积层", 0.77)
        plan = {"action": "experiment", "agent": "code",
                "task": "把通道加宽到 128", "hypothesis": "更宽更好"}
        out = self.graph._maybe_force_direction_switch(plan)
        self.assertEqual(out["agent"], "idea")
        self.assertIn("task requires innovation", out["task"])

    def test_innovation_required_plan_with_method_passes(self):
        """计划含创新方法关键词(mixup)→ 放行。"""
        self._record(1, "加深卷积层", 0.75)
        self._record(2, "加深卷积层", 0.76)
        self._record(3, "加深卷积层", 0.77)
        plan = {"action": "experiment", "agent": "code",
                "task": "implement mixup data mixing", "hypothesis": "mixup 提升泛化"}
        out = self.graph._maybe_force_direction_switch(plan)
        self.assertEqual(out["agent"], "code")

    def test_innovation_required_after_idea_notes_allows_tuning(self):
        """IDEA_NOTES.md 已存在(= 创新流程完成)→ 后续调参放行。"""
        (self.graph.workspace / "IDEA_NOTES.md").write_text(
            "mixup 方案", encoding="utf-8")
        self._record(1, "实现 mixup", 0.80)
        self._record(2, "实现 mixup", 0.81)
        self._record(3, "实现 mixup", 0.82)
        plan = {"action": "experiment", "agent": "code",
                "task": "把 dropout 降到 0.2", "hypothesis": "微调"}
        out = self.graph._maybe_force_direction_switch(plan)
        self.assertEqual(out["agent"], "code")

    def test_no_innovation_requirement_noop(self):
        """未配置 requires_innovation → 门完全不干预(纯提示词驱动)。"""
        self.graph._stagnation_cfg = {"target": 0.83}
        self._record(1, "加深卷积层", 0.75)
        self._record(2, "加深卷积层", 0.75)
        self._record(3, "加深卷积层", 0.75)
        plan = {"action": "experiment", "agent": "code",
                "task": "把通道加宽到 128", "hypothesis": "更宽更好"}
        out = self.graph._maybe_force_direction_switch(plan)
        self.assertEqual(out["agent"], "code")
        self.assertEqual(out["task"], "把通道加宽到 128")

    def test_directive_protection_blocks_backstop(self):
        """用户指令明确'保持当前方向' → 兜底也禁用。"""
        self.graph._current_directive = "保持当前方向继续调参"
        self._record(1, "加深卷积层", 0.75)
        self._record(2, "加深卷积层", 0.75)
        self._record(3, "加深卷积层", 0.75)
        plan = {"action": "experiment", "agent": "code",
                "task": "把通道加宽到 128", "hypothesis": "更宽更好"}
        out = self.graph._maybe_force_direction_switch(plan)
        self.assertEqual(out["agent"], "code")


if __name__ == "__main__":
    unittest.main()
