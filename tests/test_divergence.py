"""G2/G5 测试:发散检测提前终止 + 指标有效性判定。"""
import unittest

from core.monitor import ExperimentMonitor


class DivergenceVerdictTests(unittest.TestCase):
    """G2 发散判定(纯函数)。"""

    def _verdict(self, losses, streak=3):
        history = [{"loss": str(l)} for l in losses]
        return ExperimentMonitor._divergence_verdict(history, streak)

    def test_nan_triggers(self):
        self.assertEqual(self._verdict([0.1, 0.2, "nan"]), "nan")

    def test_inf_triggers(self):
        self.assertEqual(self._verdict([0.1, 0.2, "inf"]), "nan")

    def test_rising_loss_triggers(self):
        self.assertEqual(self._verdict([0.1, 0.2, 0.3]), "loss_rising")

    def test_rising_loss_with_improving_acc_is_noise(self):
        """冒烟实测修复:收敛区间的 loss 抖动(0.0067→0.0070)配 acc 改善,
        不是发散 —— 曾误杀健康基线并标记 failed。"""
        history = [
            {"loss": "0.006", "test_acc": "0.990"},
            {"loss": "0.007", "test_acc": "0.991"},
            {"loss": "0.008", "test_acc": "0.992"},
        ]
        self.assertEqual(
            ExperimentMonitor._divergence_verdict(history, 3), "")

    def test_rising_loss_with_flat_acc_is_noise(self):
        history = [
            {"loss": "0.006", "test_acc": "0.990"},
            {"loss": "0.007", "test_acc": "0.990"},
            {"loss": "0.008", "test_acc": "0.990"},
        ]
        self.assertEqual(
            ExperimentMonitor._divergence_verdict(history, 3), "")

    def test_rising_loss_with_falling_acc_still_diverged(self):
        """真正的发散:loss 上升且 acc 同步下滑 → 仍判发散。"""
        history = [
            {"loss": "0.10", "test_acc": "0.90"},
            {"loss": "0.30", "test_acc": "0.80"},
            {"loss": "0.90", "test_acc": "0.60"},
        ]
        self.assertEqual(
            ExperimentMonitor._divergence_verdict(history, 3), "loss_rising")

    def test_rising_loss_no_acc_signal_keeps_conservative(self):
        """无 acc 证据(旧脚本)时维持 loss-only 保守判定。"""
        self.assertEqual(self._verdict([0.1, 0.2, 0.3]), "loss_rising")

    def test_tiny_rise_is_jitter_not_divergence(self):
        """T6 实测修复:CIFAR 级小网络 loss 抖动(0.50→0.503→0.506,
        相对上升 <2%)三连升不判发散 —— 曾把健康基线/mixup 误杀并沉淀成
        否证假设。"""
        self.assertEqual(self._verdict([0.50, 0.503, 0.506]), "")

    def test_meaningful_rise_still_triggers(self):
        """相对上升 ≥2% 的真实发散仍触发。"""
        self.assertEqual(self._verdict([0.50, 0.55, 0.61]), "loss_rising")

    def test_rising_need_streak(self):
        # 只上升 2 次(不足 streak=3)→ 不触发
        self.assertEqual(self._verdict([0.1, 0.2]), "")

    def test_falling_loss_ok(self):
        self.assertEqual(self._verdict([0.3, 0.2, 0.1]), "")

    def test_noisy_loss_ok(self):
        self.assertEqual(self._verdict([0.3, 0.1, 0.2, 0.15]), "")

    def test_empty_history(self):
        self.assertEqual(self._verdict([]), "")


class MetricsValidityTests(unittest.TestCase):
    """G5 有效性判定(供 reflect 前过滤无效指标)。"""

    def _extract(self, lines):
        m = object.__new__(ExperimentMonitor)
        m.backend = None
        return m._extract_metrics(lines)

    def test_nan_metrics_flagged(self):
        metrics = self._extract([
            "METRIC_JSON {\"loss\": \"nan\", \"test_acc\": \"nan\"}"])
        self.assertEqual(metrics["loss"], "nan")

    def test_valid_metrics_preserved(self):
        metrics = self._extract([
            "METRIC_JSON {\"loss\": 0.01, \"test_acc\": 0.95}"])
        self.assertEqual(metrics["test_acc"], "0.95")


if __name__ == "__main__":
    unittest.main()
