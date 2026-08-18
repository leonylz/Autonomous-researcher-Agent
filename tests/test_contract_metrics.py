"""H1/H4 契约指标测试:METRIC_JSON 契约行优先 + 正则 fallback + 字段归一。"""
import unittest

from core.monitor import ExperimentMonitor


def _extract(lines):
    """直接调用私有方法(构造轻量实例)。"""
    m = object.__new__(ExperimentMonitor)
    m.backend = None
    return m._extract_metrics(lines)


class ContractMetricsTests(unittest.TestCase):
    def test_contract_line_fields_preserved(self):
        """契约行字段名原样(test_acc 直接可达,不再变 accuracy)。"""
        metrics = _extract([
            "Epoch 1/10 | loss=0.1 | test_acc=0.5",
            "METRIC_JSON {\"epoch\": 1, \"loss\": 0.013, \"test_acc\": 0.992}",
        ])
        self.assertEqual(metrics["test_acc"], "0.992")
        self.assertEqual(metrics["loss"], "0.013")
        self.assertEqual(metrics["epoch"], "1")

    def test_contract_line_wins_over_regex(self):
        """契约行存在时优先(即使正则也能匹配到不同值)。"""
        metrics = _extract([
            "METRIC_JSON {\"test_acc\": 0.99}",
            "Epoch 9/10 | loss=0.2 | test_acc=0.50",
        ])
        self.assertEqual(metrics["test_acc"], "0.99")

    def test_regex_fallback_normalizes_accuracy(self):
        """无契约行 → 正则 fallback,accuracy 归一为 test_acc(修复历史断裂)。"""
        metrics = _extract(["Epoch 3/10 | loss=0.05 | test_acc=0.88"])
        self.assertEqual(metrics["test_acc"], "0.88")
        self.assertEqual(metrics["loss"], "0.05")

    def test_contract_line_plus_regex_epoch_fill(self):
        """契约行有 loss/acc,正则补充 epoch(模板日志里都有)。"""
        metrics = _extract([
            "Epoch 5/10 | loss=0.01 | test_acc=0.95",
            "METRIC_JSON {\"loss\": 0.012, \"test_acc\": 0.951}",
        ])
        self.assertEqual(metrics["test_acc"], "0.951")
        self.assertEqual(metrics["epoch"], "5")

    def test_empty_logs(self):
        self.assertEqual(_extract([]), {})


if __name__ == "__main__":
    unittest.main()
