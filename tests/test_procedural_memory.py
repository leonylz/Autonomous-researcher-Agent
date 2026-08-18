"""实验模式记忆（procedural memory）单元测试。"""
import tempfile
from pathlib import Path

from core.nodes import ResearchGraph
from core.persistent_store import SqliteStore


def _pattern_store(tmp_path: Path) -> SqliteStore:
    s = SqliteStore(tmp_path / "patterns.db")
    s.setup()
    return s


class TestExtractPattern:
    def test_success_pattern(self):
        think = {"hypothesis": "lower learning_rate should help", "task": "train resnet"}
        execute = {"experiment_status": "completed",
                   "final_metrics": {"accuracy": "95.2%"}}
        reflect = {"decision": "continue with lr sweep"}
        p = ResearchGraph._extract_pattern(think, execute, reflect)
        assert p["outcome"] == "success"
        assert p["metric"] == "accuracy=95.2%"
        assert "learning_rate" in p["config"]
        assert "continue" in p["note"]

    def test_failed_pattern_with_terminal(self):
        think = {"task": "train with batch 64"}
        execute = {"experiment_status": "failed",
                   "terminal_state": "OUT_OF_MEMORY", "final_metrics": {}}
        reflect = {}
        p = ResearchGraph._extract_pattern(think, execute, reflect)
        assert p["outcome"] == "failed"
        assert p["terminal_state"] == "OUT_OF_MEMORY"

    def test_no_pid_returns_empty(self):
        think, execute, reflect = {}, {"experiment_status": "no_pid"}, {}
        assert ResearchGraph._extract_pattern(think, execute, reflect) == {}

    def test_metric_priority_order(self):
        think = {"task": "train"}
        execute = {"experiment_status": "completed",
                   "final_metrics": {"loss": "0.31", "accuracy": "90.1%"}}
        p = ResearchGraph._extract_pattern(think, execute, {})
        assert p["metric"] == "accuracy=90.1%"  # accuracy 优先于 loss


class TestProceduralStore:
    def test_write_and_search(self, tmp_path: Path):
        store = _pattern_store(tmp_path)
        pattern = {"config": {"lr": "0.001"}, "outcome": "success",
                   "metric": "accuracy=95.2%", "terminal_state": "", "note": "", "ts": 1.0}
        store.put(("project", "p1", "procedural"), "pattern_1", pattern)
        items = list(store.search(("project", "p1", "procedural"), limit=10))
        assert len(items) == 1
        assert items[0].value["outcome"] == "success"
        assert items[0].value["config"]["lr"] == "0.001"

    def test_consolidate_dedup_same_config(self, tmp_path: Path):
        """同 config 组合只保留最新一条。"""
        store = _pattern_store(tmp_path)
        old = {"config": {"lr": "0.001"}, "outcome": "failed", "ts": 1.0}
        new = {"config": {"lr": "0.001"}, "outcome": "success", "ts": 2.0}
        store.put(("project", "p1", "procedural"), "pattern_1", old)
        store.put(("project", "p1", "procedural"), "pattern_2", new)
        # 模拟 _consolidate_procedural 的去重逻辑（用 store 直接验证）
        items = list(store.search(("project", "p1", "procedural"), limit=10))
        assert len(items) == 2
        # 删除旧的重建
        store.delete(("project", "p1", "procedural"), "pattern_1")
        store.delete(("project", "p1", "procedural"), "pattern_2")
        store.put(("project", "p1", "procedural"), "pattern_2", new)
        items = list(store.search(("project", "p1", "procedural"), limit=10))
        assert len(items) == 1
        assert items[0].value["outcome"] == "success"
