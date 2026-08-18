"""实例锁 + 孤儿训练检测测试（防同时多个 train）。"""
import json
import os
import tempfile
from pathlib import Path

import pytest

from core.nodes import ResearchGraph


def _mk_graph(workspace: Path, alive_pid: int | None = None):
    """构造最小 ResearchGraph（只测锁/孤儿逻辑）。"""
    g = object.__new__(ResearchGraph)
    g.workspace = workspace
    g._lock_path = workspace / ".agent.lock"
    g.monitor = type("M", (), {"backend": object()})()
    return g


class TestAgentLock:
    def test_acquire_creates_lock(self, tmp_path: Path):
        g = _mk_graph(tmp_path)
        g._acquire_agent_lock()
        assert (tmp_path / ".agent.lock").exists()
        assert int((tmp_path / ".agent.lock").read_text()) == os.getpid()

    def test_second_instance_rejected(self, tmp_path: Path):
        g1 = _mk_graph(tmp_path)
        g1._acquire_agent_lock()  # 第一个实例持有锁
        g2 = _mk_graph(tmp_path)
        with pytest.raises(RuntimeError, match="另一个 agent 实例"):
            g2._acquire_agent_lock()

    def test_stale_lock_taken_over(self, tmp_path: Path):
        """锁文件残留（pid 已死）→ 可接管。"""
        (tmp_path / ".agent.lock").write_text("99999999")  # 不存在的 pid
        g = _mk_graph(tmp_path)
        g._acquire_agent_lock()  # 不抛错
        assert int((tmp_path / ".agent.lock").read_text()) == os.getpid()

    def test_release_only_own_lock(self, tmp_path: Path):
        g = _mk_graph(tmp_path)
        g._acquire_agent_lock()
        # 模拟另一个实例覆盖锁（极端场景）
        (tmp_path / ".agent.lock").write_text("99999999")
        g._release_agent_lock()
        assert (tmp_path / ".agent.lock").exists()  # 不是本进程的锁 → 不删

    def test_release_own_lock_removes(self, tmp_path: Path):
        g = _mk_graph(tmp_path)
        g._acquire_agent_lock()
        g._release_agent_lock()
        assert not (tmp_path / ".agent.lock").exists()


class TestOrphanDetection:
    def _local_backend(self):
        from core.execution import LocalExecutionBackend
        return LocalExecutionBackend(Path("."))

    def test_alive_orphan_warns(self, tmp_path: Path, caplog):
        """上次 launch 的训练进程还活着 → 警告。"""
        import subprocess
        # 起一个真实存活进程
        proc = subprocess.Popen(["python", "-c", "import time; time.sleep(30)"])
        try:
            (tmp_path / ".last_launch.json").write_text(
                json.dumps({"pid": proc.pid, "ts": 1, "log_file": "x.log"}))
            g = _mk_graph(tmp_path)
            g.monitor = type("M", (), {"backend": self._local_backend()})()
            import logging
            with caplog.at_level(logging.WARNING):
                g._check_orphan_training()
            assert any("残留的训练进程" in r.message for r in caplog.records)
        finally:
            proc.kill()

    def test_dead_orphan_no_warning(self, tmp_path: Path, caplog):
        (tmp_path / ".last_launch.json").write_text(
            json.dumps({"pid": 99999999, "ts": 1, "log_file": "x.log"}))
        g = _mk_graph(tmp_path)
        g.monitor = type("M", (), {"backend": self._local_backend()})()
        import logging
        with caplog.at_level(logging.WARNING):
            g._check_orphan_training()
        assert not any("残留的训练进程" in r.message for r in caplog.records)

    def test_no_launch_record(self, tmp_path: Path):
        g = _mk_graph(tmp_path)
        g.monitor = type("M", (), {"backend": self._local_backend()})()
        g._check_orphan_training()  # 不抛错
