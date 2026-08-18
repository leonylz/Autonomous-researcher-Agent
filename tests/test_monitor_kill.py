"""monitor 超时杀进程策略测试。

锁定的语义（修复背景）：
- 超时 ≠ 卡死。只有"超时 + 日志长时间无更新（stale）"才 SIGTERM
- 超时但日志仍在产出（max_wait_hours 设短了）→ 不杀，训练继续
- interrupted（用户退出）→ 永不杀
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

from core.execution import LocalExecutionBackend
from core.monitor import ExperimentMonitor


class _AlwaysAliveBackend(LocalExecutionBackend):
    """进程永远活着；继承 LocalExecutionBackend 以通过 kill 的类型检查。"""

    def __init__(self, log_path: Path, touch_log: bool = False):
        super().__init__(log_path.parent)
        self.log_path = log_path
        self.touch_log = touch_log

    def is_process_alive(self, pid):
        return True

    def tail_file(self, p, lines=5):
        return ["epoch 1 loss 0.5"]

    def get_gpu_status(self):
        return {"utilization": "10%"}

    def final_status(self, pid):
        return {"state": "unknown", "success": None}


def _spawn_sleep_proc() -> subprocess.Popen:
    """起一个真实可被 kill 探测的进程。"""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )


def test_timeout_with_stale_log_kills(tmp_path: Path):
    """超时 + 日志无更新（stale）→ SIGTERM 发送（进程被终止）。"""
    log = tmp_path / "train.log"
    log.write_text("epoch 1\n", encoding="utf-8")
    # 把日志 mtime 改成 1 小时前 → 满足 stale 判定
    old = time.time() - 3600
    os.utime(log, (old, old))
    proc = _spawn_sleep_proc()
    try:
        m = ExperimentMonitor(poll_interval=0, backend=_AlwaysAliveBackend(log))
        with patch("core.monitor.os.kill") as mock_kill:
            res = m.wait_for_completion(
                proc.pid, str(log), notify=False,
                max_wait_hours=0.00001, stale_after_minutes=0.01)
            assert res["status"] == "interrupted"
            assert res["timed_out"] is True
            # SIGTERM 被调用（真卡死）
            mock_kill.assert_called_once()
            args = mock_kill.call_args[0]
            assert args[0] == proc.pid
            assert args[1] == signal.SIGTERM
    finally:
        proc.kill()


def test_timeout_with_active_log_does_not_kill(tmp_path: Path):
    """超时但日志仍在产出（max_wait_hours 设短）→ 不杀。"""
    log = tmp_path / "train.log"
    log.write_text("epoch 1\n", encoding="utf-8")

    class _ActiveLog(_AlwaysAliveBackend):
        def __init__(self, log_path):
            super().__init__(log_path)
            self._last = 0

        def tail_file(self, p, lines=5):
            # 每轮 touch 日志 → mtime 更新 → 活跃
            now = time.time()
            if now - self._last > 0.05:
                self._last = now
                with open(p, "a", encoding="utf-8") as fh:
                    fh.write(f"epoch {now}\n")
            return ["epoch x"]

    proc = _spawn_sleep_proc()
    try:
        m = ExperimentMonitor(poll_interval=0, backend=_ActiveLog(log))
        with patch("core.monitor.os.kill") as mock_kill:
            res = m.wait_for_completion(
                proc.pid, str(log), notify=False,
                max_wait_hours=0.00001, stale_after_minutes=10)
            assert res["status"] == "interrupted"
            assert res["timed_out"] is True
            mock_kill.assert_not_called()  # 关键：不杀在跑的正常训练
    finally:
        proc.kill()


def test_interrupted_never_kills(tmp_path: Path):
    """用户退出（should_stop）→ 永不 SIGTERM。"""
    log = tmp_path / "train.log"
    log.write_text("epoch 1\n", encoding="utf-8")
    proc = _spawn_sleep_proc()
    try:
        m = ExperimentMonitor(poll_interval=0, backend=_AlwaysAliveBackend(log))
        with patch("core.monitor.os.kill") as mock_kill:
            res = m.wait_for_completion(
                proc.pid, str(log), notify=False,
                should_stop=lambda: True)
            assert res["interrupted"] is True
            mock_kill.assert_not_called()
    finally:
        proc.kill()


class TestExtractMetricsEqualsFormat:
    """_extract_metrics 回归：训练模板用 `key=value` 格式（实测曾全提取为 null）。"""

    def test_equals_separator_extracts_loss_and_accuracy(self):
        m = ExperimentMonitor(poll_interval=1, backend=None)
        lines = ["Epoch 10/10 | loss=0.0149 | acc=0.9920 | 7.7s",
                 "=== 训练完成，best_acc=0.9920 ==="]
        metrics = m._extract_metrics(lines)
        assert metrics["loss"] == "0.0149", metrics
        assert metrics["accuracy"] == "0.9920", metrics
        assert metrics["epoch"] == "10", metrics

    def test_colon_format_still_works(self):
        m = ExperimentMonitor(poll_interval=1, backend=None)
        metrics = m._extract_metrics(["loss: 0.123", "accuracy: 95.2%", "epoch 100/200"])
        assert metrics["loss"] == "0.123", metrics
        assert metrics["accuracy"] == "95.2", metrics
        assert metrics["epoch"] == "100", metrics
