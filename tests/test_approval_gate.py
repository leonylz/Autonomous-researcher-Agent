"""B 批 HITL 测试:三模式审批、缓存复用、launch 拦截、等待结算。"""
import json
import tempfile
import unittest
from pathlib import Path

from core.approval import ApprovalGate
from core.execution import LocalExecutionBackend
from core.nodes import launch_experiment, set_tool_context


class ApprovalModeTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.ws = Path(self.tempdir.name) / "workspace"
        self.ws.mkdir()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_off_mode_no_interception(self):
        gate = ApprovalGate(self.ws, {"mode": "off"})
        needs, reason = gate.needs_approval("launch_experiment", {"command": "x"})
        self.assertFalse(needs)

    def test_exception_mode_intercepts_require_for(self):
        gate = ApprovalGate(self.ws, {"mode": "exception"})
        needs, reason = gate.needs_approval("launch_experiment", {"command": "x"})
        self.assertTrue(needs)
        # 非 require_for 的常规动作不拦
        needs2, _ = gate.needs_approval("read_file", {})
        self.assertFalse(needs2)

    def test_all_mode_intercepts_everything(self):
        gate = ApprovalGate(self.ws, {"mode": "all"})
        needs, _ = gate.needs_approval("read_file", {})
        self.assertTrue(needs)

    def test_cache_reuse(self):
        gate = ApprovalGate(self.ws, {"mode": "exception"})
        key = gate.cache_key_for("launch_experiment", {"command": "python t.py"})
        self.assertIsNone(gate.cached_decision(key))
        gate.cache_decision(key, "approved")
        self.assertEqual(gate.cached_decision(key), "approved")


class LaunchApprovalTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.ws = Path(self.tempdir.name) / "workspace"
        self.ws.mkdir()
        self.gate = ApprovalGate(self.ws, {"mode": "exception"})
        set_tool_context(self.ws, LocalExecutionBackend(self.ws),
                         python_exe="python", approval=self.gate)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_launch_blocked_and_creates_request(self):
        result = json.loads(launch_experiment.func(
            command="python train.py", log_file="out.log"))
        self.assertTrue(result.get("approval_pending"))
        self.assertTrue(result.get("approval_id"))
        self.assertFalse(result["experiment_launched"])
        # PENDING_APPROVALS.md 已写
        self.assertTrue((self.ws / "PENDING_APPROVALS.md").exists())

    def test_approved_cache_bypasses(self):
        key = self.gate.cache_key_for(
            "launch_experiment",
            {"command": "python train.py", "log_file": "out.log"})
        self.gate.cache_decision(key, "approved")
        result = json.loads(launch_experiment.func(
            command="python train.py", log_file="out.log"))
        # 不再进审批:继续走后续检查(无 dry-run → 被 dry-run 门拦)
        self.assertNotIn("approval_pending", result)
        self.assertIn("no successful dry-run detected", result.get("error", ""))

    def test_denied_cache_blocks(self):
        key = self.gate.cache_key_for(
            "launch_experiment",
            {"command": "python train.py", "log_file": "out.log"})
        self.gate.cache_decision(key, "denied")
        result = json.loads(launch_experiment.func(
            command="python train.py", log_file="out.log"))
        self.assertIn("denied", result.get("error", ""))


class WaitSettlementTests(unittest.TestCase):
    """execute_node 的等待结算(直接测 ApprovalGate.wait_for_approval)。"""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.ws = Path(self.tempdir.name) / "workspace"
        self.ws.mkdir()
        self.gate = ApprovalGate(self.ws, {"mode": "exception"})

    def tearDown(self):
        self.tempdir.cleanup()

    def test_approve_via_file(self):
        req = self.gate.create_request("launch_experiment", {"command": "x"})
        # 模拟人工在 PENDING_APPROVALS.md 写 APPROVE(行首,与 dashboard 一致)
        content = (self.ws / "PENDING_APPROVALS.md").read_text(encoding="utf-8")
        content += f"\n[{req.id}]\n**APPROVE**\n"
        (self.ws / "PENDING_APPROVALS.md").write_text(content, encoding="utf-8")
        decision = self.gate.wait_for_approval(req.id, poll_interval=1, timeout=10)
        self.assertEqual(decision, "approved")

    def test_deny_via_file(self):
        req = self.gate.create_request("launch_experiment", {"command": "x"})
        content = (self.ws / "PENDING_APPROVALS.md").read_text(encoding="utf-8")
        content += f"\n[{req.id}]\nYour decision: DENY\n"
        (self.ws / "PENDING_APPROVALS.md").write_text(content, encoding="utf-8")
        decision = self.gate.wait_for_approval(req.id, poll_interval=1, timeout=10)
        self.assertEqual(decision, "denied")

    def test_timeout_returns_denied(self):
        req = self.gate.create_request("launch_experiment", {"command": "x"})
        decision = self.gate.wait_for_approval(req.id, poll_interval=1, timeout=2)
        self.assertEqual(decision, "denied")  # 超时 = 放弃(安全语义)


if __name__ == "__main__":
    unittest.main()
