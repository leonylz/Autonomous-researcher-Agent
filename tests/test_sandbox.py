"""策略沙箱测试:权限分级 + 环境剥离 + 工具门控。"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from core.execution import LocalExecutionBackend
from core.nodes import (
    launch_experiment,
    run_shell,
    set_tool_context,
    write_file,
)
from core.sandbox import Sandbox, resolve_sandbox


class SandboxPolicyTests(unittest.TestCase):
    def test_default_mode_is_workspace_write(self):
        sb = resolve_sandbox(None)
        self.assertEqual(sb.mode, "workspace-write")
        self.assertTrue(sb.allow_write)

    def test_unknown_mode_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "read-only"):
            resolve_sandbox({"mode": "bogus"})

    def test_read_only_rejects_write_and_exec(self):
        sb = Sandbox("read-only")
        self.assertFalse(sb.allow_write)
        self.assertFalse(sb.allow_exec)
        self.assertIsNotNone(sb.reject_reason())

    def test_full_mode_allows_everything(self):
        sb = Sandbox("full")
        self.assertTrue(sb.allow_write)
        self.assertTrue(sb.allow_full)


class SandboxEnvStripTests(unittest.TestCase):
    def test_environment_strips_api_keys_and_keeps_path(self):
        old = dict(os.environ)
        try:
            os.environ["DEEPSEEK_API_KEY"] = "sk-secret"
            os.environ["MY_AUTH_TOKEN"] = "tok"
            os.environ["PATH"] = old.get("PATH", "C:\\Windows")
            sb = Sandbox("workspace-write")
            env = sb.environment()
            self.assertNotIn("DEEPSEEK_API_KEY", env)
            self.assertNotIn("MY_AUTH_TOKEN", env)
            self.assertIn("PATH", env)
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_keep_env_explicitly_allows(self):
        old = dict(os.environ)
        try:
            os.environ["DEEPSEEK_API_KEY"] = "sk-secret"
            sb = Sandbox("workspace-write", keep_env=["DEEPSEEK_API_KEY"])
            env = sb.environment()
            self.assertEqual(env.get("DEEPSEEK_API_KEY"), "sk-secret")
        finally:
            os.environ.clear()
            os.environ.update(old)


class SandboxToolGateTests(unittest.TestCase):
    """挂接验证:read-only 模式下写/执行工具被策略拒绝。"""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name) / "workspace"
        self.workspace.mkdir()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_read_only_blocks_write_file(self):
        set_tool_context(self.workspace, LocalExecutionBackend(self.workspace),
                         sandbox=Sandbox("read-only"))
        result = json.loads(write_file.func(path="x.txt", content="hi"))
        self.assertIn("error", result)
        self.assertIn("read-only", result["error"])
        self.assertFalse((self.workspace / "x.txt").exists())

    def test_read_only_blocks_run_shell(self):
        set_tool_context(self.workspace, LocalExecutionBackend(self.workspace),
                         sandbox=Sandbox("read-only"))
        out = run_shell.func(command="echo hi")
        self.assertIn("read-only", out)

    def test_read_only_blocks_launch_experiment(self):
        set_tool_context(self.workspace, LocalExecutionBackend(self.workspace),
                         sandbox=Sandbox("read-only"))
        result = json.loads(
            launch_experiment.func(command="python train.py", log_file="out.log")
        )
        self.assertIn("error", result)
        self.assertIn("read-only", result["error"])
        self.assertFalse(result["experiment_launched"])

    def test_workspace_write_still_allows_tools(self):
        set_tool_context(self.workspace, LocalExecutionBackend(self.workspace))
        out = write_file.func(path="ok.txt", content="hi")
        self.assertNotIn("error", out)
        self.assertTrue((self.workspace / "ok.txt").exists())


if __name__ == "__main__":
    unittest.main()
