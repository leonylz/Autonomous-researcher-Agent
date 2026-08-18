"""安全加固回归测试:shell 黑名单绕过面、敏感文件读取、LeaderDecision 字段完整性。"""
import json
import tempfile
import unittest
from pathlib import Path

from core.execution import LocalExecutionBackend
from core.nodes import (
    LeaderDecision,
    read_file,
    run_shell,
    set_tool_context,
    write_file,
)


class ShellGuardBypassTests(unittest.TestCase):
    """高危:zsh/ksh/dash -c、python.exe 大小写、cmd /c 曾可绕过黑名单。"""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name) / "workspace"
        self.workspace.mkdir()
        set_tool_context(self.workspace, LocalExecutionBackend(self.workspace))

    def tearDown(self):
        self.tempdir.cleanup()

    def _blocked(self, command: str) -> bool:
        return "Blocked" in run_shell.func(command=command)

    def test_other_shells_with_dash_c_blocked(self):
        for shell in ("zsh", "ksh", "dash", "fish"):
            self.assertTrue(
                self._blocked(f'{shell} -c "rm -rf /tmp/x"'),
                f"{shell} -c 应被拦截")

    def test_bash_dash_c_still_blocked(self):
        self.assertTrue(self._blocked('bash -c "rm -rf /tmp/x"'))

    def test_python_exe_case_variant_destructive_blocked(self):
        # Windows 上 python.exe / Python.exe 等变体
        for variant in ("python.exe", "Python.exe", "PYTHON"):
            self.assertTrue(
                self._blocked(
                    f'{variant} -c "import os; os.remove(\'x\')"'),
                f"{variant} 破坏性调用应被拦截")

    def test_cmd_slash_c_blocked(self):
        self.assertTrue(self._blocked('cmd /c del file.txt'))

    def test_dangerous_bin_still_blocked(self):
        self.assertTrue(self._blocked("rm -rf tmp"))
        self.assertTrue(self._blocked("RM -rf tmp"))  # 大小写变体


class SensitiveFileTests(unittest.TestCase):
    """高危:agent 曾可 cat .env / read_file(".env") 偷 API key。"""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name) / "workspace"
        self.workspace.mkdir()
        (self.workspace / ".env").write_text(
            "DEEPSEEK_API_KEY=sk-secret\n", encoding="utf-8")
        (self.workspace / "id_rsa").write_text("PRIVATE KEY\n", encoding="utf-8")
        (self.workspace / "train.py").write_text("print(1)\n", encoding="utf-8")
        set_tool_context(self.workspace, LocalExecutionBackend(self.workspace))

    def tearDown(self):
        self.tempdir.cleanup()

    def test_read_file_rejects_env(self):
        out = read_file.func(path=".env")
        self.assertIn("sensitive file", out)

    def test_read_file_rejects_private_key(self):
        out = read_file.func(path="id_rsa")
        self.assertIn("sensitive file", out)

    def test_read_file_rejects_pem_suffix(self):
        (self.workspace / "cert.pem").write_text("CERT\n", encoding="utf-8")
        out = read_file.func(path="cert.pem")
        self.assertIn("sensitive file", out)

    def test_read_file_normal_file_ok(self):
        out = read_file.func(path="train.py")
        self.assertIn("print(1)", out)

    def test_run_shell_cat_env_blocked(self):
        out = run_shell.func(command="cat .env")
        self.assertIn("Blocked", out)

    def test_run_shell_tail_env_blocked(self):
        out = run_shell.func(command="tail -n 5 .env")
        self.assertIn("Blocked", out)

    def test_write_file_env_still_blocked(self):
        result = json.loads(write_file.func(path=".env", content="x=1"))
        self.assertIn("error", result)


class LeaderDecisionSchemaTests(unittest.TestCase):
    """高危:reflect 结构化输出曾丢弃 milestone/decision(MEMORY_LOG 空转)。"""

    def test_schema_has_milestone_and_decision(self):
        if LeaderDecision is None:  # pydantic 缺失时跳过
            self.skipTest("pydantic not available")
        d = LeaderDecision(action="wait",
                           milestone="Exp003: acc=0.79 best so far",
                           decision="try mixup next")
        self.assertEqual(d.milestone, "Exp003: acc=0.79 best so far")
        self.assertEqual(d.decision, "try mixup next")

    def test_think_defaults_are_empty(self):
        if LeaderDecision is None:
            self.skipTest("pydantic not available")
        d = LeaderDecision(action="experiment", task="t")
        self.assertEqual(d.milestone, "")
        self.assertEqual(d.decision, "")


if __name__ == "__main__":
    unittest.main()
