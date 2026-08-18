"""E5/D 测试:轻量组件注册表 + git_clone 安全约束。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from core.nodes import TOOL_FUNCTIONS, git_clone, set_tool_context
from core.registry import ToolRegistry


class RegistryTests(unittest.TestCase):
    def test_register_appears_in_tool_functions(self):
        reg = ToolRegistry()  # 包装真实 TOOL_FUNCTIONS

        def fake_tool_func(path: str = "."):
            return "fake"

        # 用简单对象模拟 @tool(有 .func 属性)
        fake = type("FakeTool", (), {"func": staticmethod(fake_tool_func),
                                     "description": "fake", "args_schema": None})()
        reg.register("idea", "fake_tool", fake)
        try:
            names = {n for n, _ in TOOL_FUNCTIONS.get("idea", [])}
            self.assertIn("fake_tool", names)
            self.assertIn("fake_tool", reg.all_names())
        finally:
            # 清理注册,避免污染其他测试
            TOOL_FUNCTIONS["idea"] = [
                (n, f) for n, f in TOOL_FUNCTIONS["idea"] if n != "fake_tool"]

    def test_register_replaces_duplicate(self):
        reg = ToolRegistry()
        fake = type("F", (), {"func": staticmethod(lambda: "x"),
                              "description": "", "args_schema": None})()
        reg.register("writing", "write_file", fake)  # 覆盖已有
        names = [n for n, _ in TOOL_FUNCTIONS["writing"]]
        self.assertEqual(names.count("write_file"), 1)


class GitCloneTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.ws = Path(self.tempdir.name) / "workspace"
        self.ws.mkdir()
        set_tool_context(self.ws, backend=object())

    def tearDown(self):
        self.tempdir.cleanup()

    def test_rejects_non_https(self):
        out = git_clone.func(repo_url="git@github.com:user/repo.git")
        self.assertIn("https", out)

    def test_rejects_host_outside_whitelist(self):
        out = git_clone.func(repo_url="https://evil.example.com/x.git")
        self.assertIn("not whitelisted", out)

    def test_rejects_injection_chars(self):
        out = git_clone.func(repo_url="https://github.com/x/y.git;rm -rf /")
        self.assertIn("illegal characters", out)

    def test_clone_into_workspace_repos(self):
        import shutil
        if shutil.which("git") is None:
            self.skipTest("git not on PATH")
        # 本地 file 协议不被允许(仅 https)—— 用 mock 验证目录逻辑
        with unittest.mock.patch("subprocess.run") as mock_run:
            proc = type("P", (), {"returncode": 0, "stderr": ""})()
            mock_run.return_value = proc
            out = git_clone.func(repo_url="https://github.com/pytorch/examples")
        self.assertIn("repos", out)
        self.assertTrue((self.ws / "repos").is_dir())


if __name__ == "__main__":
    unittest.main()
