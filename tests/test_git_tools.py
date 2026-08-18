"""git 工具测试:只读、路径边界、非仓库友好降级。"""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from core.nodes import git_diff, git_status, set_tool_context


def _git_available() -> bool:
    return shutil.which("git") is not None


@unittest.skipUnless(_git_available(), "git not on PATH")
class GitToolTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name) / "workspace"
        self.workspace.mkdir()
        set_tool_context(self.workspace, backend=object())
        # 初始化仓库并配置身份(CI 环境可能没有全局配置)
        for argv in (
            ["init", "-q"],
            ["config", "user.email", "test@example.com"],
            ["config", "user.name", "test"],
        ):
            subprocess.run(["git", "-C", str(self.workspace), *argv],
                           check=True, capture_output=True)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_git_status_shows_untracked_files(self):
        (self.workspace / "new.py").write_text("x = 1", encoding="utf-8")
        out = git_status.func()
        self.assertIn("new.py", out)

    def test_git_diff_shows_changes(self):
        f = self.workspace / "train.py"
        f.write_text("a = 1\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.workspace), "add", "train.py"],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(self.workspace), "commit", "-q",
                        "-m", "init"], check=True, capture_output=True)
        f.write_text("a = 2\n", encoding="utf-8")

        out = git_diff.func()
        self.assertIn("a = 2", out)

    def test_git_diff_rejects_path_traversal(self):
        out = git_diff.func(path="../outside")
        self.assertIn("out of workspace bounds", out)

    def test_git_status_not_a_repo_is_graceful(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / "plain"
            plain.mkdir()
            set_tool_context(plain, backend=object())
            out = git_status.func()
            self.assertIn("git", out.lower())  # 友好错误,不崩溃


if __name__ == "__main__":
    unittest.main()
