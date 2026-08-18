"""代码阅读工具回归测试 — LangGraph 引擎 (core/nodes.py)。

从旧引擎 (core/tools.py) 迁移。新增的安全断言:
  - 路径遍历 (`..`) 必须被拒绝(迁移时发现 _resolve_path 无边界约束);
  - 符号链接不得被展开(防工作区外内容泄漏)。
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from core.execution import LocalExecutionBackend
from core.nodes import list_tree, read_file, search_code, set_tool_context


def _make_symlink_or_skip(testcase, src: Path, dst: Path):
    """Windows 上创建符号链接需要管理员/开发者模式,失败则跳过该测试。"""
    try:
        os.symlink(src, dst)
    except OSError as exc:
        testcase.skipTest(f"symlinks unavailable on this platform: {exc}")


class RepoReadingToolTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name) / "workspace"
        self.workspace.mkdir()
        (self.workspace / "src").mkdir()
        (self.workspace / "src" / "train.py").write_text(
            "import torch\n"
            "def main():\n"
            "    lr = 1e-3\n"
            "    return lr\n"
        )
        (self.workspace / "README.md").write_text("# Demo\nuses learning rate\n")
        (self.workspace / "__pycache__").mkdir()
        (self.workspace / "__pycache__" / "junk.txt").write_text("def main(): pass\n")
        set_tool_context(self.workspace, LocalExecutionBackend(self.workspace))

    def tearDown(self):
        self.tempdir.cleanup()

    def test_search_code_finds_match_with_file_and_line(self):
        result = json.loads(search_code.func(pattern=r"def main"))
        self.assertEqual(result["count"], 1)
        hit = result["matches"][0]
        self.assertEqual(hit["file"], "src/train.py")
        self.assertEqual(hit["line"], 2)

    def test_search_code_skips_pycache(self):
        result = json.loads(search_code.func(pattern=r"def main"))
        files = {m["file"] for m in result["matches"]}
        self.assertNotIn("__pycache__/junk.txt", files)

    def test_search_code_ignore_case(self):
        result = json.loads(
            search_code.func(pattern="LEARNING", ignore_case=True)
        )
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["matches"][0]["file"], "README.md")

    def test_search_code_invalid_regex_returns_error(self):
        out = search_code.func(pattern="(")
        self.assertIn("regex error", out)

    def test_search_code_rejects_path_traversal(self):
        out = search_code.func(pattern="x", path="../")
        self.assertIn("out of workspace bounds", out)

    def test_list_tree_is_recursive_and_marks_dirs(self):
        tree = list_tree.func()
        self.assertIn("src/", tree)
        self.assertIn("src/train.py", tree)
        self.assertNotIn("__pycache__/", tree)

    def test_list_tree_depth_limit(self):
        tree = list_tree.func(max_depth=1)
        self.assertIn("src/", tree)
        self.assertNotIn("src/train.py", tree)

    def test_read_file_range_returns_slice(self):
        out = read_file.func(path="src/train.py", start_line=2, end_line=3)
        self.assertIn("def main():", out)
        self.assertIn("lr = 1e-3", out)
        self.assertNotIn("import torch", out)

    def test_read_file_without_range_unchanged(self):
        out = read_file.func(path="README.md")
        self.assertEqual(out, "# Demo\nuses learning rate\n")

    def test_list_tree_rejects_path_traversal(self):
        out = list_tree.func(path="..")
        self.assertIn("out of workspace bounds", out)

    def test_list_tree_does_not_follow_symlink_outside_workspace(self):
        outside = Path(self.tempdir.name) / "outside"
        (outside / "sub").mkdir(parents=True)
        (outside / "sub" / "secret.txt").write_text("TOPSECRET\n")
        _make_symlink_or_skip(self, outside, self.workspace / "leak")

        tree = list_tree.func()
        self.assertNotIn("leak", tree)

    def test_search_code_does_not_read_symlinked_external_file(self):
        outside = Path(self.tempdir.name) / "outside"
        outside.mkdir(exist_ok=True)
        (outside / "creds.txt").write_text("TOPSECRET token\n")
        _make_symlink_or_skip(self, outside / "creds.txt", self.workspace / "leak.txt")

        result = json.loads(search_code.func(pattern="TOPSECRET"))
        files = {m["file"] for m in result["matches"]}
        self.assertNotIn("leak.txt", files)
        self.assertEqual(result["count"], 0)


if __name__ == "__main__":
    unittest.main()
