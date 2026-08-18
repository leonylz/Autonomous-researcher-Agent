"""安全回归测试 — LangGraph 引擎 (core/nodes.py) 的工具层。

从旧引擎 (core/tools.py ToolRegistry) 迁移,同时堵住迁移中发现的两个
真回归:
  1. run_shell 曾用 shell=True → `;`/`&&` 注入可绕过二进制黑名单真正执行;
  2. _resolve_path 曾无工作区边界约束 → `..` out of workspace bounds读写;
  3. list_tree/search_code 曾跟随符号链接 → 工作区外内容泄漏。
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

from core.execution import LocalExecutionBackend
from core.nodes import (
    launch_experiment,
    list_files,
    run_shell,
    set_tool_context,
    write_file,
)


class ToolSecurityTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name) / "workspace"
        self.workspace.mkdir()
        set_tool_context(self.workspace, LocalExecutionBackend(self.workspace))

    def tearDown(self):
        self.tempdir.cleanup()

    # ── 路径out of workspace bounds（`..` / 绝对路径一律拒绝）──

    def test_write_file_rejects_parent_escape(self):
        result = json.loads(write_file.func(path="../pwned.txt", content="x"))
        self.assertIn("error", result)
        self.assertIn("out of workspace bounds", result["error"])
        self.assertFalse((Path(self.tempdir.name) / "pwned.txt").exists())

    def test_write_file_rejects_absolute_path(self):
        result = json.loads(
            write_file.func(path=str(Path(self.tempdir.name) / "abs.txt"), content="x")
        )
        self.assertIn("error", result)
        self.assertIn("out of workspace bounds", result["error"])

    def test_write_file_rejects_posix_absolute_path(self):
        """POSIX 风格绝对路径必须拒绝,绝不能 strip('/') 后静默写入工作区内同名路径。"""
        result = json.loads(write_file.func(path="/etc/evil.txt", content="x"))
        self.assertIn("error", result)
        self.assertIn("out of workspace bounds", result["error"])
        self.assertFalse((self.workspace / "etc" / "evil.txt").exists())

    def test_list_files_rejects_parent_escape(self):
        out = list_files.func(path="..")
        self.assertIn("out of workspace bounds", out)

    def test_write_file_rejects_protected_file(self):
        (self.workspace / "PROJECT_BRIEF.md").write_text("brief")
        result = json.loads(write_file.func(path="PROJECT_BRIEF.md", content="pwned"))
        self.assertIn("error", result)
        self.assertIn("protected file", result["error"])
        self.assertEqual((self.workspace / "PROJECT_BRIEF.md").read_text(), "brief")

    # ── run_shell:无 shell 执行,注入在机制上不可能 ──

    def test_run_shell_does_not_execute_shell_injection_payload(self):
        """`;` 不是分隔符:第二个命令绝不执行(injected.txt 不产生)。

        旧引擎用 shlex argv 无 shell 执行,迁移时曾回归为 shell=True(可注入),
        现在恢复为 argv 执行 —— 该属性在 Windows/Linux 上都成立。
        payload 用 python(两平台都有)而非 echo(Windows 上是 cmd 内建)。
        """
        result = run_shell.func(
            command='python -c "print(\'hello\')"; touch injected.txt'
        )
        self.assertIn("hello", result)
        self.assertIn("[exit_code=0]", result)
        self.assertFalse((self.workspace / "injected.txt").exists())

    def test_run_shell_blocks_dangerous_binaries(self):
        result = run_shell.func(command="rm -rf tmp")
        self.assertIn("Blocked executable", result)

    def test_run_shell_blocks_bash_c_injection(self):
        result = run_shell.func(command="bash -c \"rm -rf tmp\"")
        self.assertIn("Blocked", result)

    def test_run_shell_blocks_reading_secrets(self):
        result = run_shell.func(command="echo $DEEPSEEK_API_KEY")
        self.assertIn("Blocked", result)

    def test_run_shell_blocks_destructive_python(self):
        result = run_shell.func(command="python -c \"import os; os.remove('x')\"")
        self.assertIn("Blocked", result)

    # ── launch_experiment:log_file out of workspace bounds拦截（在 dry-run 检查之前）──

    def test_launch_experiment_rejects_log_path_traversal(self):
        result = json.loads(
            launch_experiment.func(command="python train.py", log_file="../outside.log")
        )
        self.assertIn("error", result)
        self.assertIn("out of workspace bounds", result["error"])
        self.assertIn("experiment_launched", result)

    def test_launch_rejects_when_training_already_running(self):
        """幂等性:已有活跃训练(pid 存活)时,重复 launch 必须被拒绝。

        防重复扣 GPU 时长 —— 同一循环内误触发两次 launch 是真实事故面。
        """
        import os

        (self.workspace / ".last_launch.json").write_text(
            json.dumps({"pid": os.getpid(), "ts": 1, "log_file": "prev.log"}),
            encoding="utf-8",
        )
        set_tool_context(self.workspace, LocalExecutionBackend(self.workspace))
        result = json.loads(
            launch_experiment.func(command="python train.py", log_file="out.log")
        )
        self.assertIn("error", result)
        self.assertIn("training already running", result["error"])
        self.assertFalse(result["experiment_launched"])

    def test_launch_allows_after_previous_training_exited(self):
        """上一个训练已结束(pid 不存在)→ 允许再次 launch。"""
        (self.workspace / ".last_launch.json").write_text(
            json.dumps({"pid": 999999999, "ts": 1, "log_file": "prev.log"}),
            encoding="utf-8",
        )
        set_tool_context(self.workspace, LocalExecutionBackend(self.workspace))
        result = json.loads(
            launch_experiment.func(command="python train.py", log_file="out.log")
        )
        # 不被幂等拦截(后续被 dry-run 门拦下是预期)
        self.assertNotIn("training already running", result.get("error", ""))

    def test_launch_template_check_not_bypassed_by_trailing_args(self):
        """脚本名提取必须找 argv 里的 .py 参数,而不是 command.split()[-1]。

        旧逻辑对 `python train.py --lr 0.1` 取到 "0.1" → 模板硬校验被跳过。
        """
        (self.workspace / "train.py").write_text("print('no checkpoint logic')")
        (self.workspace / "dry_run_log.json").write_text(
            json.dumps({"interpreter": sys.executable, "ok": True}),
            encoding="utf-8",
        )
        set_tool_context(self.workspace, LocalExecutionBackend(self.workspace),
                         python_exe=sys.executable)
        result = json.loads(
            launch_experiment.func(command="python train.py --lr 0.1",
                                   log_file="out.log")
        )
        self.assertIn("error", result)
        # 模板校验必须被触发(而不是静默跳过)
        self.assertIn("missing required structure", result["error"])


if __name__ == "__main__":
    unittest.main()
