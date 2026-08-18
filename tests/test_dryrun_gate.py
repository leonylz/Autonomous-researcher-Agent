"""系统干跑门测试:dry-run 由系统执行 + 指纹校验(解释器/脚本/依赖)。

覆盖:① 干跑成功写权威 marker;② 干跑失败返回 stderr;③ 干跑后改脚本 →
真实训练被拒;④ 干跑后不改 → 真实训练通过 marker/指纹闸(被模板校验拦下
是预期,证明走到了更深一层)。
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from core.execution import LocalExecutionBackend
from core.nodes import launch_experiment, set_tool_context

DRY_RUN_SCRIPT = '''\
import argparse
import json
import sys
p = argparse.ArgumentParser()
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()
# 模板硬校验关键词(真实脚本基于 core/train_template.py,这里最小满足)
save_every_n_epochs = 1
best_model_pth = "checkpoints/best_model.pth"

def log_metrics(metrics: dict) -> None:
    print("METRIC_JSON " + json.dumps(metrics), flush=True)

if args.dry_run:
    print("DRY RUN PASSED")
else:
    log_metrics({"epoch": 1, "loss": 0.1, "test_acc": 0.5})
    print("REAL RUN")
'''


class DryRunGateTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name) / "workspace"
        self.workspace.mkdir()
        (self.workspace / "train.py").write_text(DRY_RUN_SCRIPT, encoding="utf-8")
        set_tool_context(self.workspace, LocalExecutionBackend(self.workspace),
                         python_exe=sys.executable)

    def tearDown(self):
        # launch 的真实子进程句柄依赖 GC 释放;重试清理防 Windows 文件锁
        import gc
        import time
        for _ in range(10):
            try:
                self.tempdir.cleanup()
                return
            except OSError:
                gc.collect()
                time.sleep(0.2)
        self.tempdir.cleanup()

    def _dry_run(self):
        return json.loads(launch_experiment.func(
            command="python train.py", log_file="dry.log", dry_run=True))

    def _real_launch(self):
        return json.loads(launch_experiment.func(
            command="python train.py", log_file="real.log"))

    def test_system_dry_run_writes_authoritative_marker(self):
        result = self._dry_run()
        self.assertEqual(result["dry_run"], "passed")

        marker = json.loads(
            (self.workspace / "dry_run_log.json").read_text(encoding="utf-8"))
        self.assertTrue(marker["ok"])
        # 解释器 = 绑定解释器(sys.executable),而非 PATH 上的任意 python
        self.assertEqual(marker["interpreter"], str(Path(sys.executable).resolve()))
        self.assertTrue(marker["script_hash"])
        self.assertIn("fingerprint", marker)

    def test_dry_run_failure_returns_stderr(self):
        # 脚本必须满足模板契约(否则被结构校验先拦下,见
        # test_dry_run_rejects_missing_template_structure);此处契约满足,
        # 但运行时抛错 → dry-run 失败应回传 stderr。
        (self.workspace / "train.py").write_text(
            DRY_RUN_SCRIPT + "\nraise RuntimeError('boom')\n", encoding="utf-8")
        result = self._dry_run()
        self.assertEqual(result["dry_run"], "failed")
        self.assertIn("boom", result["error"])
        self.assertFalse((self.workspace / "dry_run_log.json").exists())

    def test_dry_run_rejects_missing_template_structure(self):
        """干跑也必须通过模板结构校验(冒烟实测修复:agent 自写脚本干跑 ok,
        真实启动却被拒 → 返工烧光 max_turns;现在干跑阶段就拦截)。"""
        (self.workspace / "train.py").write_text(
            "print('no contract at all')\n", encoding="utf-8")
        result = self._dry_run()
        self.assertIn("error", result)
        self.assertIn("missing required structure", result["error"])
        # 未执行、未写 marker → 真实启动也不可能被误放行
        self.assertFalse((self.workspace / "dry_run_log.json").exists())

    def test_real_launch_rejected_after_script_modified(self):
        self._dry_run()
        # 干跑后修改脚本 → 指纹不一致 → 真实训练必须被拒
        (self.workspace / "train.py").write_text(
            DRY_RUN_SCRIPT + "\n# modified after dry-run\n", encoding="utf-8")
        result = self._real_launch()
        self.assertIn("error", result)
        self.assertIn("fingerprint mismatch", result["error"])

    def test_real_launch_passes_fingerprint_gates_without_edits(self):
        """干跑后未改动:解释器/脚本/依赖指纹全过 → 真实训练成功启动。

        证明干跑门与指纹校验没有误伤合法流程。
        """
        self._dry_run()
        result = self._real_launch()
        self.assertTrue(result.get("experiment_launched"))
        self.assertNotIn("no successful dry-run detected", str(result))
        self.assertNotIn("fingerprint mismatch", str(result))

    def test_real_launch_without_dry_run_still_rejected(self):
        result = self._real_launch()
        self.assertIn("error", result)
        self.assertIn("no successful dry-run detected", result["error"])


if __name__ == "__main__":
    unittest.main()
