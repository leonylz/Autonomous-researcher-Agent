"""run_eval.py 冒烟测试:任务发现、dry 校验、scripted 确定性回归。"""
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import pytest  # noqa: E402


def _run_script(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_eval.py"), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(PROJECT_ROOT), timeout=600,
    )


def test_dry_validates_all_tasks():
    proc = _run_script(["--dry"])
    assert proc.returncode == 0, proc.stderr
    for tid in ("T1", "T2", "T3", "T4", "T5"):
        assert tid in proc.stdout


def test_task_metadata_is_wellformed():
    tasks_dir = PROJECT_ROOT / "examples" / "eval_tasks"
    metas = sorted(tasks_dir.glob("*/task.json"))
    assert len(metas) == 6
    for meta_path in metas:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["id"] and meta["name"] and meta["brief"]
        assert meta["target"]["metric"] and meta["target"]["value"]
        assert meta["budget"]["max_cycles"] > 0
        assert (meta_path.parent / meta["brief"]).is_file()


@pytest.mark.slow
def test_scripted_deterministic_regression():
    """ScriptedLLM 驱动完整循环:无需 API key,验证循环健康(事件/账本/锁)。"""
    proc = _run_script(["--scripted"])
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "scripted 确定性回归通过" in proc.stdout
