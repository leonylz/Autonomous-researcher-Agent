"""环境自动创建测试:创建工具选择、命令构造、异步状态机(creating→installing→ready/failed)。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.execution import (
    _env_install_command,
    _env_install_packages_command,
    _env_status,
    _pick_env_creator,
    _settle_env_status,
    _start_training_env_async,
    _write_env_status,
)


class CreatorSelectionTests(unittest.TestCase):
    @patch("core.execution.shutil.which")
    def test_uv_preferred(self, which):
        which.side_effect = lambda name: "/usr/bin/uv" if name == "uv" else None
        self.assertEqual(_pick_env_creator(), "uv")

    @patch("core.execution.shutil.which")
    def test_conda_when_no_uv(self, which):
        which.side_effect = lambda name: "/opt/conda/bin/conda" if name == "conda" else None
        self.assertEqual(_pick_env_creator(), "conda")

    @patch("core.execution.shutil.which")
    def test_venv_fallback(self, which):
        which.return_value = None
        self.assertEqual(_pick_env_creator(), "venv")


class CommandTests(unittest.TestCase):
    def test_uv_commands(self):
        ws = Path("proj")
        create = _env_install_command("uv", ws)
        self.assertEqual(create[0], "uv")
        self.assertTrue(any(".trainenv" in c for c in create))
        install = _env_install_packages_command("uv", ws)
        self.assertEqual(install[:3], ["uv", "pip", "install"])
        self.assertIn("torch", install)

    def test_conda_commands(self):
        ws = Path("proj")
        create = _env_install_command("conda", ws)
        self.assertEqual(create[0], "conda")
        self.assertIn("proj-env", create)
        install = _env_install_packages_command("conda", ws)
        self.assertIn("conda", install)

    def test_venv_commands(self):
        ws = Path("proj")
        create = _env_install_command("venv", ws)
        self.assertIn("-m", create)
        self.assertIn("venv", create)
        install = _env_install_packages_command("venv", ws)
        self.assertIn("pip", install)


class AsyncStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.ws = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_full_state_machine_to_ready(self):
        """creating → (进程结束)→ installing → (进程结束)→ ready + pin。"""
        procs = [MagicMock(pid=111), MagicMock(pid=222)]

        with patch("core.execution._pick_env_creator", return_value="venv"), \
             patch("core.execution.subprocess.Popen", side_effect=procs) as popen, \
             patch("core.execution.pid_alive",
                   side_effect=[True, False, False]), \
             patch("core.execution.check_python_deps", return_value=True):
            self.assertTrue(_start_training_env_async(self.ws))
            # 第一次 settle:creating 且进程存活 → 不变
            st = _settle_env_status(self.ws)
            self.assertEqual(st["status"], "creating")
            # 进程已死 → 推进到 installing
            st = _settle_env_status(self.ws)
            self.assertEqual(st["status"], "installing")
            # install 进程已死 → ready + pin
            st = _settle_env_status(self.ws)
            self.assertEqual(st["status"], "ready")

        self.assertEqual(popen.call_count, 2)
        pin = json.loads((self.ws / ".python_env.json").read_text(encoding="utf-8"))
        self.assertIn("interpreter", pin)

    def test_failed_when_deps_check_fails(self):
        procs = [MagicMock(pid=1), MagicMock(pid=2)]
        with patch("core.execution._pick_env_creator", return_value="venv"), \
             patch("core.execution.subprocess.Popen", side_effect=procs), \
             patch("core.execution.pid_alive", return_value=False), \
             patch("core.execution.check_python_deps", return_value=False):
            _start_training_env_async(self.ws)
            _settle_env_status(self.ws)  # → installing
            st = _settle_env_status(self.ws)  # → failed
            self.assertEqual(st["status"], "failed")
            self.assertIn("deps check failed", st["error"])

    def test_status_file_format(self):
        _write_env_status(self.ws, status="creating", creator="uv",
                          started_at=100.0, pid=1)
        st = _env_status(self.ws)
        self.assertEqual(st["status"], "creating")
        self.assertEqual(st["creator"], "uv")


if __name__ == "__main__":
    unittest.main()
