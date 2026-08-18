import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from core.execution import (
    LocalExecutionBackend,
    REMOTE_HELPER,
    SSHExecutionBackend,
    SlurmExecutionBackend,
    build_execution_backend,
    bind_python_argv,
    dryrun_interpreter_error,
    ensure_project_python,
    resolve_project_python,
    _project_venv_python,
    _parse_slurm_time_seconds,
    _SLURM_RUNNING_STATES,
    _SLURM_OK_STATES,
    _SLURM_FAIL_STATES,
)
from core.monitor import ExperimentMonitor
from core.obsidian import ObsidianExporter
from core.memory import MemoryManager


class _Completed:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class FakeBackend:
    def __init__(self, alive=None, tail=None, gpu=None, final=None):
        self.alive = list(alive or [])
        self.tail = list(tail or [])
        self.gpu = gpu or {"utilization": "N/A"}
        self.final = final or {"state": "unknown", "success": None}
        self.calls = []

    def validate(self):
        self.calls.append(("validate",))

    def read_file(self, path):
        self.calls.append(("read_file", path))
        return ""

    def write_file(self, path, content):
        self.calls.append(("write_file", path, content))
        return {"status": "written", "path": path, "bytes": len(content)}

    def list_files(self, path="."):
        self.calls.append(("list_files", path))
        return []

    def run_command(self, argv, timeout=120, env=None):
        self.calls.append(("run_command", argv, timeout, env))
        return {"stdout": "", "stderr": "", "returncode": 0}

    def launch_command(self, argv, log_file, env=None):
        self.calls.append(("launch_command", argv, log_file, env))
        return {"pid": 123, "log_file": log_file, "status": "launched"}

    def is_process_alive(self, pid):
        self.calls.append(("is_process_alive", pid))
        if self.alive:
            return self.alive.pop(0)
        return False

    def tail_file(self, path, lines=50):
        self.calls.append(("tail_file", path, lines))
        if self.tail:
            return self.tail.pop(0)
        return []

    def get_gpu_status(self):
        self.calls.append(("get_gpu_status",))
        return self.gpu

    def final_status(self, pid):
        self.calls.append(("final_status", pid))
        return self.final


class BuildExecutionBackendTests(unittest.TestCase):
    def test_build_local_backend_by_default(self):
        backend = build_execution_backend(config={}, controller_workspace=Path("/tmp/workspace"))
        self.assertIsInstance(backend, LocalExecutionBackend)

    def test_build_ssh_backend(self):
        backend = build_execution_backend(
            config={
                "execution": {
                    "mode": "ssh",
                    "ssh_host": "user@example.com",
                    "remote_workspace": "/remote/ws",
                }
            },
            controller_workspace=Path("/tmp/workspace"),
        )
        self.assertIsInstance(backend, SSHExecutionBackend)
        self.assertEqual(backend.ssh_host, "user@example.com")
        self.assertEqual(backend.remote_workspace, "/remote/ws")

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            build_execution_backend(
                config={"execution": {"mode": "bogus"}},
                controller_workspace=Path("/tmp/workspace"),
            )


class SSHExecutionBackendTests(unittest.TestCase):
    def test_remote_helper_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            outside = Path(tmp) / "outside"
            root.mkdir()
            outside.mkdir()
            try:
                os.symlink(outside, root / "escape")
            except OSError as exc:
                self.skipTest(f"symlinks unavailable on this platform: {exc}")

            payload = {
                "action": "write_file",
                "remote_workspace": str(root),
                "path": "escape/pwned.txt",
                "content": "x",
            }
            proc = subprocess.run(
                [sys.executable, "-c", REMOTE_HELPER],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0)
            body = json.loads(proc.stdout)
            self.assertFalse(body["ok"])
            self.assertIn("escapes workspace", body["error"])
            self.assertFalse((outside / "pwned.txt").exists())

    def _run_helper(self, payload):
        proc = subprocess.run(
            [sys.executable, "-c", REMOTE_HELPER],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout)

    def test_remote_helper_grep_tree_and_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            (root / "pkg").mkdir(parents=True)
            (root / "pkg" / "m.py").write_text("def main():\n    return 1\n")
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "x.py").write_text("def main(): pass\n")

            tree = self._run_helper(
                {"action": "list_tree", "remote_workspace": str(root), "path": "."}
            )
            self.assertTrue(tree["ok"])
            self.assertIn("pkg/", tree["result"]["entries"])
            self.assertIn("pkg/m.py", tree["result"]["entries"])
            self.assertNotIn("__pycache__/", tree["result"]["entries"])

            grep = self._run_helper(
                {"action": "grep_files", "remote_workspace": str(root), "pattern": "def main"}
            )
            self.assertTrue(grep["ok"])
            files = {h["file"] for h in grep["result"]["hits"]}
            self.assertEqual(files, {"pkg/m.py"})
            self.assertEqual(grep["result"]["hits"][0]["line"], 1)

            ranged = self._run_helper(
                {
                    "action": "read_file_range",
                    "remote_workspace": str(root),
                    "path": "pkg/m.py",
                    "start_line": 2,
                    "end_line": 2,
                }
            )
            self.assertTrue(ranged["ok"])
            self.assertEqual(ranged["result"]["content"], "2\t    return 1")

    def test_remote_helper_walk_and_grep_skip_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            root.mkdir()
            outside = Path(tmp) / "outside"
            (outside / "sub").mkdir(parents=True)
            (outside / "creds.txt").write_text("TOPSECRET token\n")
            try:
                os.symlink(outside, root / "leakdir")
                os.symlink(outside / "creds.txt", root / "leak.txt")
            except OSError as exc:
                self.skipTest(f"symlinks unavailable on this platform: {exc}")

            tree = self._run_helper({"action": "list_tree", "remote_workspace": str(root), "path": "."})
            self.assertTrue(tree["ok"])
            self.assertNotIn("leakdir/", tree["result"]["entries"])
            self.assertNotIn("leak.txt", tree["result"]["entries"])

            grep = self._run_helper(
                {"action": "grep_files", "remote_workspace": str(root), "pattern": "TOPSECRET"}
            )
            self.assertTrue(grep["ok"])
            self.assertEqual(grep["result"]["hits"], [])

    @patch("core.execution.shutil.which", return_value="/usr/bin/ssh")
    @patch("core.execution.subprocess.run")
    def test_validate_invokes_remote_helper(self, run_mock, _which_mock):
        run_mock.return_value = _Completed(stdout=json.dumps({"ok": True, "result": {"status": "ok"}}))
        backend = SSHExecutionBackend(
            ssh_host="user@example.com",
            remote_workspace="/remote/ws",
            remote_python="python3",
            ssh_args=["-p", "2222"],
        )

        backend.validate()

        args, kwargs = run_mock.call_args
        self.assertEqual(args[0][:4], ["ssh", "-p", "2222", "user@example.com"])
        self.assertIn("python3 -c", args[0][4])
        self.assertNotIn("import json", args[0][4])
        payload = json.loads(kwargs["input"])
        self.assertEqual(payload["action"], "validate")
        self.assertEqual(payload["remote_workspace"], "/remote/ws")
        self.assertIn("timeout", kwargs)
        self.assertFalse(kwargs["check"])

    @patch("core.execution.subprocess.run")
    def test_run_command_uses_json_stdin_and_no_shell(self, run_mock):
        run_mock.return_value = _Completed(
            stdout=json.dumps({"ok": True, "result": {"stdout": "hi", "stderr": "", "returncode": 0}})
        )
        backend = SSHExecutionBackend("user@example.com", "/remote/ws")

        result = backend.run_command(["python", "train.py"], timeout=42, env={"CUDA_VISIBLE_DEVICES": "0"})

        args, kwargs = run_mock.call_args
        self.assertEqual(args[0][0], "ssh")
        self.assertIn("base64", args[0][-1])
        self.assertNotIn("shell", kwargs)
        payload = json.loads(kwargs["input"])
        self.assertEqual(payload["action"], "run_command")
        self.assertEqual(payload["argv"], ["python", "train.py"])
        self.assertEqual(payload["timeout_seconds"], 42)
        self.assertEqual(payload["env"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(result["stdout"], "hi")

    @patch("core.execution.subprocess.run")
    def test_remote_file_not_found_maps_to_python_exception(self, run_mock):
        run_mock.return_value = _Completed(
            stdout=json.dumps({"ok": False, "error_type": "FileNotFoundError", "error": "File not found: x.txt"})
        )
        backend = SSHExecutionBackend("user@example.com", "/remote/ws")

        with self.assertRaises(FileNotFoundError):
            backend.read_file("x.txt")


class MonitorAndObsidianBackendTests(unittest.TestCase):
    def test_monitor_uses_backend_for_pid_log_and_gpu(self):
        # tail 序列：初始快照(5) → 循环内(5) → 结束(50)，共 3 次
        backend = FakeBackend(
            alive=[True, False],
            tail=[["epoch 1"],
                  ["epoch 1", "epoch 2 accuracy: 0.9"],
                  ["epoch 1", "epoch 2 accuracy: 0.9"]],
            gpu={"utilization": "88%"},
        )
        monitor = ExperimentMonitor(poll_interval=0, backend=backend)
        monitor._active_experiments[123] = {"start_time": time.time(), "status": "running"}

        with patch("core.monitor.time.sleep", return_value=None):
            result = monitor.wait_for_completion(pid=123, log_file="logs/exp.log", notify=False)

        self.assertEqual(result["status"], "completed")
        self.assertIn("epoch 2", result["log_tail"])
        self.assertIn(("get_gpu_status",), backend.calls)
        self.assertIn(("tail_file", "logs/exp.log", 5), backend.calls)
        self.assertIn(("tail_file", "logs/exp.log", 50), backend.calls)

    def test_monitor_reports_failed_from_backend_final_status(self):
        # A backend that reports a failed terminal state -> status "failed",
        # not a silent "completed".
        backend = FakeBackend(
            alive=[True, False],
            tail=[["epoch 1"], ["epoch 1", "Traceback: boom"]],
            final={"state": "FAILED", "success": False},
        )
        monitor = ExperimentMonitor(poll_interval=0, backend=backend)
        monitor._active_experiments[7] = {"start_time": time.time(), "status": "running"}

        with patch("core.monitor.time.sleep", return_value=None):
            result = monitor.wait_for_completion(pid=7, log_file="logs/exp.log", notify=False)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["terminal_state"], "FAILED")
        self.assertFalse(result["success"])
        self.assertIn(("final_status", 7), backend.calls)

    def test_obsidian_dashboard_reads_remote_status_via_backend(self):
        backend = FakeBackend(alive=[True], tail=[["remote epoch 7"]])
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            (project_dir / "PROJECT_BRIEF.md").write_text("Train model")
            workspace = project_dir / "workspace"
            workspace.mkdir()
            (workspace / "state.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "pid": 321,
                        "log_file": "logs/exp.log",
                        "started_at": time.time(),
                    }
                )
            )
            memory = MemoryManager(project_dir=project_dir)
            exporter = ObsidianExporter(
                config={"obsidian": {"enabled": True}},
                project_dir=project_dir,
                backend=backend,
            )

            result = exporter.refresh_dashboard(memory=memory, cycle_count=2)
            dashboard = Path(result["path"]).read_text()

        self.assertIn("TRAINING (PID 321", dashboard)
        self.assertIn("remote epoch 7", dashboard)
        self.assertIn(("is_process_alive", 321), backend.calls)
        self.assertIn(("tail_file", "logs/exp.log", 8), backend.calls)

    def test_obsidian_status_surfaces_failure(self):
        # A failed run must NOT render as IDLE on the dashboard.
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            (project_dir / "PROJECT_BRIEF.md").write_text("Train model")
            (project_dir / "workspace").mkdir()
            exporter = ObsidianExporter(
                config={"obsidian": {"enabled": True}},
                project_dir=project_dir,
                backend=FakeBackend(),
            )
        self.assertEqual(
            exporter._format_status({"status": "failed", "terminal_state": "TIMEOUT"}),
            "FAILED (TIMEOUT)",
        )
        self.assertEqual(exporter._format_status({"status": "failed"}), "FAILED")
        self.assertEqual(exporter._format_status({"status": "no_pid"}), "FAILED (no PID)")
        self.assertEqual(exporter._format_status({"status": "completed"}), "COMPLETED")


class SlurmExecutionBackendTests(unittest.TestCase):
    LOGIN = "user@login-node"

    def _backend(self, **kw):
        defaults = dict(
            ssh_host=self.LOGIN,
            remote_workspace="/nfs/ws",
            slurm_partition="gpu",
            slurm_time="24:00:00",
            slurm_gpus_per_job=1,
        )
        defaults.update(kw)
        return SlurmExecutionBackend(**defaults)

    # --- factory + validation ---

    def test_factory_builds_slurm_backend(self):
        backend = build_execution_backend(
            config={
                "execution": {
                    "mode": "slurm",
                    "ssh_host": self.LOGIN,
                    "remote_workspace": "/nfs/ws",
                    "slurm_partition": "gpu-h200",
                    "slurm_time": "12:00:00",
                    "slurm_gpus_per_job": 2,
                    "ssh_args": ["-p", "2222"],
                }
            },
            controller_workspace=Path("/tmp/workspace"),
        )
        self.assertIsInstance(backend, SlurmExecutionBackend)
        self.assertEqual(backend.slurm_partition, "gpu-h200")
        self.assertEqual(backend.slurm_time, "12:00:00")
        self.assertEqual(backend.slurm_gpus_per_job, 2)
        self.assertEqual(backend.ssh_args, ["-p", "2222"])

    def test_unknown_mode_message_lists_slurm(self):
        with self.assertRaisesRegex(ValueError, "local, ssh, slurm"):
            build_execution_backend(
                config={"execution": {"mode": "bogus"}},
                controller_workspace=Path("/tmp/workspace"),
            )

    def test_validate_requires_partition_and_time(self):
        # partition missing -> raises before any ssh round-trip
        with self.assertRaisesRegex(ValueError, "slurm_partition is required"):
            self._backend(slurm_partition="").validate()
        with self.assertRaisesRegex(ValueError, "slurm_time is required"):
            self._backend(slurm_time="").validate()

    # --- launch (submit-and-exit) ---

    @patch("core.execution.subprocess.run")
    def test_launch_submits_and_parses_job_id(self, run_mock):
        run_mock.return_value = _Completed(
            stdout=json.dumps(
                {"ok": True, "result": {"slurm_job_id": 12345, "log_file": "logs/exp.log"}}
            )
        )
        backend = self._backend(slurm_gpus_per_job=2)

        result = backend.launch_command(
            ["python", "train.py"],
            "logs/exp.log",
            env={"CUDA_VISIBLE_DEVICES": "3", "FOO": "bar"},
        )

        self.assertEqual(result["pid"], 12345)
        self.assertEqual(result["slurm_job_id"], 12345)
        self.assertEqual(result["status"], "submitted")

        args, kwargs = run_mock.call_args
        self.assertEqual(args[0][0], "ssh")            # transport is ssh, no local shell
        self.assertNotIn("shell", kwargs)
        payload = json.loads(kwargs["input"])
        self.assertEqual(payload["action"], "submit_slurm")
        self.assertEqual(payload["argv"], ["python", "train.py"])
        self.assertEqual(payload["partition"], "gpu")
        self.assertEqual(payload["gres"], 2)
        self.assertEqual(payload["env"]["FOO"], "bar")  # remote helper does the CUDA strip

    @patch("core.execution.subprocess.run")
    def test_launch_failure_raises(self, run_mock):
        run_mock.return_value = _Completed(
            stdout=json.dumps(
                {"ok": False, "error_type": "RuntimeError",
                 "error": "sbatch: error: invalid partition specified"}
            )
        )
        with self.assertRaises(RuntimeError):
            self._backend().launch_command(["python", "t.py"], "logs/exp.log")

    # --- liveness: sacct state map + anti-hang bounds ---

    def _alive_with_state(self, sacct_stdout):
        backend = self._backend()
        with patch.object(backend, "_ssh_shell", return_value=_Completed(stdout=sacct_stdout)):
            return backend.is_process_alive(12345)

    def test_is_alive_state_map(self):
        # Drive every enumerated state from the maps themselves so dropping a
        # state from its bucket (e.g. removing COMPLETING from running) regresses.
        for state in _SLURM_RUNNING_STATES:
            self.assertTrue(self._alive_with_state(state + "\n"), state)
        for state in _SLURM_OK_STATES:
            self.assertFalse(self._alive_with_state(state + "\n"), state)
        for state in _SLURM_FAIL_STATES:
            self.assertFalse(self._alive_with_state(state + "\n"), state)
        # Normalization edges + a non-fail indeterminate state.
        self.assertFalse(self._alive_with_state("CANCELLED+\n"))          # '+' suffix stripped
        self.assertFalse(self._alive_with_state("CANCELLED by 1001\n"))   # ' by <uid>' stripped
        # PREEMPTED is not a fail state -> indeterminate -> kept alive (1st grace poll)
        self.assertTrue(self._alive_with_state("PREEMPTED\n"))

    def test_is_alive_sacct_nonzero_rc_is_unknown_grace(self):
        # sacct exits non-zero (transient accounting error) -> indeterminate,
        # NOT dead: keep the job alive for the bounded grace window.
        backend = self._backend(slurm_unknown_grace_polls=2)
        with patch.object(backend, "_ssh_shell", return_value=_Completed(stdout="", returncode=1)):
            self.assertEqual([backend.is_process_alive(555) for _ in range(3)], [True, True, False])

    def test_is_alive_ssh_failure_is_unknown_grace(self):
        # ssh timeout -> indeterminate, NOT dead.
        backend = self._backend(slurm_unknown_grace_polls=2)
        with patch.object(backend, "_ssh_shell",
                          side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=15)):
            self.assertEqual([backend.is_process_alive(556) for _ in range(3)], [True, True, False])

    def test_is_alive_pending_never_reaped_by_wallclock(self):
        # A job sacct still reports PENDING must NOT be reaped even long past
        # --time + buffer (queue wait is not bounded by --time).
        backend = self._backend(slurm_time="00:01:00", slurm_time_buffer=0)  # 60s cap
        with patch.object(backend, "_ssh_shell", return_value=_Completed(stdout="PENDING\n")):
            with patch("core.execution.time.time", side_effect=[1000.0, 1000.0 + 100000]):
                self.assertTrue(backend.is_process_alive(99))   # first poll
                self.assertTrue(backend.is_process_alive(99))   # 100000s later, still PENDING

    def test_is_alive_unknown_is_bounded(self):
        """Regression guard: a vanished/unreachable job must NOT hang forever."""
        backend = self._backend(slurm_unknown_grace_polls=3)
        # sacct empty AND squeue empty on every probe -> 'unknown' every time.
        with patch.object(backend, "_ssh_shell", return_value=_Completed(stdout="")):
            results = [backend.is_process_alive(777) for _ in range(4)]
        self.assertEqual(results, [True, True, True, False])

    @patch("core.execution.time.time")
    def test_is_alive_wallclock_cap(self, time_mock):
        backend = self._backend(slurm_time="00:01:00", slurm_time_buffer=0)  # 60s cap
        time_mock.side_effect = [1000.0, 1000.0 + 120]  # first seeds, second is past cap
        with patch.object(backend, "_ssh_shell", return_value=_Completed(stdout="")):
            self.assertTrue(backend.is_process_alive(42))   # within cap, unknown -> grace
            self.assertFalse(backend.is_process_alive(42))  # past --time+buffer -> reaped

    @patch("core.execution.subprocess.run")
    def test_liveness_reuses_host_and_args(self, run_mock):
        run_mock.return_value = _Completed(stdout="RUNNING\n")
        backend = self._backend(ssh_args=["-p", "2222"])

        self.assertTrue(backend.is_process_alive(12345))

        args, _ = run_mock.call_args
        self.assertEqual(args[0][:4], ["ssh", "-p", "2222", self.LOGIN])
        self.assertIn("sacct -j 12345", args[0][4])
        self.assertIn("State%30", args[0][4])              # explicit width, no truncation

    def test_final_status_reflects_terminal_state(self):
        backend = self._backend()
        # A COMPLETED job -> success True
        with patch.object(backend, "_ssh_shell", return_value=_Completed(stdout="COMPLETED\n")):
            self.assertFalse(backend.is_process_alive(1))   # records terminal state
        self.assertEqual(backend.final_status(1), {"state": "COMPLETED", "success": True})
        # A TIMEOUT job -> success False (not silently "completed")
        with patch.object(backend, "_ssh_shell", return_value=_Completed(stdout="TIMEOUT\n")):
            self.assertFalse(backend.is_process_alive(2))
        self.assertEqual(backend.final_status(2), {"state": "TIMEOUT", "success": False})
        # Never observed reaching a terminal state -> indeterminate
        self.assertEqual(backend.final_status(999), {"state": "unknown", "success": None})

    def test_get_gpu_status_parses_queue(self):
        backend = self._backend()
        with patch.object(backend, "_ssh_shell", return_value=_Completed(stdout="   2 PENDING\n   1 RUNNING\n")):
            status = backend.get_gpu_status()
        self.assertEqual(status["utilization"], "slurm")
        self.assertEqual(status["pending"], 2)
        self.assertEqual(status["running"], 1)

    def test_cancel_calls_scancel(self):
        backend = self._backend()
        with patch.object(backend, "_ssh_shell", return_value=_Completed(returncode=0)) as shell:
            self.assertTrue(backend.cancel(12345))
        shell.assert_called_once()
        self.assertIn("scancel 12345", shell.call_args[0][0])
        # non-zero scancel -> False (not "return True unconditionally")
        with patch.object(backend, "_ssh_shell", return_value=_Completed(returncode=1)):
            self.assertFalse(backend.cancel(12345))
        # transport failure is swallowed -> False, never propagated
        with patch.object(backend, "_ssh_shell",
                          side_effect=subprocess.TimeoutExpired(cmd="scancel", timeout=8)):
            self.assertFalse(backend.cancel(12345))

    def test_parse_slurm_time_seconds(self):
        self.assertEqual(_parse_slurm_time_seconds("60"), 3600)            # bare minutes
        self.assertEqual(_parse_slurm_time_seconds("01:30"), 90)           # minutes:seconds
        self.assertEqual(_parse_slurm_time_seconds("12:00:00"), 43200)     # h:m:s
        self.assertEqual(_parse_slurm_time_seconds("2-00:00:00"), 172800)  # days-h:m:s
        self.assertEqual(_parse_slurm_time_seconds("1-12"), 129600)        # days-hours
        self.assertEqual(_parse_slurm_time_seconds("garbage"), 10 ** 9)    # sentinel


class SlurmRemoteHelperTests(unittest.TestCase):
    """Run the embedded REMOTE_HELPER as a subprocess (sbatch is absent here, so
    submission fails AFTER the script is written — we assert on the script)."""

    def _run_helper(self, payload):
        proc = subprocess.run(
            [sys.executable, "-c", REMOTE_HELPER],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout)

    def test_submit_slurm_builds_safe_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            root.mkdir()
            self._run_helper(
                {
                    "action": "submit_slurm",
                    "remote_workspace": str(root),
                    "argv": ["python", "t.py", "--x", "a b"],
                    "log_file": "logs/exp.log",
                    "env": {"CUDA_VISIBLE_DEVICES": "3", "FOO": "b a r"},
                    "partition": "gpu",
                    "time": "01:00:00",
                    "gres": 2,
                    "raw_gres": "",
                    "qos": "",
                    "account": "",
                    "job_name": "ar_exp",
                    "setup": "module load cuda/12.4",
                    "extra_sbatch": ["--nodes=1"],
                }
            )
            # The output-log parent must be pre-created (Slurm won't make it).
            self.assertTrue((root / "logs").is_dir())
            script = (root / ".sbatch_ar_exp").read_text()

        self.assertIn("#SBATCH --partition=gpu", script)
        self.assertIn("#SBATCH --time=01:00:00", script)
        self.assertIn('#SBATCH --output="logs/exp.log"', script)   # quoted (whitespace-safe)
        self.assertIn("#SBATCH --gres=gpu:2", script)
        self.assertIn("#SBATCH --nodes=1", script)
        self.assertIn("module load cuda/12.4", script)
        # env quoted safely; injection-prone arg quoted; GPU mask stripped.
        self.assertIn("export FOO='b a r'", script)
        self.assertIn("'a b'", script)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", script)
        # No persistent login-node construct (the 2026-05-29 MIL invariant).
        for forbidden in ("tmux", "srun", "--wait", "squeue", "while "):
            self.assertNotIn(forbidden, script)

    def _run_helper_with_path(self, payload, extra_path):
        env = {**os.environ, "PATH": extra_path + os.pathsep + os.environ["PATH"]}
        proc = subprocess.run(
            [sys.executable, "-c", REMOTE_HELPER],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout)

    @staticmethod
    def _fake_sbatch(bindir, body_line):
        fake = bindir / "sbatch"
        fake.write_text("#!/bin/bash\n" + body_line + "\n")
        fake.chmod(0o755)

    def _submit_payload(self, root, job_name):
        return {
            "action": "submit_slurm", "remote_workspace": str(root),
            "argv": ["python", "t.py"], "log_file": "out.log", "env": {},
            "partition": "gpu", "time": "01:00:00", "gres": 1,
            "raw_gres": "", "job_name": job_name,
        }

    def test_submit_slurm_parses_federated_job_id(self):
        # sbatch --parsable can emit the federated "<id>;<cluster>" form.
        if os.name == "nt":
            self.skipTest("fake sbatch is a POSIX shell script")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"; root.mkdir()
            binp = Path(tmp) / "bin"; binp.mkdir()
            self._fake_sbatch(binp, "printf '12345;cluster0\\n'")
            body = self._run_helper_with_path(self._submit_payload(root, "ar_fed"), str(binp))
        self.assertTrue(body["ok"], body)
        self.assertEqual(body["result"]["slurm_job_id"], 12345)

    def test_submit_slurm_rejects_non_numeric_output(self):
        # A non --parsable line (e.g. "Submitted batch job 99") must be rejected,
        # not mis-parsed into a bogus job id.
        if os.name == "nt":
            self.skipTest("fake sbatch is a POSIX shell script")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"; root.mkdir()
            binp = Path(tmp) / "bin"; binp.mkdir()
            self._fake_sbatch(binp, "printf 'Submitted batch job 99\\n'")
            body = self._run_helper_with_path(self._submit_payload(root, "ar_bad"), str(binp))
        self.assertFalse(body["ok"])
        self.assertIn("did not return a job id", body["error"])

    def test_submit_slurm_raw_gres_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            root.mkdir()
            self._run_helper(
                {
                    "action": "submit_slurm",
                    "remote_workspace": str(root),
                    "argv": ["python", "t.py"],
                    "log_file": "out.log",
                    "env": {},
                    "partition": "gpu",
                    "time": "01:00:00",
                    "gres": 1,
                    "raw_gres": "gpu:a100:4",
                    "job_name": "ar_raw",
                }
            )
            script = (root / ".sbatch_ar_raw").read_text()
        self.assertIn("#SBATCH --gres=gpu:a100:4", script)
        self.assertNotIn("--gres=gpu:1", script)


class LocalBackendAliveCheckTests(unittest.TestCase):
    """is_process_alive 回归测试。

    Windows 上 os.kill(pid, 0) 对「已退出但进程对象尚未回收」的 pid 会误报
    存活，导致 monitor 零 LLM 轮询在死进程上无限空转（真实事故：训练结束
    后 monitor 卡死 5+ 分钟）。修复后必须满足：
      - 已退出进程（即使 Popen 句柄仍持有）→ False
      - 存活进程 → True
    """

    def setUp(self):
        self.backend = LocalExecutionBackend(Path(tempfile.mkdtemp()))

    def test_exited_process_is_dead_even_with_held_handle(self):
        # 复现事故现场：launch_command 用 Popen 启动、句柄不回收，
        # 子进程很快退出。旧实现（os.kill）在此场景误报 alive。
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(0.5)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            proc.wait(timeout=30)  # 子进程已退出；proc 句柄仍被持有
            self.assertFalse(self.backend.is_process_alive(proc.pid),
                             "已退出进程被误判为存活（monitor 会死循环）")
        finally:
            # 句柄一直持有到 assert 之后——正是要复现的「Popen 未回收」场景
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=30)

    def test_live_process_is_alive(self):
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            self.assertTrue(self.backend.is_process_alive(proc.pid),
                            "存活进程被误判为已退出（会提前 reap 训练）")
        finally:
            proc.kill()
            proc.wait(timeout=30)

    def test_never_spawned_pid_is_dead(self):
        # 找一个必然不存在的 pid：spawn 一个进程拿到 pid 后等它退出，
        # 句柄关闭后内核可能已回收对象——判定必须是 False。
        proc = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        pid = proc.pid
        proc.wait(timeout=30)
        del proc
        time.sleep(0.2)  # 给内核回收窗口
        self.assertFalse(self.backend.is_process_alive(pid))


class TestPythonEnvBinding(unittest.TestCase):
    """训练解释器绑定回归（2026-08-13 事故：run_shell 与 launch 两条路径
    把 `python` 解析成两个解释器——干跑过、真训练崩）。"""

    def test_bind_bare_python(self):
        argv = bind_python_argv(["python", "train.py", "--epochs", "10"],
                                r"D:\Anaconda\envs\test\python.exe")
        self.assertEqual(argv, [r"D:\Anaconda\envs\test\python.exe",
                                "train.py", "--epochs", "10"])

    def test_bind_python_exe_and_py(self):
        for bare in ("python.exe", "python3", "py"):
            argv = bind_python_argv([bare, "train.py"], "/usr/bin/python")
            self.assertEqual(argv[0], "/usr/bin/python", bare)

    def test_absolute_path_unchanged(self):
        exe = r"D:\Anaconda\envs\test\python.exe"
        argv = bind_python_argv([exe, "train.py"], "/other/python")
        self.assertEqual(argv, [exe, "train.py"])

    def test_non_python_launcher_unchanged(self):
        # .bat 等自绑环境的启动器不受影响
        argv = bind_python_argv(["run_training.bat"], "/other/python")
        self.assertEqual(argv, ["run_training.bat"])

    def test_empty_python_exe_noop(self):
        argv = bind_python_argv(["python", "train.py"], "")
        self.assertEqual(argv, ["python", "train.py"])


class TestDryrunInterpreterCheck(unittest.TestCase):
    def test_mismatch_returns_error(self):
        err = dryrun_interpreter_error(
            {"interpreter": r"D:\Anaconda\envs\test\python.exe"},
            r"D:\Anaconda\python.exe")
        self.assertTrue(err and "does not match" in err, err)

    def test_match_returns_empty(self):
        self.assertEqual(
            dryrun_interpreter_error({"interpreter": r"D:\Anaconda\python.exe"},
                                     r"D:\Anaconda\python.exe"), "")

    def test_missing_field_returns_empty(self):
        # 旧版干跑记录没有 interpreter 字段 → 不拦截（向后兼容）
        self.assertEqual(dryrun_interpreter_error({"steps": 1}, "/x/python"), "")

    def test_none_data_returns_empty(self):
        self.assertEqual(dryrun_interpreter_error(None, "/x/python"), "")


class TestResolveProjectPython(unittest.TestCase):
    """解析逻辑测试（patch 依赖检查与候选列表，不依赖本机环境）。"""

    def test_config_specified_and_ok(self):
        with patch("core.execution.check_python_deps", return_value=True):
            exe = r"D:\Anaconda\envs\test\python.exe"
            out = resolve_project_python({"execution": {"python": exe}}, Path("."))
            self.assertEqual(out, exe)

    def test_config_specified_but_missing_deps_falls_back_to_probe(self):
        """指定的路径失效（换机器）→ 退化探测，而不是返回空（可移植性）。"""
        with patch("core.execution._python_candidates",
                   return_value=[r"D:\other\python.exe"]), \
             patch("core.execution.check_python_deps",
                   side_effect=lambda p, timeout=30: p == r"D:\other\python.exe"), \
             patch("core.execution._cuda_capable", return_value=False):
            out = resolve_project_python(
                {"execution": {"python": r"D:\x\python.exe"}}, Path("."))
            self.assertEqual(out, r"D:\other\python.exe")

    def test_probe_picks_first_candidate_with_deps(self):
        with patch("core.execution._python_candidates",
                   return_value=[r"D:\a\python.exe", r"D:\b\python.exe"]), \
             patch("core.execution.check_python_deps",
                   side_effect=lambda p, timeout=30: p == r"D:\b\python.exe"), \
             patch("core.execution._cuda_capable", return_value=False):
            out = resolve_project_python({}, Path("."))
            self.assertEqual(out, r"D:\b\python.exe")

    def test_probe_prefers_cuda_capable_env(self):
        """多个合格环境时优先 CUDA 可用的（真实场景：adcd 与 test 并存）。"""
        with patch("core.execution._python_candidates",
                   return_value=[r"D:\a\python.exe", r"D:\b\python.exe"]), \
             patch("core.execution.check_python_deps", return_value=True), \
             patch("core.execution._cuda_capable",
                   side_effect=lambda p, timeout=30: p == r"D:\b\python.exe"):
            out = resolve_project_python({}, Path("."))
            self.assertEqual(out, r"D:\b\python.exe")

    def test_nothing_resolved_returns_empty(self):
        with patch("core.execution._python_candidates", return_value=[]), \
             patch("core.execution.check_python_deps", return_value=False), \
             patch("core.execution._cuda_capable", return_value=False):
            self.assertEqual(resolve_project_python({}, Path(".")), "")


class TestEnsureProjectPython(unittest.TestCase):
    """环境策略：config → 钉住记录 → 项目 venv → 创建 → 借用(兜底)。"""

    def _deps_ok_for(self, ok_paths):
        """check_python_deps 只对指定路径返回 True。"""
        def _check(path):
            return str(path) in ok_paths
        return patch("core.execution.check_python_deps", side_effect=_check)

    def test_config_spec_wins_and_pins(self):
        exe = r"D:\Anaconda\envs\test\python.exe"
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            with self._deps_ok_for({exe}):
                out = ensure_project_python(
                    {"execution": {"python": exe}}, ws)
            self.assertEqual(out, exe)
            # 解析结果被钉住
            pin = json.loads((ws / ".python_env.json").read_text(encoding="utf-8"))
            self.assertEqual(pin["interpreter"], str(Path(exe).resolve()))

    def test_pinned_env_reused_without_rescanning(self):
        """钉住记录存在且可用 → 直接复用,不再探测/创建(跨轮跨进程一致)。"""
        pinned = r"D:\Anaconda\envs\pinned\python.exe"
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / ".python_env.json").write_text(
                json.dumps({"interpreter": pinned, "pinned_at": 1.0}),
                encoding="utf-8")
            with self._deps_ok_for({pinned}), \
                 patch("core.execution._create_training_env") as mock_create, \
                 patch("core.execution.resolve_project_python") as mock_resolve:
                out = ensure_project_python({}, ws)
            self.assertEqual(out, pinned)
            mock_create.assert_not_called()
            mock_resolve.assert_not_called()

    def test_project_venv_used_before_borrowing(self):
        """项目 venv 存在且可用 → 用之,不借 base(隔离)。"""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            venv = _project_venv_python(ws)
            with self._deps_ok_for({venv}), \
                 patch("core.execution._start_training_env_async") as mock_create, \
                 patch("core.execution.resolve_project_python") as mock_resolve:
                out = ensure_project_python({}, ws)
            self.assertEqual(out, venv)
            mock_create.assert_not_called()
            mock_resolve.assert_not_called()

    def test_creates_project_env_async_when_nothing_else(self):
        """无任何可用环境 → 异步启动创建并返回 ''(launch 轮询状态)。"""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            with patch("core.execution.check_python_deps", return_value=False), \
                 patch("core.execution._start_training_env_async",
                       return_value=True) as mock_create:
                out = ensure_project_python({}, ws)
            self.assertEqual(out, "")
            mock_create.assert_called_once()

    def test_auto_create_disabled_and_no_pin_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            with patch("core.execution.check_python_deps", return_value=False), \
                 patch("core.execution._start_training_env_async") as mock_create, \
                 patch("core.execution.resolve_project_python", return_value=""):
                out = ensure_project_python(
                    {"execution": {"auto_create_env": False}}, ws)
            self.assertEqual(out, "")
            mock_create.assert_not_called()

    def test_borrow_is_last_resort_and_pins(self):
        borrowed = r"D:\Anaconda\python.exe"
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            with patch("core.execution.check_python_deps", return_value=False), \
                 patch("core.execution._start_training_env_async",
                       return_value=False) as mock_create, \
                 patch("core.execution.resolve_project_python",
                       return_value=borrowed):
                out = ensure_project_python({}, ws)
            self.assertEqual(out, borrowed)
            mock_create.assert_called_once()
            pin = json.loads((ws / ".python_env.json").read_text(encoding="utf-8"))
            self.assertEqual(pin["interpreter"], str(Path(borrowed).resolve()))

    def test_create_is_idempotent_while_creating(self):
        """创建中再次解析 → 不重复启动创建(幂等)。"""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            with patch("core.execution.check_python_deps", return_value=False), \
                 patch("core.execution._start_training_env_async",
                       return_value=True) as mock_create:
                # 第一次:启动创建(返回 "")
                first = ensure_project_python({}, ws)
                self.assertEqual(first, "")
                # 模拟创建中状态 → 第二次不重复启动
                from core.execution import _write_env_status
                _write_env_status(ws, status="creating", creator="uv",
                                  started_at=1.0, pid=1)
                second = ensure_project_python({}, ws)
                self.assertEqual(second, "")
            mock_create.assert_called_once()


class TestLaunchLazyEnvResolution(unittest.TestCase):
    """launch 时惰性重解析：系统启动时没解析到环境，但 agent 中途修复
    （如 pip install / 建 env）后，launch 会重解析并接管绑定。"""

    def test_launch_reresolves_after_agent_fixed_env(self):
        import core.nodes as nodes_mod
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "dry_run_log.json").write_text(
                '{"status": "ok", "steps": 1}', encoding="utf-8")
            with patch.object(nodes_mod, "_tool_python", ""), \
                 patch.object(nodes_mod, "_tool_config", {"execution": {}}), \
                 patch.object(nodes_mod, "_tool_workspace", ws), \
                 patch.object(nodes_mod, "ensure_project_python",
                              return_value=r"D:\fixed\python.exe") as mock_ensure, \
                 patch.object(nodes_mod, "_tool_backend") as backend:
                backend.launch_command.return_value = {
                    "pid": 1, "log_file": "x.log", "status": "launched"}
                out = nodes_mod.launch_experiment.func("python train.py", "x.log", "0")
                data = json.loads(out)
                self.assertTrue(data.get("experiment_launched"), out)
                mock_ensure.assert_called_once()
                call_argv = backend.launch_command.call_args.kwargs.get("argv")
                self.assertEqual(call_argv[0], r"D:\fixed\python.exe")

    def test_launch_errors_when_still_unresolved(self):
        import core.nodes as nodes_mod
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "dry_run_log.json").write_text(
                '{"status": "ok", "steps": 1}', encoding="utf-8")
            with patch.object(nodes_mod, "_tool_python", ""), \
                 patch.object(nodes_mod, "_tool_config", {"execution": {}}), \
                 patch.object(nodes_mod, "_tool_workspace", ws), \
                 patch.object(nodes_mod, "_tool_backend", FakeBackend()), \
                 patch.object(nodes_mod, "ensure_project_python", return_value=""):
                out = nodes_mod.launch_experiment.func("python train.py", "x.log", "0")
                data = json.loads(out)
                self.assertFalse(data.get("experiment_launched"), out)
                self.assertIn("no usable training interpreter", data.get("error", ""), out)


if __name__ == "__main__":
    unittest.main()
