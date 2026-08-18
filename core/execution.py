"""
Execution backends for Deep Researcher Agent.

Local mode preserves the current behavior. SSH mode keeps the controller
state local while running file operations, shell commands, training, log
tailing, PID checks, and GPU inspection on one remote host.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import base64
import shutil
import shlex
import subprocess
import sys
import textwrap
import time
from pathlib import Path, PurePosixPath
from typing import Optional

logger = logging.getLogger("autoresearcher.execution")

# 传给子进程的环境变量名中含这些片段即视为敏感（API key / token / secret），
# 一律不传给训练/Shell 子进程。训练脚本不需要 LLM API key。
SENSITIVE_ENV_PATTERNS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH", "API_KEY")


def _scrub_env(extra: Optional[dict] = None) -> dict:
    """构造子进程环境：从 os.environ 剔除含敏感名的变量，再合并 extra。

    这是 API key 防泄露的硬约束：LLM key 只应在进程内被 ChatOpenAI 使用，
    绝不应出现在训练脚本、Shell 命令、远程 helper 或 Slurm 脚本的环境里。
    PATH/HOME 等必需变量不含敏感模式,天然保留。
    """
    safe = {
        k: v for k, v in os.environ.items()
        if not any(p in k.upper() for p in SENSITIVE_ENV_PATTERNS)
    }
    safe.update(extra or {})
    return safe


# Directories and files that repo-reading tools (list_tree / grep_files) skip,
# so the agent sees source code instead of VCS metadata and build caches.
WALK_SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".ipynb_checkpoints",
}
# grep_files skips files larger than this (likely data/binaries, not source).
GREP_MAX_FILE_BYTES = 2_000_000


# --- Slurm liveness taxonomy (used by SlurmExecutionBackend) ---
# We map a job's `sacct` State to three buckets. Reference: `man sacct`
# JOB STATE CODES. PENDING/RUNNING/etc. occupy a slot ("running"); COMPLETED
# is "completed"; the rest are "failed". PREEMPTED is intentionally ABSENT:
# under a requeue policy a preempted job returns to PENDING, so we let it fall
# through to "unknown" (bounded grace) rather than reaping it early.
_SLURM_RUNNING_STATES = {
    "PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED",
    "CONFIGURING", "COMPLETING",
}
_SLURM_OK_STATES = {"COMPLETED"}
_SLURM_FAIL_STATES = {
    "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY",
    "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT",
}


def _parse_slurm_time_seconds(spec: str) -> int:
    """Parse a Slurm ``--time`` spec to seconds.

    Accepts the documented forms: ``minutes``, ``minutes:seconds``,
    ``hours:minutes:seconds``, ``days-hours``, ``days-hours:minutes``,
    ``days-hours:minutes:seconds``. Returns a large sentinel when unparseable
    so the wall-clock liveness cap never fires spuriously (the consecutive
    -unknown grace still bounds the loop).
    """
    s = str(spec or "").strip()
    if not s:
        return 10 ** 9
    try:
        days = 0
        if "-" in s:
            d, s = s.split("-", 1)
            days = int(d)
        parts = s.split(":") if s else []
        if days:
            # days-hours[:minutes[:seconds]]
            hours = int(parts[0]) if len(parts) >= 1 else 0
            minutes = int(parts[1]) if len(parts) >= 2 else 0
            seconds = int(parts[2]) if len(parts) >= 3 else 0
        elif len(parts) == 1:
            hours, minutes, seconds = 0, int(parts[0]), 0          # bare minutes
        elif len(parts) == 2:
            hours, minutes, seconds = 0, int(parts[0]), int(parts[1])  # minutes:seconds
        else:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
        return days * 86400 + hours * 3600 + minutes * 60 + seconds
    except (ValueError, TypeError, IndexError):
        return 10 ** 9


REMOTE_HELPER = textwrap.dedent(
    """
    import json
    import os
    import pathlib
    import shlex
    import subprocess
    import sys


    def normalize_rel(raw):
        if raw is None or not str(raw).strip():
            raise ValueError("Path cannot be empty")
        rel = pathlib.PurePosixPath(str(raw))
        if rel.is_absolute():
            raise ValueError("Path must be relative to workspace")
        if any(part == ".." for part in rel.parts):
            raise ValueError(f"Path escapes workspace: {raw}")
        parts = [part for part in rel.parts if part not in ("", ".")]
        return pathlib.Path(*parts)


    def resolve_path(root, raw):
        rel = normalize_rel(raw)
        resolved = (root / rel).resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Path escapes workspace: {raw}") from exc
        return resolved


    WALK_SKIP_DIRS = {
        ".git", "__pycache__", "node_modules", ".venv", "venv",
        ".mypy_cache", ".pytest_cache", ".idea", ".ipynb_checkpoints",
    }
    GREP_MAX_FILE_BYTES = 2000000


    def walk_tree(root, max_depth, max_entries):
        max_depth = max(1, int(max_depth))
        max_entries = max(1, int(max_entries))
        entries = []

        def walk(current, depth):
            if depth > max_depth or len(entries) >= max_entries:
                return
            try:
                children = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name))
            except OSError:
                return
            for child in children:
                if len(entries) >= max_entries:
                    return
                if child.name in WALK_SKIP_DIRS:
                    continue
                if child.is_symlink():
                    continue
                rel = child.relative_to(root).as_posix()
                if child.is_dir():
                    entries.append(rel + "/")
                    walk(child, depth + 1)
                else:
                    entries.append(rel)

        walk(root, 1)
        return entries


    def grep_tree(root, base, pattern, max_results, ignore_case):
        import re
        if not pattern:
            raise ValueError("Search pattern cannot be empty")
        max_results = max(1, int(max_results))
        flags = re.IGNORECASE if ignore_case else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            raise ValueError("Invalid search pattern: " + str(exc))
        targets = []
        if root.is_file():
            targets = [root]
        else:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = sorted(d for d in dirnames if d not in WALK_SKIP_DIRS)
                for name in sorted(filenames):
                    targets.append(pathlib.Path(dirpath) / name)
        hits = []
        for file_path in targets:
            if len(hits) >= max_results:
                break
            try:
                if file_path.is_symlink():
                    continue
                if file_path.stat().st_size > GREP_MAX_FILE_BYTES:
                    continue
                with open(file_path, "r", errors="strict") as handle:
                    for lineno, line in enumerate(handle, start=1):
                        if regex.search(line):
                            hits.append({
                                "file": file_path.relative_to(base).as_posix(),
                                "line": lineno,
                                "text": line.rstrip("\\n")[:300],
                            })
                            if len(hits) >= max_results:
                                break
            except (UnicodeDecodeError, OSError, ValueError):
                continue
        return hits


    def gpu_status():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        gpus.append(
                            {
                                "utilization": f"{parts[0]}%",
                                "memory": f"{parts[1]}MB/{parts[2]}MB",
                            }
                        )
                return {"gpus": gpus, "utilization": gpus[0]["utilization"] if gpus else "N/A"}
        except Exception:
            pass
        return {"utilization": "N/A"}


    def main():
        payload = json.load(sys.stdin)
        root = pathlib.Path(payload["remote_workspace"]).expanduser().resolve(strict=False)
        action = payload["action"]
        result = None

        if action == "validate":
            root.mkdir(parents=True, exist_ok=True)
            result = {"status": "ok"}
        elif action == "read_file":
            path = resolve_path(root, payload["path"])
            if not path.exists():
                raise FileNotFoundError(f"File not found: {payload['path']}")
            result = {"content": path.read_text()}
        elif action == "read_file_range":
            path = resolve_path(root, payload["path"])
            if not path.exists():
                raise FileNotFoundError(f"File not found: {payload['path']}")
            lines = path.read_text().splitlines()
            start = max(1, int(payload.get("start_line", 1)))
            end_raw = payload.get("end_line")
            end = len(lines) if end_raw is None else min(len(lines), int(end_raw))
            if end < start:
                result = {"content": ""}
            else:
                selected = lines[start - 1:end]
                result = {"content": "\\n".join(str(start + i) + "\\t" + t for i, t in enumerate(selected))}
        elif action == "list_tree":
            raw = payload.get("path", ".")
            base = root if raw in ("", ".") else resolve_path(root, raw)
            if not base.is_dir():
                raise NotADirectoryError("Not a directory: " + str(raw))
            result = {"entries": walk_tree(base, payload.get("max_depth", 3), payload.get("max_entries", 300))}
        elif action == "grep_files":
            raw = payload.get("path", ".")
            base = root if raw in ("", ".") else resolve_path(root, raw)
            result = {"hits": grep_tree(base, root, payload["pattern"], payload.get("max_results", 50), payload.get("ignore_case", False))}
        elif action == "write_file":
            path = resolve_path(root, payload["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            content = payload["content"]
            path.write_text(content)
            result = {"status": "written", "path": payload["path"], "bytes": len(content)}
        elif action == "list_files":
            raw = payload.get("path", ".")
            if raw in ("", "."):
                path = root
            else:
                path = resolve_path(root, raw)
            if not path.is_dir():
                raise NotADirectoryError(f"Not a directory: {raw}")
            result = {"files": sorted(p.name for p in path.iterdir())[:100]}
        elif action == "run_command":
            try:
                proc = subprocess.run(
                    payload["argv"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=int(payload.get("timeout_seconds", 120)),
                    cwd=str(root),
                    # remote helper 独立运行，无 _scrub_env；controller 已清洗 env
                    env=(payload.get("env") or {}),
                    check=False,
                )
                result = {
                    "stdout": proc.stdout[-2000:],
                    "stderr": proc.stderr[-500:],
                    "returncode": proc.returncode,
                }
            except subprocess.TimeoutExpired:
                result = {"error": f"Command timed out after {int(payload.get('timeout_seconds', 120))}s"}
        elif action == "launch_command":
            log_file = payload["log_file"]
            log_path = resolve_path(root, log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as handle:
                proc = subprocess.Popen(
                    payload["argv"],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    # controller 的 _invoke 已用 _scrub_env 清洗 env 再发出，
                    # remote helper 无需再 scrub（字符串内也没有该函数）
                    env=(payload.get("env") or {}),
                    start_new_session=True,
                    cwd=str(root),
                )
            result = {"pid": proc.pid, "log_file": log_file, "status": "launched"}
        elif action == "is_process_alive":
            try:
                os.kill(int(payload["pid"]), 0)
                result = {"alive": True}
            except OSError:
                result = {"alive": False}
        elif action == "tail_file":
            path = resolve_path(root, payload["path"])
            if not path.exists():
                result = {"lines": []}
            else:
                lines = path.read_text().splitlines()
                result = {"lines": lines[-int(payload.get('lines', 50)) :]}
        elif action == "get_gpu_status":
            result = gpu_status()
        elif action == "submit_slurm":
            # Build the sbatch script HERE, in Python, with shell=False — no
            # remote shell string is ever assembled from caller-supplied argv,
            # so there is no injection surface. Then `sbatch --parsable` and
            # EXIT: nothing persistent is left on the login node (the v7
            # submit-and-exit invariant). Slurm enforces --time.
            argv = payload["argv"]
            if not isinstance(argv, list) or not argv:
                raise ValueError("submit_slurm requires a non-empty argv list")
            log_file = payload["log_file"]
            log_path = resolve_path(root, log_file)        # reuses traversal guard
            log_path.parent.mkdir(parents=True, exist_ok=True)
            root.mkdir(parents=True, exist_ok=True)
            # Slurm assigns GPUs via --gres; an inherited CUDA_VISIBLE_DEVICES /
            # GPU would pin every job to the wrong physical device. Strip them.
            env = {
                k: v for k, v in (payload.get("env") or {}).items()
                if k not in ("CUDA_VISIBLE_DEVICES", "GPU")
            }
            job_name = str(payload.get("job_name") or "ar_job")
            # #SBATCH directive lines are tokenized by Slurm on whitespace
            # (honoring double quotes), NOT run through a shell — so a path with
            # spaces must be double-quoted. Strip any embedded double-quote to
            # keep quoting unambiguous (paths realistically never contain one).
            def _q(value):
                return chr(34) + str(value).replace(chr(34), "") + chr(34)
            lines = ["#!/bin/bash"]
            lines.append("#SBATCH --job-name=" + _q(job_name))
            lines.append("#SBATCH --partition=" + str(payload["partition"]))
            lines.append("#SBATCH --chdir=" + _q(str(root)))
            # --output is relative; Slurm resolves it against --chdir, matching
            # how tail_file(log_file) resolves it under the workspace root.
            lines.append("#SBATCH --output=" + _q(log_file))
            lines.append("#SBATCH --time=" + str(payload["time"]))
            raw_gres = payload.get("raw_gres") or ""
            gres = payload.get("gres")
            if raw_gres:
                lines.append("#SBATCH --gres=" + str(raw_gres))
            elif isinstance(gres, int) and gres >= 1:
                lines.append("#SBATCH --gres=gpu:" + str(gres))
            if payload.get("qos"):
                lines.append("#SBATCH --qos=" + str(payload["qos"]))
            if payload.get("account"):
                lines.append("#SBATCH --account=" + str(payload["account"]))
            for extra in (payload.get("extra_sbatch") or []):
                lines.append("#SBATCH " + str(extra))
            setup = payload.get("setup") or ""
            if setup:
                lines.append(str(setup))
            for k, v in env.items():
                lines.append("export " + str(k) + "=" + shlex.quote(str(v)))
            lines.append(" ".join(shlex.quote(str(a)) for a in argv))
            script = chr(10).join(lines) + chr(10)
            script_path = root / (".sbatch_" + job_name)
            script_path.write_text(script)
            try:
                proc = subprocess.run(
                    ["sbatch", "--parsable", str(script_path)],
                    capture_output=True, text=True, encoding="utf-8", timeout=60,
                    cwd=str(root), check=False,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("sbatch not found on remote host: " + str(exc))
            if proc.returncode != 0:
                raise RuntimeError(
                    "sbatch failed: " + (proc.stderr or proc.stdout).strip()[:400]
                )
            token = ""
            if proc.stdout.strip():
                token = proc.stdout.strip().splitlines()[0].split(";")[0].strip()
            if not token.isdigit():
                raise RuntimeError(
                    "sbatch did not return a job id: " + proc.stdout.strip()[:200]
                )
            result = {
                "slurm_job_id": int(token),
                "log_file": log_file,
                "script_path": str(script_path),
            }
        else:
            raise ValueError(f"Unknown action: {action}")

        json.dump({"ok": True, "result": result}, sys.stdout)


    if __name__ == "__main__":
        try:
            main()
        except Exception as exc:
            json.dump(
                {
                    "ok": False,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
                sys.stdout,
            )
    """
).strip()

REMOTE_HELPER_B64 = base64.b64encode(REMOTE_HELPER.encode("utf-8")).decode("ascii")
REMOTE_LAUNCHER = "import base64,sys;exec(base64.b64decode(sys.argv[1]).decode())"


def normalize_relative_path(path: str) -> str:
    """Normalize a workspace-relative path and reject traversal."""
    if path is None or not str(path).strip():
        raise ValueError("Path cannot be empty")

    pure = PurePosixPath(str(path))
    if pure.is_absolute():
        raise ValueError("Path must be relative to workspace")
    if any(part == ".." for part in pure.parts):
        raise ValueError(f"Path escapes workspace: {path}")

    normalized = str(pure)
    return "." if normalized in ("", ".") else normalized


def _resolve_under_root(root: Path, rel_path: str) -> Path:
    parts = [part for part in PurePosixPath(rel_path).parts if part not in ("", ".")]
    resolved = (root / Path(*parts)).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace: {rel_path}") from exc
    return resolved


def _walk_tree(root: Path, base: Path, max_depth: int, max_entries: int) -> list[str]:
    """Depth-limited recursive listing relative to `base`, skipping noise dirs."""
    max_depth = max(1, int(max_depth))
    max_entries = max(1, int(max_entries))
    entries: list[str] = []

    def walk(current: Path, depth: int):
        if depth > max_depth or len(entries) >= max_entries:
            return
        try:
            children = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name))
        except (PermissionError, OSError):
            return
        for child in children:
            if len(entries) >= max_entries:
                return
            if child.name in WALK_SKIP_DIRS:
                continue
            # Never follow or list symlinks: they can point outside the
            # workspace, which would defeat the sandbox enforced elsewhere.
            if child.is_symlink():
                continue
            rel = child.relative_to(base).as_posix()
            if child.is_dir():
                entries.append(rel + "/")
                walk(child, depth + 1)
            else:
                entries.append(rel)

    walk(root, 1)
    return entries


def _grep_tree(root: Path, base: Path, pattern: str, max_results: int, ignore_case: bool) -> list[dict]:
    """Scan text files under `root` for `pattern`, returning file/line/text hits."""
    import re as _re

    if not pattern:
        raise ValueError("Search pattern cannot be empty")
    max_results = max(1, int(max_results))
    flags = _re.IGNORECASE if ignore_case else 0
    try:
        regex = _re.compile(pattern, flags)
    except _re.error as exc:
        raise ValueError(f"Invalid search pattern: {exc}") from exc

    targets: list[Path] = []
    if root.is_file():
        targets = [root]
    else:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in WALK_SKIP_DIRS)
            for name in sorted(filenames):
                targets.append(Path(dirpath) / name)

    hits: list[dict] = []
    for file_path in targets:
        if len(hits) >= max_results:
            break
        try:
            # os.walk does not descend symlinked dirs, but symlinked *files*
            # still appear and would otherwise be opened — that could read a
            # file outside the workspace. Skip any symlink target.
            if file_path.is_symlink():
                continue
            if file_path.stat().st_size > GREP_MAX_FILE_BYTES:
                continue
            with open(file_path, "r", errors="strict") as handle:
                for lineno, line in enumerate(handle, start=1):
                    if regex.search(line):
                        hits.append(
                            {
                                "file": file_path.relative_to(base).as_posix(),
                                "line": lineno,
                                "text": line.rstrip("\n")[:300],
                            }
                        )
                        if len(hits) >= max_results:
                            break
        except (UnicodeDecodeError, PermissionError, OSError, ValueError):
            # Binary file, unreadable, or escaped base — skip silently.
            continue
    return hits


class ExecutionBackend:
    """Abstract execution backend."""

    def validate(self):
        raise NotImplementedError

    def read_file(self, path: str) -> str:
        raise NotImplementedError

    def read_file_range(self, path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
        raise NotImplementedError

    def write_file(self, path: str, content: str) -> dict:
        raise NotImplementedError

    def list_files(self, path: str = ".") -> list[str]:
        raise NotImplementedError

    def list_tree(self, path: str = ".", max_depth: int = 3, max_entries: int = 300) -> list[str]:
        raise NotImplementedError

    def grep_files(
        self,
        pattern: str,
        path: str = ".",
        max_results: int = 50,
        ignore_case: bool = False,
    ) -> list[dict]:
        raise NotImplementedError

    def run_command(self, argv: list[str], timeout: int = 120, env: Optional[dict] = None) -> dict:
        raise NotImplementedError

    def launch_command(self, argv: list[str], log_file: str, env: Optional[dict] = None) -> dict:
        raise NotImplementedError

    def is_process_alive(self, pid: int) -> bool:
        raise NotImplementedError

    def tail_file(self, path: str, lines: int = 50) -> list[str]:
        raise NotImplementedError

    def get_gpu_status(self) -> dict:
        raise NotImplementedError

    def final_status(self, pid: int) -> dict:
        """Outcome of a finished job: ``{"state": <str>, "success": <bool|None>}``.

        Default is indeterminate (``success=None``): backends that only track an
        OS pid cannot recover an exit code after the process is gone, so the
        caller keeps treating the run as "completed". The Slurm backend overrides
        this with the real ``sacct`` terminal state so FAILED / TIMEOUT / CANCELLED
        are not silently reported as success.
        """
        return {"state": "unknown", "success": None}


# ═══ 训练解释器绑定（环境一致性）：干跑和训练必须是同一个 python ═══
#
# 事故背景（2026-08-13）：worker 在命令里写 `python` 字面量，run_shell（shell
# 管道）和 launch_experiment（Popen 直启）两条路径解析出了不同解释器——干跑
# 用有 torchvision 的过、真训练用没有的崩。结论：解释器是谁属于「事实」，
# 由系统层绑定，不能由 LLM 在命令字符串里临场决定。

_BARE_PYTHON_NAMES = ("python", "python.exe", "python3", "python3.exe", "py", "py.exe")


def bind_python_argv(argv: list, python_exe: str) -> list:
    """把命令首元素里的裸 python 字面量替换为绑定解释器的绝对路径。

    只替换「裸名字」（无路径分隔符）：LLM 没指定是哪个 python 时，系统层
    填默认绑定值。若 LLM 显式写了绝对路径则尊重原值——其一致性由
    dryrun_interpreter_error 的硬校验拦截。
    python_exe 为空 → 原样返回（未解析到环境时保持旧行为，不硬拦）。
    """
    if not argv or not python_exe:
        return list(argv)
    first = argv[0]
    if Path(first).name.lower() in _BARE_PYTHON_NAMES and Path(first).name == first:
        return [python_exe] + list(argv[1:])
    return list(argv)


def check_python_deps(python_exe: str, timeout: int = 30) -> bool:
    """该解释器能否 import torch + torchvision（训练环境最低门槛）。

    用 importlib.util.find_spec 读模块元数据判断（**不 import torch** ——
    import 首载 10s+，而 _python_candidates 会对每个候选解释器调用本函数，
    N 个 env 就是 N×10s+；且全部失败后还会触发 2GB 的 torch 下载）。
    import 真伪由实际 launch 时的 dry-run 验证。
    """
    code = (
        "import importlib.util as u, sys; "
        "ok = u.find_spec('torch') is not None "
        "and u.find_spec('torchvision') is not None; "
        "sys.exit(0 if ok else 1)"
    )
    try:
        result = subprocess.run(
            [python_exe, "-c", code],
            capture_output=True, timeout=timeout,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _python_candidates() -> list:
    """候选解释器：本机 conda envs（字母序）→ base → 当前解释器。

    conda root 通过「向上找含 envs/ 子目录的祖先」定位，兼容
    <root>/envs/<env>/python.exe（如 D:\\Anaconda\\envs\\langchain）与
    base 解释器（D:\\Anaconda\\python.exe）两种布局。
    """
    candidates = []
    exe = Path(sys.executable)
    envs_dir = None
    node = exe.parent
    while node != node.parent:
        if (node / "envs").is_dir():
            envs_dir = node / "envs"  # node 即 conda root
            break
        node = node.parent
    if envs_dir is not None:
        for env_dir in sorted(envs_dir.iterdir()):
            candidate = env_dir / "python.exe"
            if candidate.exists():
                candidates.append(str(candidate))
        base_py = envs_dir.parent / "python.exe"
        if base_py.exists():
            candidates.append(str(base_py))
    if exe.exists() and str(exe) not in candidates:
        candidates.append(str(exe))
    return candidates


def _cuda_capable(python_exe: str, timeout: int = 30) -> bool:
    """该解释器的 torch 是否可用 CUDA（训练优先选 GPU 环境）。"""
    try:
        result = subprocess.run(
            [python_exe, "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True, timeout=timeout,
        )
        return result.returncode == 0 and result.stdout.strip() == b"True"
    except (OSError, subprocess.TimeoutExpired):
        return False


def resolve_project_python(config: dict, workspace: Path) -> str:
    """解析项目训练解释器（环境绑定的唯一事实源），返回绝对路径；失败返回 ""。

    优先级：config.execution.python（用户指定）→ 本机探测。探测两轮：
    先选「torch+torchvision 齐全且 CUDA 可用」的 env（训练要 GPU），
    再退而求其次选依赖齐全的。解析结果同时用于 launch 的命令绑定和
    干跑记录校验。
    """
    execution = (config or {}).get("execution", {}) or {}
    specified = str(execution.get("python", "") or "").strip()
    if specified:
        if check_python_deps(specified):
            return specified
        # 指定的路径失效（如项目迁移到别的机器）→ 不硬拦，退化到探测。
        # pin 是偏好不是硬约束：可移植性优先。
        logger.warning(
            "config 指定的训练解释器失效（缺 torch/torchvision）：%s —— 退化到自动探测",
            specified)

    fallback = ""
    for candidate in _python_candidates():
        if not check_python_deps(candidate):
            continue
        if _cuda_capable(candidate):
            return candidate
        if not fallback:
            fallback = candidate
    return fallback


def _pick_env_creator() -> str:
    """选择环境创建工具:uv(最快)→ conda → venv 兜底。"""
    if shutil.which("uv"):
        return "uv"
    if shutil.which("conda"):
        return "conda"
    return "venv"


def _env_install_command(creator: str, workspace: Path) -> list[str]:
    """构造创建命令(纯函数,可测)。torch 安装用默认行为
    (Linux=CUDA 构建/Windows=CPU 版);特殊版本经
    execution.torch_index_url 或 AR_TORCH_INDEX_URL 注入。"""
    index_url = os.environ.get("AR_TORCH_INDEX_URL", "")
    if creator == "uv":
        cmd = ["uv", "venv", str(workspace / ".trainenv")]
        if index_url:
            cmd += ["--index-url", index_url]
        return cmd
    if creator == "conda":
        return ["conda", "create", "-n", f"{workspace.name}-env", "python=3.11", "-y"]
    # venv 兜底
    return [sys.executable, "-m", "venv", str(workspace / ".trainenv")]


def _env_install_packages_command(creator: str, workspace: Path) -> list[str]:
    """构造 torch 安装命令(环境创建完成后执行)。

    conda 用 `conda run -n <env> pip install`;uv/venv 直接对项目解释器安装。
    torch 默认安装行为(Linux=CUDA 构建/Windows=CPU 版);
    特殊版本经 AR_TORCH_INDEX_URL 注入。
    """
    index_url = os.environ.get("AR_TORCH_INDEX_URL", "")
    packages = ["torch", "torchvision"]
    if creator == "uv":
        cmd = ["uv", "pip", "install", "--python",
               _project_venv_python(workspace), *packages]
        if index_url:
            cmd += ["--index-url", index_url]
        return cmd
    if creator == "conda":
        cmd = ["conda", "run", "-n", f"{workspace.name}-env",
               "pip", "install", *packages]
        if index_url:
            cmd += ["--index-url", index_url]
        return cmd
    return [_project_venv_python(workspace), "-m", "pip",
            "install", *packages] + \
        (["--index-url", index_url] if index_url else [])


def _env_status(workspace: Path) -> dict:
    """读环境创建状态文件(缺失 → {})。"""
    status_path = workspace / ".python_env.status"
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_env_status(workspace: Path, **fields) -> None:
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        status = dict(_env_status(workspace))
        status.update(fields)
        (workspace / ".python_env.status").write_text(
            json.dumps(status, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _start_training_env_async(workspace: Path) -> bool:
    """后台启动环境创建(两阶段:create env → install torch)。

    输出落盘 .trainenv_install.log,状态写 .python_env.status
    (creating→installing→ready/failed,由 _settle_env_status 惰性推进)。
    agent 侧不阻塞 —— launch 检测到 creating 时返回"稍后重试",
    期间零 LLM 成本(与 monitor 同一哲学)。
    """
    creator = _pick_env_creator()
    log_path = workspace / ".trainenv_install.log"
    try:
        with open(log_path, "a", encoding="utf-8") as log:
            proc = subprocess.Popen(
                _env_install_command(creator, workspace),
                stdout=log, stderr=log, cwd=str(workspace),
            )
    except OSError:
        _write_env_status(workspace, status="failed",
                          creator=creator, error="create command failed to start")
        return False
    _write_env_status(workspace, status="creating", phase="create",
                      creator=creator, started_at=time.time(), pid=proc.pid)
    logger.info("环境创建后台启动(creator=%s, pid=%d) -> %s",
                creator, proc.pid, log_path)
    return True


def _spawn_install(workspace: Path, status: dict) -> None:
    """推进到 installing 阶段:启动 torch 安装子进程。"""
    creator = status.get("creator", "venv")
    log_path = workspace / ".trainenv_install.log"
    try:
        with open(log_path, "a", encoding="utf-8") as log:
            proc = subprocess.Popen(
                _env_install_packages_command(creator, workspace),
                stdout=log, stderr=log, cwd=str(workspace),
            )
    except OSError:
        _write_env_status(workspace, status="failed", phase="install",
                          error="install command failed to start")
        return
    _write_env_status(workspace, status="installing", phase="install",
                      pid=proc.pid, install_started_at=time.time())


def _settle_env_status(workspace: Path) -> dict:
    """惰性推进创建状态机(调用方轮询时结算):

    creating(create 进程结束)→ 启动 installing → installing(install 进程
    结束)→ deps 检查 → ready(+钉住)/ failed。
    """
    status = _env_status(workspace)
    phase = status.get("status")
    if phase not in ("creating", "installing"):
        return status
    pid = status.get("pid")
    try:
        if pid and pid_alive(int(pid)):
            return status  # 仍在跑
    except Exception:
        pass
    # 进程已结束 → 推进
    if phase == "creating":
        _spawn_install(workspace, status)
        return _env_status(workspace)
    # installing 结束 → 结算
    creator = status.get("creator", "venv")
    python_exe = _project_venv_python(workspace)
    ok = False
    try:
        if creator == "conda":
            candidate = f"{workspace.name}-env"
            env_py = ""
            listing = subprocess.run(["conda", "env", "list", "--json"],
                                     capture_output=True, text=True, timeout=30)
            if listing.returncode == 0:
                for entry in json.loads(listing.stdout).get("envs", []):
                    if Path(entry).name == candidate:
                        env_py = str(Path(entry) / ("python.exe" if os.name == "nt" else "bin/python"))
                        break
            if env_py and check_python_deps(env_py):
                python_exe, ok = env_py, True
        else:
            ok = check_python_deps(python_exe)
    except Exception:
        ok = False
    if ok:
        _write_env_status(
            workspace, status="ready",
            elapsed_sec=round(time.time() - float(status.get("started_at", time.time())), 1))
        _pin(workspace / ".python_env.json", python_exe)
        logger.info("环境创建完成: %s", python_exe)
        return _env_status(workspace)
    _write_env_status(workspace, status="failed",
                      error="deps check failed after install (see .trainenv_install.log)")
    return _env_status(workspace)


def _create_training_env(workspace: Path, install_timeout: int = 1800) -> str:
    """同步创建项目环境(测试/兼容用):后台启动 + 轮询等待到 ready/failed。"""
    if _start_training_env_async(workspace):
        deadline = time.time() + install_timeout
        while time.time() < deadline:
            status = _settle_env_status(workspace)
            if status.get("status") == "ready":
                return _project_venv_python(workspace)
            if status.get("status") == "failed":
                return ""
            time.sleep(2)
    return ""


def _project_venv_python(workspace: Path) -> str:
    """项目专属 venv(.trainenv)的解释器路径。"""
    venv_dir = workspace / ".trainenv"
    if os.name == "nt":
        return str(venv_dir / "Scripts" / "python.exe")
    return str(venv_dir / "bin" / "python")


def _load_pin(pin_path: Path) -> dict:
    try:
        data = json.loads(pin_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _pin(pin_path: Path, interpreter: str) -> None:
    """原子写钉住记录(解释器事实源)。失败只警告,不阻塞解析。"""
    try:
        pin_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = pin_path.with_name(pin_path.name + ".tmp")
        tmp.write_text(json.dumps({
            "interpreter": str(Path(interpreter).resolve()),
            "pinned_at": time.time(),
        }, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, pin_path)
    except OSError as exc:
        logger.warning("训练解释器钉住记录写入失败: %s", exc)


def ensure_project_python(config: dict, workspace: Path,
                          auto_create: Optional[bool] = None) -> str:
    """解析项目训练解释器并钉住到 workspace/.python_env.json。

    策略（项目隔离,防污染 base/系统环境）:
      1. config.execution.python 显式指定 → 用之（用户自己的选择）
      2. .python_env.json 已钉住且仍可用 → 直接复用
         —— 跨 cycle、跨进程永远同一解释器（干跑/训练一致性的根基）
      3. 项目 venv（workspace/.trainenv）存在且可用 → 用之
      4. auto_create（默认开,可被调用方关闭）→ 创建项目 venv（隔离,不碰 base）
      5. 借用现成环境（兜底;显式警告污染风险）
    任何成功路径都会把结果钉住,后续轮次/进程只读钉住记录,不再重新
    探测 —— 除非钉住记录失效（环境迁移/被删除）。

    auto_create=False 时跳过第 4 步：解析/借用但不创建 —— 用于 agent
    构造阶段（不该因解析环境就触发 2GB 的 torch 下载）；真正 launch 时
    的惰性重解析恢复创建能力。
    """
    pin_path = workspace / ".python_env.json"
    execution = (config or {}).get("execution", {}) or {}
    allow_create = (execution.get("auto_create_env", True)
                    if auto_create is None else auto_create)

    # 1. config 显式指定（用户自己的选择,优先级最高）
    specified = str(execution.get("python", "") or "").strip()
    if specified:
        if check_python_deps(specified):
            _pin(pin_path, specified)
            return specified
        logger.warning(
            "config 指定的训练解释器失效（缺 torch/torchvision）：%s —— 退化",
            specified)

    # 2. 已钉住且仍可用 → 复用（一致性事实源,不再探测）
    pin = _load_pin(pin_path)
    if pin.get("interpreter") and check_python_deps(pin["interpreter"]):
        return pin["interpreter"]

    # 3. 项目 venv 优先（隔离,绝不默认碰 base）
    venv_py = _project_venv_python(workspace)
    if check_python_deps(venv_py):
        _pin(pin_path, venv_py)
        return venv_py

    # 4. 自动创建项目环境(异步:uv/conda/venv + torch;可能耗时几分钟。
    #    创建期间 launch 返回"稍后重试",agent 不阻塞不烧 token;
    #    已在创建/安装中 → 不重复启动)
    if allow_create:
        if _env_status(workspace).get("status") in ("creating", "installing"):
            return ""
        if _start_training_env_async(workspace):
            return ""  # 创建中 —— 调用方(launch)检查 .python_env.status 给提示

    # 5. 借用现成环境（兜底;借用有污染 base 的风险,显式警告）
    borrowed = resolve_project_python(config, workspace)
    if borrowed:
        logger.warning(
            "借用现成环境 %s 作为训练解释器 —— agent 安装依赖可能污染该环境;"
            "建议保留 execution.auto_create_env 让系统创建项目专属 venv",
            borrowed)
        _pin(pin_path, borrowed)
    return borrowed


def dryrun_interpreter_error(dry_run_data: dict, python_exe: str) -> str:
    """干跑记录的解释器与训练解释器不一致 → 返回错误信息；一致/未记录 → ""。"""
    recorded = str((dry_run_data or {}).get("interpreter", "") or "")
    if not recorded:
        return ""  # 旧版干跑记录没有该字段 → 不拦截
    if Path(recorded).resolve() != Path(python_exe).resolve():
        return (
            f"The interpreter used for the dry run ({recorded}) does not match "
            f"the interpreter resolved for the training command ({python_exe}). "
            f"Launch training with the SAME interpreter used for the dry run."
        )
    return ""


_FINGERPRINT_CACHE: dict[str, dict] = {}


def python_fingerprint(python_exe: str) -> dict:
    """查询解释器的依赖指纹（python/torch/torchvision 版本），进程内缓存。

    解释器路径相同 ≠ 依赖版本相同（干跑后 pip 升级等情况）——
    指纹是干跑/训练一致性校验的第二道闸。
    用 importlib.metadata 读 dist-info 元数据（不 import 库 —— import torch
    首载要 10s+，会拖慢每个新项目的第一次 launch）。
    查询失败返回 {}（不阻塞,由路径校验兜底）。
    """
    key = str(python_exe)
    if key in _FINGERPRINT_CACHE:
        return _FINGERPRINT_CACHE[key]
    code = (
        "import sys, json; "
        "from importlib.metadata import version as _v; "
        "out = {'python': sys.version.split()[0]}; "
        "import importlib.util as u; "
        "out['torch'] = _v('torch') if u.find_spec('torch') else ''; "
        "out['torchvision'] = _v('torchvision') if u.find_spec('torchvision') else ''; "
        "print(json.dumps(out))"
    )
    fp: dict = {}
    try:
        proc = subprocess.run(
            [str(python_exe), "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            fp = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        pass
    _FINGERPRINT_CACHE[key] = fp
    return fp


def script_hash(workspace: Path, script_rel: str) -> str:
    """训练脚本内容指纹（md5），用于检测「干跑后脚本被改动」。"""
    p = workspace / script_rel
    try:
        return hashlib.md5(p.read_bytes()).hexdigest()
    except OSError:
        return ""


def pid_alive(pid: int) -> bool:
    """跨平台进程存活探测（零成本，无 LLM）。

    Windows 走 _win32_pid_alive（GetExitCodeProcess）；POSIX 走
    os.kill(pid, 0)。所有「这个 pid 还活着吗」的判定统一走这里，
    不要直接调 os.kill——Windows 上它有两种坏行为：
      1. 已退出但进程对象未回收的 pid → 误报存活（monitor 死循环）
      2. 对象已完全回收的 pid → 抛 SystemError（CPython bug，except
         OSError 兜不住，直接崩溃）
    """
    if os.name == "nt":
        return _win32_pid_alive(pid)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _win32_pid_alive(pid: int) -> bool:
    """Windows 专用进程存活判定（stdlib ctypes，零成本）。

    os.kill(pid, 0) 在 Windows 上对「已退出但进程对象尚未被内核回收」的
    pid 会误报存活（实测：子进程退出后、甚至 Popen 句柄被 GC 后，os.kill
    仍返回成功），导致 monitor 的零 LLM 轮询在死进程上无限空转。
    改用 GetExitCodeProcess：只有退出码为 STILL_ACTIVE(259) 才是真存活；
    进程对象已回收（OpenProcess 失败）或退出码为其他值 → 判定已退出。
    """
    import ctypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False  # 进程对象已回收（或 pid 不存在）
    try:
        code = ctypes.c_ulong(0)
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            # 实际不可达（OpenProcess 成功则此调用必成功）；保守判死，
            # 避免重蹈轮询卡死的覆辙。
            return False
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


class LocalExecutionBackend(ExecutionBackend):
    """Current on-machine behavior."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()

    def validate(self):
        self.workspace.mkdir(parents=True, exist_ok=True)

    def read_file(self, path: str) -> str:
        file_path = _resolve_under_root(self.workspace, normalize_relative_path(path))
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return file_path.read_text()

    def read_file_range(self, path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
        file_path = _resolve_under_root(self.workspace, normalize_relative_path(path))
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        lines = file_path.read_text().splitlines()
        start = max(1, int(start_line))
        end = len(lines) if end_line is None else min(len(lines), int(end_line))
        if end < start:
            return ""
        selected = lines[start - 1 : end]
        return "\n".join(f"{start + i}\t{text}" for i, text in enumerate(selected))

    def write_file(self, path: str, content: str) -> dict:
        rel_path = normalize_relative_path(path)
        file_path = _resolve_under_root(self.workspace, rel_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return {"status": "written", "path": rel_path, "bytes": len(content)}

    def list_files(self, path: str = ".") -> list[str]:
        rel_path = normalize_relative_path(path)
        dir_path = self.workspace if rel_path == "." else _resolve_under_root(self.workspace, rel_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        return sorted([f.name for f in dir_path.iterdir()])[:100]

    def list_tree(self, path: str = ".", max_depth: int = 3, max_entries: int = 300) -> list[str]:
        rel_path = normalize_relative_path(path)
        root = self.workspace if rel_path == "." else _resolve_under_root(self.workspace, rel_path)
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        return _walk_tree(root, root, max_depth=max_depth, max_entries=max_entries)

    def grep_files(
        self,
        pattern: str,
        path: str = ".",
        max_results: int = 50,
        ignore_case: bool = False,
    ) -> list[dict]:
        rel_path = normalize_relative_path(path)
        root = self.workspace if rel_path == "." else _resolve_under_root(self.workspace, rel_path)
        return _grep_tree(root, self.workspace, pattern, max_results=max_results, ignore_case=ignore_case)

    def run_command(self, argv: list[str], timeout: int = 120, env: Optional[dict] = None) -> dict:
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=str(self.workspace),
                env=_scrub_env(env or {}),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"Command timed out after {timeout}s"}

        return {
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-500:],
            "returncode": result.returncode,
        }

    def launch_command(self, argv: list[str], log_file: str, env: Optional[dict] = None) -> dict:
        rel_path = normalize_relative_path(log_file)
        log_path = _resolve_under_root(self.workspace, rel_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w") as handle:
            proc = subprocess.Popen(
                argv,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=_scrub_env(env or {}),
                start_new_session=True,
                cwd=str(self.workspace),
            )

        return {"pid": proc.pid, "log_file": rel_path, "status": "launched"}

    def is_process_alive(self, pid: int) -> bool:
        return pid_alive(pid)

    def tail_file(self, path: str, lines: int = 50) -> list[str]:
        rel_path = normalize_relative_path(path)
        file_path = _resolve_under_root(self.workspace, rel_path)
        if not file_path.exists():
            return []
        return file_path.read_text().splitlines()[-lines:]

    def get_gpu_status(self) -> dict:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().splitlines()
                gpus = []
                for line in lines:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        gpus.append(
                            {
                                "utilization": f"{parts[0]}%",
                                "memory": f"{parts[1]}MB/{parts[2]}MB",
                            }
                        )
                return {"gpus": gpus, "utilization": gpus[0]["utilization"] if gpus else "N/A"}
        except Exception:
            pass
        return {"utilization": "N/A"}


class SSHExecutionBackend(ExecutionBackend):
    """Run the tool-visible workspace on a remote host over SSH."""

    def __init__(
        self,
        ssh_host: str,
        remote_workspace: str,
        remote_python: str = "python3",
        ssh_args: Optional[list[str]] = None,
    ):
        self.ssh_host = ssh_host
        self.remote_workspace = remote_workspace
        self.remote_python = remote_python or "python3"
        self.ssh_args = [str(arg) for arg in (ssh_args or [])]

    def validate(self):
        if not self.ssh_host:
            raise ValueError("execution.ssh_host is required when execution.mode=ssh")
        if not self.remote_workspace:
            raise ValueError("execution.remote_workspace is required when execution.mode=ssh")
        if shutil.which("ssh") is None:
            raise RuntimeError("ssh binary not found on PATH")
        self._invoke("validate", transport_timeout=30)

    def read_file(self, path: str) -> str:
        payload = self._invoke("read_file", path=normalize_relative_path(path))
        return payload["content"]

    def read_file_range(self, path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
        payload = self._invoke(
            "read_file_range",
            path=normalize_relative_path(path),
            start_line=int(start_line),
            end_line=None if end_line is None else int(end_line),
        )
        return payload["content"]

    def write_file(self, path: str, content: str) -> dict:
        return self._invoke("write_file", path=normalize_relative_path(path), content=content)

    def list_files(self, path: str = ".") -> list[str]:
        payload = self._invoke("list_files", path=normalize_relative_path(path))
        return payload["files"]

    def list_tree(self, path: str = ".", max_depth: int = 3, max_entries: int = 300) -> list[str]:
        payload = self._invoke(
            "list_tree",
            path=normalize_relative_path(path),
            max_depth=int(max_depth),
            max_entries=int(max_entries),
        )
        return payload["entries"]

    def grep_files(
        self,
        pattern: str,
        path: str = ".",
        max_results: int = 50,
        ignore_case: bool = False,
    ) -> list[dict]:
        payload = self._invoke(
            "grep_files",
            pattern=pattern,
            path=normalize_relative_path(path),
            max_results=int(max_results),
            ignore_case=bool(ignore_case),
            transport_timeout=60,
        )
        return payload["hits"]

    def run_command(self, argv: list[str], timeout: int = 120, env: Optional[dict] = None) -> dict:
        return self._invoke(
            "run_command",
            argv=argv,
            timeout_seconds=timeout,
            env=_scrub_env(env or {}),
            transport_timeout=timeout + 10,
        )

    def launch_command(self, argv: list[str], log_file: str, env: Optional[dict] = None) -> dict:
        return self._invoke(
            "launch_command",
            argv=argv,
            log_file=normalize_relative_path(log_file),
            env=_scrub_env(env or {}),
            transport_timeout=30,
        )

    def is_process_alive(self, pid: int) -> bool:
        payload = self._invoke("is_process_alive", pid=int(pid), transport_timeout=15)
        return bool(payload["alive"])

    def tail_file(self, path: str, lines: int = 50) -> list[str]:
        payload = self._invoke("tail_file", path=normalize_relative_path(path), lines=lines, transport_timeout=15)
        return payload["lines"]

    def get_gpu_status(self) -> dict:
        return self._invoke("get_gpu_status", transport_timeout=20)

    def _ssh_shell(self, remote_cmd: str, timeout: int = 15) -> subprocess.CompletedProcess:
        """Run ONE transient remote shell command, reusing this backend's host
        and ssh_args (single source of truth — no split-brain transport).

        Used by the Slurm subclass for ``sacct`` / ``squeue`` / ``scancel``, the
        only places an arbitrary remote shell string is needed. Each call runs
        one command and returns immediately; nothing persistent is started on
        the remote. The only values interpolated into these strings are
        validated integers (job ids) or operator-controlled config.
        """
        return subprocess.run(
            ["ssh", *self.ssh_args, self.ssh_host, remote_cmd],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            check=False,
        )

    def _invoke(self, action: str, transport_timeout: int = 60, **kwargs) -> dict:
        payload = {
            "action": action,
            "remote_workspace": self.remote_workspace,
            **kwargs,
        }
        remote_command = (
            f"{shlex.quote(self.remote_python)} -c {shlex.quote(REMOTE_LAUNCHER)} "
            f"{shlex.quote(REMOTE_HELPER_B64)}"
        )
        command = ["ssh", *self.ssh_args, self.ssh_host, remote_command]
        try:
            result = subprocess.run(
                command,
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=transport_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"SSH backend action '{action}' timed out after {transport_timeout}s") from exc

        if result.returncode != 0:
            stderr_tail = (result.stderr or "").strip().splitlines()[-5:]
            message = " | ".join(stderr_tail) if stderr_tail else "unknown ssh error"
            raise RuntimeError(f"SSH backend action '{action}' failed: {message}")

        try:
            payload = json.loads((result.stdout or "").strip() or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"SSH backend action '{action}' returned invalid JSON") from exc

        if not payload.get("ok"):
            error = payload.get("error", "unknown remote error")
            error_type = payload.get("error_type", "RuntimeError")
            if error_type == "FileNotFoundError":
                raise FileNotFoundError(error)
            if error_type == "NotADirectoryError":
                raise NotADirectoryError(error)
            if error_type == "ValueError":
                raise ValueError(error)
            raise RuntimeError(error)

        return payload.get("result", {})


class SlurmExecutionBackend(SSHExecutionBackend):
    """Run experiments on a Slurm-managed cluster via a login node.

    The login node shares an NFS workspace with the compute nodes, so every
    file / repo-reading / ``run_command`` operation is inherited unchanged from
    :class:`SSHExecutionBackend` (they run on the login node over the same
    JSON-over-stdin helper transport). Only three things differ on a scheduler:

      - **launch** — instead of starting a process, submit an ``sbatch`` job
        with ``--parsable`` over ONE transient ssh call that exits immediately.
        The integer Slurm job id is returned in the ``pid`` field so the
        existing PID-keyed monitor / state.json plumbing works unchanged. No
        ``srun --wait``, no ``tmux``, no polling loop is ever left on the login
        node (the 2026-05-29 Tokyo-U MIL incident: a persistent login-node
        process is impermissible).
      - **liveness** — ``sacct`` is the sole authority while the cluster is
        reachable; the controller polls it transiently. Slurm enforces
        ``--time`` (reporting ``TIMEOUT``), so a running job always reaches a
        terminal state on its own.
      - **gpu status** — the login node has no usable ``nvidia-smi``; report
        the partition's queue occupancy from ``squeue`` instead.

    Two safeguards live INSIDE :meth:`is_process_alive` so the monitor's
    unbounded ``while is_process_alive(pid): sleep`` loop provably terminates
    even if the cluster becomes unreachable. They apply ONLY when sacct cannot
    confirm the job's state — a job sacct still reports as queued/running is
    never reaped (a long PENDING queue wait is not bounded by ``--time``):

      1. *Bounded unknown grace* — after ``slurm_unknown_grace_polls``
         consecutive indeterminate probes (ssh down / sacct purged), the job is
         declared dead.
      2. *Wall-clock backstop* — if the job is still unconfirmable once
         ``--time`` + ``slurm_time_buffer`` has elapsed since the first poll,
         it is declared dead (Slurm would have produced a terminal state by
         then for any job that actually ran).
    """

    def __init__(
        self,
        ssh_host: str,
        remote_workspace: str,
        remote_python: str = "python3",
        ssh_args: Optional[list[str]] = None,
        slurm_partition: str = "",
        slurm_time: str = "",
        slurm_gpus_per_job: Optional[int] = None,
        slurm_gres: str = "",
        slurm_qos: str = "",
        slurm_account: str = "",
        slurm_setup: str = "",
        slurm_extra_sbatch: Optional[list[str]] = None,
        slurm_unknown_grace_polls: int = 4,
        slurm_time_buffer: int = 1800,
    ):
        super().__init__(ssh_host, remote_workspace, remote_python, ssh_args)
        self.slurm_partition = slurm_partition
        self.slurm_time = slurm_time
        self.slurm_gpus_per_job = slurm_gpus_per_job
        self.slurm_gres = slurm_gres
        self.slurm_qos = slurm_qos
        self.slurm_account = slurm_account
        self.slurm_setup = slurm_setup
        self.slurm_extra_sbatch = list(slurm_extra_sbatch or [])
        self.slurm_unknown_grace_polls = int(slurm_unknown_grace_polls)
        self.slurm_time_buffer = int(slurm_time_buffer)
        self._time_cap_seconds = _parse_slurm_time_seconds(slurm_time)
        # Per-job liveness state, keyed by Slurm job id.
        self._first_seen: dict[int, float] = {}
        self._unknown_count: dict[int, int] = {}
        self._last_terminal: dict[int, str] = {}

    def validate(self):
        if not self.ssh_host:
            raise ValueError("execution.ssh_host is required when execution.mode=slurm")
        if not self.remote_workspace:
            raise ValueError("execution.remote_workspace is required when execution.mode=slurm")
        if not self.slurm_partition:
            raise ValueError("execution.slurm_partition is required when execution.mode=slurm")
        if not self.slurm_time:
            raise ValueError("execution.slurm_time is required when execution.mode=slurm")
        if shutil.which("ssh") is None:
            raise RuntimeError("ssh binary not found on PATH")
        # Workspace reachable + remote python OK (inherited helper transport).
        self._invoke("validate", transport_timeout=30)
        # Require ALL three tools: `command -v a b c` succeeds if ANY one
        # resolves, so chain a check per tool.
        probe = self._ssh_shell(
            "command -v sbatch >/dev/null 2>&1 "
            "&& command -v sacct >/dev/null 2>&1 "
            "&& command -v squeue >/dev/null 2>&1 && echo OK",
            timeout=15,
        )
        if probe.returncode != 0 or "OK" not in (probe.stdout or ""):
            raise RuntimeError(
                "Slurm tools (sbatch/sacct/squeue) not found on the login node; "
                "is execution.ssh_host a Slurm submit host?"
            )

    def launch_command(self, argv: list[str], log_file: str, env: Optional[dict] = None) -> dict:
        normalized_log = normalize_relative_path(log_file)
        job_name = "ar_" + (Path(normalized_log).stem or "job")
        payload = self._invoke(
            "submit_slurm",
            argv=list(argv),
            log_file=normalized_log,
            env=_scrub_env(env or {}),           # remote helper strips CUDA_VISIBLE_DEVICES/GPU
            partition=self.slurm_partition,
            time=self.slurm_time,
            gres=self.slurm_gpus_per_job,
            raw_gres=self.slurm_gres,
            qos=self.slurm_qos,
            account=self.slurm_account,
            job_name=job_name,
            setup=self.slurm_setup,
            extra_sbatch=list(self.slurm_extra_sbatch),
            transport_timeout=90,
        )
        job_id = int(payload["slurm_job_id"])
        # `pid` carries the Slurm job id so the existing monitor / state.json /
        # obsidian plumbing (which keys on `pid`) works without changes.
        return {
            "pid": job_id,
            "slurm_job_id": job_id,
            "log_file": payload.get("log_file", normalized_log),
            "status": "submitted",
        }

    def _sacct_state(self, job_id: int) -> tuple[str, str]:
        """Return (bucket, raw_state) for a Slurm job; bucket in
        {running, completed, failed, unknown}. One transient sacct query, with
        a squeue fallback for a job too new / already purged from accounting."""
        cmd = f"sacct -j {int(job_id)} --format=State%30 -X -n -P 2>/dev/null | head -1"
        try:
            r = self._ssh_shell(cmd, timeout=15)
        except (subprocess.TimeoutExpired, OSError):
            return "unknown", "ssh_failed"
        if r.returncode != 0:
            return "unknown", f"sacct_rc={r.returncode}"
        out = (r.stdout or "").strip()
        # split()[0] drops a trailing " by <uid>" (e.g. "CANCELLED by 1001");
        # .replace("+","") strips the "CANCELLED+" suffix Slurm appends.
        raw = out.split()[0].replace("+", "").upper() if out else ""
        if not raw:
            sq = f"squeue -j {int(job_id)} -h -o '%T' 2>/dev/null | head -1"
            try:
                r2 = self._ssh_shell(sq, timeout=15)
                raw = (r2.stdout or "").strip().upper()
            except (subprocess.TimeoutExpired, OSError):
                raw = ""
            if not raw:
                return "unknown", "sacct_empty"
        if raw in _SLURM_RUNNING_STATES:
            return "running", raw
        if raw in _SLURM_OK_STATES:
            return "completed", raw
        if raw in _SLURM_FAIL_STATES:
            return "failed", raw
        return "unknown", raw

    def is_process_alive(self, pid: int) -> bool:
        """Alive iff the Slurm job is in a running-bucket state. Indeterminate
        probes keep the job alive only for a bounded number of consecutive
        polls; a job is also force-reaped past ``--time`` + buffer. Both bounds
        guarantee the monitor's polling loop always terminates."""
        job_id = int(pid)
        now = time.time()
        first = self._first_seen.setdefault(job_id, now)
        bucket, raw = self._sacct_state(job_id)
        if bucket == "running":
            # PENDING/RUNNING/etc. are authoritative. A long queue wait is NOT
            # bounded by --time (which only counts while RUNNING), so never reap
            # a job sacct still confirms is queued or running.
            self._unknown_count[job_id] = 0
            return True
        if bucket in ("completed", "failed"):
            self._last_terminal[job_id] = raw
            return False
        # Indeterminate (ssh/sacct unreachable, or the job purged from both
        # sacct and squeue). Two bounds keep the monitor's polling loop finite
        # WITHOUT ever reaping a job sacct confirms is live:
        #   - a wall-clock backstop: Slurm enforces --time, so once --time +
        #     buffer has elapsed and we STILL cannot confirm the job, it is
        #     almost certainly gone;
        #   - a consecutive-unknown grace for shorter outages.
        if now - first > self._time_cap_seconds + self.slurm_time_buffer:
            return False
        self._unknown_count[job_id] = self._unknown_count.get(job_id, 0) + 1
        return self._unknown_count[job_id] <= self.slurm_unknown_grace_polls

    def get_gpu_status(self) -> dict:
        """Report the partition's queue occupancy (login node has no usable
        nvidia-smi). Advisory only — the monitor just logs ``utilization``."""
        cmd = (
            "squeue --me -p " + shlex.quote(self.slurm_partition)
            + " --states=PD,R -h -o '%T' 2>/dev/null | sort | uniq -c"
        )
        pending = running = 0
        try:
            r = self._ssh_shell(cmd, timeout=20)
        except (subprocess.TimeoutExpired, OSError):
            return {
                "utilization": "slurm", "partition": self.slurm_partition,
                "pending": 0, "running": 0, "note": "squeue unavailable",
            }
        if r.returncode == 0:
            for line in (r.stdout or "").splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    count, state = int(parts[0]), parts[1].upper()
                    if state.startswith("PEND") or state == "PD":
                        pending = count
                    elif state.startswith("R"):
                        running = count
        return {
            "utilization": "slurm", "partition": self.slurm_partition,
            "pending": pending, "running": running,
        }

    def cancel(self, pid: int) -> bool:
        """Best-effort ``scancel`` for a Slurm job. Not wired into a caller yet
        (orphaned jobs are otherwise reclaimed by ``--time``); available for a
        future kill-on-shutdown path."""
        try:
            r = self._ssh_shell("scancel " + str(int(pid)), timeout=8)
            return r.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def last_terminal_state(self, pid: int) -> Optional[str]:
        """Raw sacct state of a finished job, if observed (e.g. ``TIMEOUT``)."""
        return self._last_terminal.get(int(pid))

    def final_status(self, pid: int) -> dict:
        """Real outcome from the observed ``sacct`` terminal state.

        ``success`` is True only for ``COMPLETED``; ``FAILED`` / ``TIMEOUT`` /
        ``CANCELLED`` / ``OUT_OF_MEMORY`` / … are reported as failures. If the
        job was never observed reaching a terminal state (e.g. the cluster went
        unreachable and it was reaped by the wall-clock backstop), the outcome
        is indeterminate.
        """
        raw = self._last_terminal.get(int(pid))
        if raw is None:
            return {"state": "unknown", "success": None}
        return {"state": raw, "success": raw in _SLURM_OK_STATES}


def build_execution_backend(config: Optional[dict], controller_workspace: Path) -> ExecutionBackend:
    """Construct the execution backend from project config."""
    config = config or {}
    execution = config.get("execution", {}) or {}
    mode = execution.get("mode", "local")

    if mode == "ssh":
        return SSHExecutionBackend(
            ssh_host=execution.get("ssh_host", ""),
            remote_workspace=execution.get("remote_workspace", ""),
            remote_python=execution.get("remote_python", "python3"),
            ssh_args=execution.get("ssh_args", []) or [],
        )
    if mode == "slurm":
        return SlurmExecutionBackend(
            ssh_host=execution.get("ssh_host", ""),
            remote_workspace=execution.get("remote_workspace", ""),
            remote_python=execution.get("remote_python", "python3"),
            ssh_args=execution.get("ssh_args", []) or [],
            slurm_partition=execution.get("slurm_partition", ""),
            slurm_time=execution.get("slurm_time", ""),
            slurm_gpus_per_job=execution.get("slurm_gpus_per_job"),
            slurm_gres=execution.get("slurm_gres", ""),
            slurm_qos=execution.get("slurm_qos", ""),
            slurm_account=execution.get("slurm_account", ""),
            slurm_setup=execution.get("slurm_setup", ""),
            slurm_extra_sbatch=execution.get("slurm_extra_sbatch", []) or [],
            slurm_unknown_grace_polls=int(execution.get("slurm_unknown_grace_polls", 4)),
            slurm_time_buffer=int(execution.get("slurm_time_buffer", 1800)),
        )
    if mode != "local":
        raise ValueError(f"Unknown execution.mode '{mode}'. Supported: local, ssh, slurm")
    return LocalExecutionBackend(controller_workspace)
