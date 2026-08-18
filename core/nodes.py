"""
AutoResearcher — LangGraph + LangChain 完整改造版

用 node.py 的模式重写 ResearchLoop：
  - StateGraph + MemorySaver 管理状态流转（替代 while 循环 + state.json）
  - @tool + create_agent 做 Worker 工具调用（替代自定义 <tool_call> 协议）
  - Supervisor 节点做路由器（和 node.py 完全一致）
  - 所有原版组件完整保留（memory/ledger/journal/safety/obsidian/monitor）

架构：
  START → supervisor → conditional_edge → think/execute/monitor/reflect → supervisor → ... → END
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import signal
import sys
import time
import argparse
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore  # 保留作为 fallback
from .rollback import SqliteCheckpointer, WorkspaceSnapshot, RollbackHandler
from .sandbox import Sandbox, resolve_sandbox
from .persistent_store import SqliteStore  # 生产级 SQLite 持久化 Store
from .event_journal import EventJournal  # append-only 跨进程事件日志（dashboard 观测通道）
from .eval import AgentRecorder  # Agent Eval 录制（config eval.enabled 时启用）
from .rag import RagKnowledgeBase  # 论文/文档知识库 RAG（复用 CrossProjectStore）

from .execution import (
    build_execution_backend,
    pid_alive,
    ensure_project_python,
    bind_python_argv,
    dryrun_interpreter_error,
    python_fingerprint,
    script_hash,
    _BARE_PYTHON_NAMES,
)
from .monitor import ExperimentMonitor
from .ledger import ExperimentLedger, detect_stagnation, check_phase_gate
from .journal import ResearchJournal
from .prompt_builder import PromptBuilder
from .obsidian import ObsidianExporter
from .audit import AuditLogger
from .user_profile import UserProfileStore
from .approval import ApprovalGate
from .cost_tracker import CostTracker
from .guardrails import InputGuard, OutputGuard
from .cross_project_memory import CrossProjectStore
from .retry import retry_llm_call, classify_error, ErrorCategory, FatalLLMError, LLMRetryError
from .bad_case_collector import BadCaseCollector
from .stream_guard import StreamOutputGuard, scan_full_output
from . import safety

logger = logging.getLogger("autoresearcher.nodes")

# think 在"目标已达成"时可能把元陈述写进 hypothesis 字段(如
# 「无需新假设,目标已达成。」)—— 不是可验证假设,结算时过滤不入账本。
_META_HYPOTHESIS_RE = re.compile(
    r"无需新假设|无新假设|不需要新假设|no new hypothesis|目标已达成|任务完成|无需假设",
    re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════
# 1. State — 共享状态（替代 MemoryManager 文件 + state.json）
# ═══════════════════════════════════════════════════════════════════

class ResearchState(TypedDict):
    """节点间共享状态。

    LangGraph 的 MemorySaver 在每次节点执行后自动保存 state，
    所以不再需要手动写 state.json 和 MEMORY_LOG.md。
    跨 session 恢复靠 .cycle_counter（最简持久化）。
    """
    # ── 任务定义（原 PROJECT_BRIEF.md，Agent 不修改）──
    task: str

    # ── 记忆（迁移到 Store，不再放 State 避免膨胀）──
    # milestones / decisions 现在通过 self.memory.get_log() 和 self.store 访问

    # ── 循环控制 ──
    cycle: int
    max_cycles: int
    cooldown: int

    # ── 路由（supervisor 写入，route_next 读取）──
    next_agent: str

    # ── 人工指令 ──
    directive: str

    # ── 各阶段结果 ──
    think_result: str
    execute_result: str
    reflect_result: str

    # ── 最终输出 ──
    final_answer: str

    # ── 多步实验计划（Plan-then-Execute with Replan，JSON 序列化）──
    # [{step_id, title, agent, status: pending|running|done|failed, result}]
    plan: str


# ═══════════════════════════════════════════════════════════════════
# 1.5 Leader 结构化输出模型（对齐 OpenAI with_structured_output）
# ═══════════════════════════════════════════════════════════════════

try:
    from pydantic import BaseModel, Field as PydanticField  # noqa: N812
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False


if _HAS_PYDANTIC:
    class LeaderDecision(BaseModel):
        """Leader (Think/Reflect) 的结构化输出。

        替代原来正则 ``{...}`` 模糊解析，用 LangChain
        ``with_structured_output(LeaderDecision)`` 获取确定性 JSON 路由指令。

        对齐 OpenAI Agents SDK：结构化输出消除解析失败风险，
        路由决策从文本猜测变为字段读取。
        """
        action: str = PydanticField(
            default="experiment",
            description="下一步动作: experiment / wait / finish / continue",
        )
        agent: str = PydanticField(
            default="code",
            description="执行 agent: code / idea / writing / none",
        )
        hypothesis: str = PydanticField(
            default="",
            description="本实验的科学假设（1-2 句话）",
        )
        task: str = PydanticField(
            default="",
            description="下发给 Worker 的具体任务描述",
        )
        next_stage: str = PydanticField(
            default="",
            description="期望的下一个阶段: execute / monitor / reflect / finish",
        )
        reason: str = PydanticField(
            default="",
            description="选择此动作的理由（1 句话）",
        )
        # ── REFLECT 输出(think 阶段可留空)──
        # 必须有这两个字段:with_structured_output 会静默丢弃 schema 外的字段,
        # 缺失会导致 milestone/decision 永不落盘(MEMORY_LOG/账本/反卡死空转)。
        milestone: str = PydanticField(
            default="",
            description="本轮的关键结果里程碑(有实验结论时填写,如 'Exp003: acc=0.79 best')",
        )
        decision: str = PydanticField(
            default="",
            description="本轮决策总结(下一步做什么、为什么,供记忆日志)",
        )
        # ── 多步实验计划（Plan-then-Execute with Replan）──
        # 首次 think 生成 3-5 步实验序列；有 plan 时引用下一个 pending 步骤即可
        plan: list = PydanticField(
            default=[],
            description=(
                "实验序列计划（可选）：[{step_id, title, agent, status}]。"
                "status ∈ pending/running/done/failed。"
                "首次规划时生成 3-5 步（如 baseline → 调参 → 换架构 → 消融）；"
                "已有 plan 时返回 [] 表示沿用"
            ),
        )
else:
    LeaderDecision = None  # type: ignore[assignment,misc]


# ═══════════════════════════════════════════════════════════════════
# 2. 工具 — LangChain @tool（完整复刻 ToolRegistry 的 10 个工具）
# ═══════════════════════════════════════════════════════════════════

_tool_workspace: Path = Path(".")
_tool_backend = None
_tool_memory = None        # MemoryManager 引用（log_memory 用）
_tool_python = ""          # 绑定的训练解释器（绝对路径；launch 命令里裸 python 会被替换成它）
_tool_config: dict = {}    # 项目配置（launch 惰性重解析环境时用）
_tool_sandbox = None       # 策略沙箱（权限分级 + 环境剥离）
_tool_approval = None      # HITL 审批门(off 模式为 None)
_file_read_cache: dict = {}  # read_file 去重缓存：路径 → (mtime, 行数)
_FILE_READ_CACHE_MAX = 512
# 受保护文件(写入拒绝;比较统一小写,防 Windows NTFS 大小写不敏感绕过)
_tool_protected = {n.lower() for n in
                   {"state.json", "MEMORY_LOG.md", "PROJECT_BRIEF.md",
                    ".lock", "dry_run_log.json"}}
# 敏感文件:内容含密钥/凭据,read_file 与 shell 读取一律拒绝
_SENSITIVE_FILES = {".env", ".env.local", "id_rsa", "id_ed25519", "id_dsa", "id_ecdsa"}
_SENSITIVE_SUFFIXES = (".pem", ".key", ".p12", ".pfx")

# ── 硬安全约束：危险命令 / 破坏性 Python / API key 读取一律拦截（代码强制，非 prompt）──
_DANGEROUS_BINS = {
    "rm", "sudo", "su", "mkfs", "dd", "shutdown", "reboot", "poweroff",
    "halt", "unlink", "format", "pkill", "killall", "kill", "mkfs.*",
}
_DESTRUCTIVE_PY_TOKENS = ("os.remove", "os.unlink", "shutil.rmtree", "shutil.move")
_APIKEY_READ_TOKENS = ("os.environ", "getenv", "environ[")
_SENSITIVE_VAR_PATTERNS = ("API_KEY", "APITOKEN", "AUTH_TOKEN", "SECRET", "PASSWORD")
# shell 解释器全集:任何解释器 + -c 都是一行命令注入入口(bash/sh 之外的
# zsh/ksh/dash/fish 曾可绕过黑名单)
_SHELL_INTERPRETERS = ("bash", "sh", "zsh", "ksh", "dash", "fish")
# 读取类命令:配合敏感文件检查,拦 `cat .env` 等
_FILE_READ_COMMANDS = ("cat", "type", "less", "more", "head", "tail", "sed",
                       "awk", "grep", "find", "xxd", "od")


def _guard_shell_command(command: str) -> None:
    """校验一条 Shell 命令是否安全。不安全则抛 ValueError（agent 无法绕过）。"""
    if not command or not command.strip():
        raise ValueError("Command cannot be empty")

    import shlex
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"Invalid command syntax: {exc}") from exc
    if not argv:
        raise ValueError("Command cannot be empty")

    # 1. 危险可执行文件（含 shell 解释器 -c 注入绕过;大小写归一防 Windows 绕过）
    first_raw = Path(argv[0]).name
    first = first_raw.lower()
    if first in _DANGEROUS_BINS:
        raise ValueError(f"Blocked executable: {argv[0]}")
    if first in _SHELL_INTERPRETERS and any(
            flag in ("-c", "-e", "-s") for flag in argv[1:3]):
        raise ValueError(f"Blocked: {first} -c …（防绕过）")
    if first in ("cmd", "cmd.exe") and any(flag == "/c" for flag in argv[1:3]):
        raise ValueError("Blocked: cmd /c（Windows 命令注入）")

    # 2. 破坏性 Python 一行命令（python.exe 等大小写/扩展名变体同样拦截）
    if first.startswith("python") or first in ("py", "python3.exe"):
        joined = " ".join(argv)
        if any(tok in joined for tok in _DESTRUCTIVE_PY_TOKENS):
            raise ValueError("Blocked: destructive python call (os.remove / rmtree)")

    # 3. 读取环境变量（防 agent 偷 API key）：
    #    a) python 侧 os.environ/getenv
    #    b) shell 侧 $XXX_API_KEY / $XXX_TOKEN / printenv XXX / env 等
    cmd_upper = command.upper()
    has_sensitive = any(p in cmd_upper for p in _SENSITIVE_VAR_PATTERNS)
    if has_sensitive:
        reads_env = any(tok in command for tok in _APIKEY_READ_TOKENS)
        shell_read = ("$" in command) or first in ("printenv", "env")
        if reads_env or shell_read:
            raise ValueError("Blocked: reading environment secrets")

    # 4. 读取敏感文件（cat .env / head id_rsa 等）:API key 防泄露的最后一公里
    if first in _FILE_READ_COMMANDS:
        for tok in argv[1:]:
            low = tok.lower().rstrip("/")
            if low in _SENSITIVE_FILES or low.endswith(_SENSITIVE_SUFFIXES):
                raise ValueError(f"Blocked: reading sensitive file: {tok}")


def set_tool_context(workspace: Path, backend, memory=None, python_exe: str = "",
                     config: Optional[dict] = None,
                     sandbox=None, approval=None) -> None:
    global _tool_workspace, _tool_backend, _tool_memory, _tool_python, _tool_config
    global _tool_sandbox, _tool_approval
    _tool_workspace = workspace
    _tool_backend = backend
    _tool_memory = memory
    _tool_python = python_exe
    _tool_config = config or {}
    _tool_sandbox = sandbox or Sandbox()  # 默认 workspace-write(与旧行为一致)
    _tool_approval = approval  # HITL 审批门(off 时 None 等价不启用)


def _sandbox_gate(operation: str):
    """策略沙箱门控:read-only 模式拒绝写/执行类操作。返回错误串或 None。"""
    if _tool_sandbox is None:
        return None
    if operation in ("write", "exec") and not _tool_sandbox.allow_write:
        return _tool_sandbox.reject_reason() or "sandbox denied this operation"
    return None


def _resolve_path(path: str):
    """解析工具可见路径并强制约束在工作区内。

    返回 Path;越界(`..`、绝对路径、符号链接逃逸)返回 None。
    调用方收到 None 时必须返回「路径越界」错误,绝不能读写该路径。
    resolve() 会跟随符号链接,因此工作区内的链接指向外部也会被拦下。
    """
    p = path.strip().replace("\\", "/")
    # 绝对路径必须显式拒绝:直接 join 会被 pathlib 的盘符/根路径语义
    # 静默改写(如 strip('/') 后 '/etc/x' → 工作区内 'etc/x'),平台行为不一致。
    if p.startswith("/") or (len(p) >= 2 and p[1] == ":"):
        return None
    target = _tool_workspace if p == "." else _tool_workspace / p.strip("/")
    try:
        resolved = target.resolve()
    except OSError:
        return None
    base = _tool_workspace.resolve()
    if resolved != base and base not in resolved.parents:
        return None
    return target


# ── 基础工具 ──

@tool
def list_files(path: str = ".") -> str:
    """List files in a directory (non-recursive). Args: path — directory path, default '.'. """
    target = _resolve_path(path)
    if target is None:
        return f"(path out of workspace bounds: {path})"
    if not target.exists():
        return f"(directory not found: {path})"
    items = sorted(target.iterdir())
    lines = []
    for item in items:
        try:
            rel = item.relative_to(_tool_workspace).as_posix()
        except ValueError:
            rel = str(item)
        lines.append(f"- {rel}/" if item.is_dir() else f"- {rel}")
    return "\n".join(lines) or "(empty)"


@tool
def list_tree(path: str = ".", max_depth: int = 3, max_entries: int = 300) -> str:
    """List a directory tree recursively (depth-limited). Skips .git, __pycache__, etc.
    Directories end with '/'. Args: path — root; max_depth — recursion depth; max_entries — cap."""
    target = _resolve_path(path)
    if target is None:
        return f"(path out of workspace bounds: {path})"
    if not target.exists():
        return f"(directory not found: {path})"
    skip = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
    entries = []
    for item in sorted(target.rglob("*")):
        if item.is_symlink():
            continue  # 符号链接一律不展开（防指向工作区外的内容泄漏）
        if any(s in item.parts for s in skip):
            continue
        depth = len(item.relative_to(target).parts)
        if depth > max_depth:
            continue
        if len(entries) >= max_entries:
            entries.append("... (truncated)")
            break
        try:
            rel = item.relative_to(_tool_workspace).as_posix()
        except ValueError:
            rel = str(item)
        entries.append(f"{'  ' * depth}- {rel}/" if item.is_dir() else f"{'  ' * depth}- {rel}")
    return "\n".join(entries) or "(empty)"


# 对齐 Claude Code 源码（FileReadTool）：MAX_LINES_TO_READ = 2000
# "让模型拿到足够信息，但不要一口气把超大文件全塞进上下文"
_READ_MAX_LINES = 2000


@tool
def read_file(path: str, start_line: int = 0, end_line: int = 0) -> str:
    """Read file content. Args: path — file path; start_line/end_line — line range (optional, 1-indexed).

    Usage rules:
    - **Dedicated tool for reading**: do NOT use run_shell cat/type to read files
    - List directories with list_files; search content with search_code
    - Large files: at most 2000 lines by default; use start_line/end_line for fine-grained reads
    - If the same file was read before and unchanged, you will get a "content unchanged"
      notice; **do NOT re-read it** — act on the information you already have
    """
    target = _resolve_path(path)
    if target is None:
        return f"(path out of workspace bounds: {path})"
    if (target.name.lower() in _SENSITIVE_FILES
            or target.suffix.lower() in _SENSITIVE_SUFFIXES):
        return f"(sensitive file, read denied: {path})"
    if not target.is_file():
        return f"file not found: {path}"
    try:
        mtime = target.stat().st_mtime
        cache_key = str(target)
        cached = _file_read_cache.get(cache_key)
        # 去重只在「上次整文件读取已完整交付」时生效:内容未变 且 上次
        # 没有被截断(文件 ≤ 回灌阈值)。否则 agent 实际只看到文件头,
        # 提示"不要重复读取"会逼它走 shell 转储(冒烟实测的侦察循环)。
        if (cached and cached[0] == mtime and start_line == 0
                and end_line == 0 and cached[2]):
            return (f"[File {path} was already read before and is unchanged "
                    f"(mtime={mtime:.0f}, {cached[1]} lines). "
                    f"Do NOT re-read it — act on the information you already have.]")
        lines = target.read_text(encoding="utf-8").split("\n")
        # 去重缓存有界:超限清空(长期运行防内存缓慢增长)
        if len(_file_read_cache) >= _FILE_READ_CACHE_MAX:
            _file_read_cache.clear()
        _file_read_cache[cache_key] = (mtime, len(lines), False)
    except Exception as exc:
        return f"read failed: {exc}"
    if start_line > 0 or end_line > 0:
        s = max(0, start_line - 1) if start_line > 0 else 0
        e = min(len(lines), end_line) if end_line > 0 else len(lines)
        lines = lines[s:e]
        return "\n".join(lines)
    if len(lines) > _READ_MAX_LINES:
        head = "\n".join(lines[:_READ_MAX_LINES])
        return (f"{head}\n\n"
                f"... [File has {len(lines)} lines; only the first {_READ_MAX_LINES} "
                f"shown. Use start_line/end_line to read specific ranges.]")
    content = "\n".join(lines)
    # 记录「本次是否完整交付」:内容 ≤ 回灌阈值 → 下次重复读可去重
    _file_read_cache[cache_key] = (
        mtime, len(lines), len(content) <= _READ_FILE_SUMMARY_CHARS)
    return content


@tool
def write_file(path: str, content: str) -> str:
    """Write a file into the workspace (cannot overwrite protected files).
    Args: path — file path; content — full file content."""
    import os as _os
    import tempfile as _tempfile
    target = _resolve_path(path)
    if target is None:
        return json.dumps({"error": f"path out of workspace bounds: {path}"})
    gate = _sandbox_gate("write")
    if gate:
        return json.dumps({"error": gate})
    if (target.name.lower() in _tool_protected
            or target.name.lower() in _SENSITIVE_FILES
            or target.suffix.lower() in _SENSITIVE_SUFFIXES):
        return json.dumps({"error": f"protected file, cannot overwrite: {path}"})
    target.parent.mkdir(parents=True, exist_ok=True)
    # 原子写：先写临时文件再 rename，防磁盘满时截断写坏已有文件
    try:
        fd, tmp = _tempfile.mkstemp(dir=str(target.parent), suffix=".tmp",
                                    prefix=".ar_", text=True)
        try:
            with _os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                _os.fsync(f.fileno())
            _os.replace(tmp, target)   # 原子替换
        except BaseException:
            try:
                _os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError as exc:
        return json.dumps({"error": f"write failed (possibly disk full): {exc}"})
    return f"written to {target.relative_to(_tool_workspace).as_posix()}"


@tool
def search_code(pattern: str, path: str = ".", max_results: int = 50,
                ignore_case: bool = False) -> str:
    """Regex-search file contents (grep style). Returns file names, line numbers, matching lines.
    Args: pattern — regex; path — root (default '.'); max_results — cap; ignore_case — flag."""
    import re as _re
    target = _resolve_path(path)
    if target is None:
        return f"(path out of workspace bounds: {path})"
    if not target.exists():
        return f"(path not found: {path})"
    flags = _re.IGNORECASE if ignore_case else 0
    try:
        regex = _re.compile(pattern, flags)
    except _re.error as exc:
        return f"regex error: {exc}"
    hits = []
    files = list(target.rglob("*")) if target.is_dir() else [target] if target.is_file() else []
    for f in files:
        if not f.is_file() or f.is_symlink():
            continue  # 符号链接不读（防指向工作区外的文件内容泄漏）
        if any(s in f.parts for s in {".git", "__pycache__", "node_modules"}):
            continue
        try:
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").split("\n"), 1):
                if regex.search(line):
                    hits.append({"file": f.relative_to(_tool_workspace).as_posix(), "line": i, "text": line[:200]})
                    if len(hits) >= max_results:
                        break
        except Exception:
            continue
        if len(hits) >= max_results:
            break
    return json.dumps({"matches": hits, "count": len(hits)}, ensure_ascii=False, indent=2)


@tool
def run_shell(command: str, timeout: int = 120) -> str:
    """Execute a shell command. Args: command — the command; timeout — seconds.

    Usage rules:
    - **Prefer dedicated tools**: read files with read_file, write with write_file,
      list directories with list_files, search code with search_code, launch training
      with launch_experiment — do NOT use Shell when a dedicated tool exists
    - **Windows note**: this environment is Windows (cmd semantics); Unix commands
      like cp/cat/mv/sed/grep do not exist (they fail with WinError 2). For
      copying/renaming use read_file + write_file; training templates are already
      seeded in the workspace (train_template.py / train.py) — edit them directly.
    - **Do NOT** use `python -c` to dump file contents into temp files and read them
      (read_file supports whole-file and line-range reads; results are returned in full)
    - **Do NOT** launch training with Shell (python train.py ...) — always use
      launch_experiment (it handles dry-run validation, environment, and logs)
    - Good Shell use cases: quick checks, environment probing, data preprocessing,
      file comparison
    """
    import shlex
    import subprocess
    gate = _sandbox_gate("exec")
    if gate:
        return gate
    try:
        _guard_shell_command(command)  # 硬安全约束
        # 无 shell 执行（与 launch_experiment / 旧引擎一致）：
        # shell=True 会让 `;`/`&&`/`|` 注入真正执行，而 shlex 拆 argv 后
        # 这些符号只是首个可执行文件的普通参数 —— 注入在机制上不可能。
        argv = shlex.split(command)
        result = subprocess.run(
            argv, shell=False, cwd=str(_tool_workspace),
            env=_tool_sandbox.environment(),  # 环境剥离:API key 不进子进程
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        parts = []
        if out:
            parts.append(out)
        if err:
            parts.append(f"[stderr]\n{err}")
        parts.append(f"[exit_code={result.returncode}]")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"command timeout ({timeout}s): {command}"
    except Exception as exc:
        return f"command failed: {exc}"


@tool
def launch_experiment(command: str, log_file: str, gpu: str = "",
                      dry_run: bool = False) -> str:
    """Launch a training experiment (nohup-style). Returns JSON with PID and log_file.

    Usage rules:
    - **The ONLY entry point for launching training**: never use run_shell to start training
    - **Dry runs also go through this tool** (dry_run=true): the system executes the
      dry run with the SAME interpreter as training and writes the authoritative record —
      dry-run/training environment mismatch is mechanism-impossible
    - **After a successful launch your task is done**: waiting for training is the
      system monitor node's job — **do NOT** monitor progress with sleep/tail/timeout
    - Multiple experiments: call this tool once per experiment, then return one summary
    - The returned PID/log_file are the authoritative basis for monitoring; report them in your summary
    """
    global _tool_python  # 惰性重解析成功后回写绑定值
    import shlex as _shlex
    import shutil as _shutil
    if _tool_backend is None:
        return json.dumps({"error": "no execution backend", "experiment_launched": False})
    try:
        gate = _sandbox_gate("exec")
        if gate:
            return json.dumps({"error": gate, "experiment_launched": False})

        # 1. 硬安全约束：危险命令一律拒绝
        _guard_shell_command(command)

        # 1.4 HITL 审批门(B0/B1):exception/all 模式下 launch 需人工批准。
        #    批一次缓存,后续复用(不打断自主节奏);等待由 execute_node
        #    零 LLM 轮询完成,不阻塞在工具内。
        approval = _tool_approval
        if approval is not None:
            cache_key = approval.cache_key_for(
                "launch_experiment",
                {"command": command[:200], "log_file": log_file})
            cached = approval.cached_decision(cache_key)
            if cached == "denied":
                return json.dumps({
                    "error": "a human denied this experiment launch; do not retry it.",
                    "experiment_launched": False,
                }, ensure_ascii=False)
            if cached != "approved":
                needs, reason = approval.needs_approval(
                    "launch_experiment",
                    {"command": command[:200], "log_file": log_file})
                if needs:
                    req = approval.create_request(
                        "launch_experiment",
                        {"command": command[:200], "log_file": log_file})
                    return json.dumps({
                        "error": f"waiting for human approval (id={req.id}): {reason}",
                        "approval_pending": True,
                        "approval_id": req.id,
                        "cache_key": cache_key,
                        "command": command,
                        "log_file": log_file,
                        "experiment_launched": False,
                    }, ensure_ascii=False)

        # 1.5 log_file 必须落在工作区内（防越界写日志）
        log_path = _resolve_path(log_file)
        if log_path is None:
            return json.dumps({
                "error": f"log_file out of workspace bounds: {log_file}",
                "experiment_launched": False,
            })
        log_file_rel = log_path.relative_to(_tool_workspace).as_posix()

        # 1.6 幂等性：已有活跃训练时拒绝重复启动（防重复扣 GPU 时长/资源竞争）。
        #     `.last_launch.json` 由上次 launch 写入;pid 仍存活 → 视为训练中。
        try:
            last_launch = json.loads(
                (_tool_workspace / ".last_launch.json").read_text(encoding="utf-8")
            )
            if last_launch.get("pid") and pid_alive(int(last_launch["pid"])):
                return json.dumps({
                    "error": f"training already running (pid={last_launch['pid']}, "
                             f"log={last_launch.get('log_file', '')}). "
                             "Do not launch again; let the monitor handle it, or terminate it first.",
                    "experiment_launched": False,
                })
        except (OSError, ValueError, json.JSONDecodeError):
            pass  # 无记录/记录损坏 → 放行

        # 2. 磁盘空间检查：不足则拒绝（防 OOM/写坏用户数据）
        min_free = float(os.environ.get("AR_MIN_FREE_GB", "0.5"))
        try:
            free_gb = _shutil.disk_usage(_tool_workspace).free / (1024 ** 3)
            if free_gb < min_free:
                return json.dumps({
                    "error": f"insufficient disk space ({free_gb:.2f} GB free < {min_free} GB); refused to launch to protect data",
                    "experiment_launched": False,
                })
        except OSError:
            pass  # 磁盘检查失败不阻塞（保守放行）

        # 3. 强制 dry-run：真实训练前必须已有 dry_run_log.json（非空）。
        #    dry_run=true 模式本身就是要生成 marker → 跳过本检查。
        dry_run_marker = _tool_workspace / "dry_run_log.json"
        if not dry_run and not (
                dry_run_marker.exists() and dry_run_marker.stat().st_size > 0):
            return json.dumps({
                "error": "no successful dry-run detected (dry_run_log.json missing). "
                         "You MUST run a dry-run first to validate the script, then launch real training.",
                "experiment_launched": False,
            })

        # ── 环境绑定解析（干跑与真实训练共用同一解释器,一致性的事实源）──
        # 放在所有快速失败检查之后:惰性重解析可能触发项目 venv 创建
        # （= torch 下载,2GB）,只有"真的要 launch"才允许走到这一步。
        argv_raw = command.split()
        script_arg = next(
            (a for a in argv_raw[1:] if a.endswith(".py")), "train.py"
        )
        first_name = Path(argv_raw[0]).name if argv_raw else ""
        is_bare_python = (first_name.lower() in _BARE_PYTHON_NAMES
                          and Path(argv_raw[0]).name == argv_raw[0]) if argv_raw else False

        python_exe = _tool_python
        if not python_exe and is_bare_python:
            # 惰性重解析：agent 可能已在本轮修复环境（pip install / 建 env 等）。
            python_exe = ensure_project_python(_tool_config, _tool_workspace)
            if python_exe:
                _tool_python = python_exe
                logger.info("launch 惰性重解析训练解释器成功: %s", python_exe)

        if is_bare_python and not python_exe:
            # 裸 python 且系统未解析到训练解释器 → 检查是否在自动创建环境
            # (uv/conda/venv + torch,异步):创建中 → 明确提示稍后重试,
            # 失败 → 报错(含日志尾部),而不是静默降级回「碰运气」的旧行为。
            from .execution import _env_status, _settle_env_status
            env_st = _settle_env_status(_tool_workspace)
            st = env_st.get("status", "")
            if st in ("creating", "installing"):
                elapsed = round(time.time() - float(env_st.get("started_at", time.time())))
                return json.dumps({
                    "error": f"training environment is being created automatically ({env_st.get('creator','?')}, "
                             f"{elapsed}s elapsed, phase {env_st.get('phase','')}). "
                             "Retry this tool later (environment creation does not consume LLM cost).",
                    "experiment_launched": False,
                }, ensure_ascii=False)
            if st == "failed":
                return json.dumps({
                    "error": "training environment auto-creation failed: "
                             f"{env_st.get('error', 'unknown')}. "
                             "Check workspace/.trainenv_install.log, "
                             "or create the environment manually and set execution.python in config.yaml.",
                    "experiment_launched": False,
                }, ensure_ascii=False)
            return json.dumps({
                "error": "no usable training interpreter found (python with torch+torchvision). "
                         "Set an absolute path in config.yaml execution.python, "
                         "or install the dependencies on this machine and retry.",
                "experiment_launched": False,
            }, ensure_ascii=False)
        argv = bind_python_argv(argv_raw, python_exe)

        # ── 训练脚本模板硬校验(干跑与真实启动共用,先于一切执行)──
        # 脚本必须含 config 驱动的 checkpoint 逻辑契约
        # (save_every_n_epochs / best_model.pth / log_metrics)。
        # 干跑同样执行本校验:契约不满足就不执行、不写 marker —— 否则出现
        # 「干跑通过 → 真实启动被拒」的返工循环(冒烟实测:agent 自写脚本
        # 干跑 ok,真实 launch 却报缺结构,浪费整轮 max_turns)。
        # 脚本名 = argv 里第一个 .py 参数（排除解释器本身）—— 末尾参数
        # (如 --lr 0.1)会让 command.split()[-1] 取错,导致校验被绕过。
        target = _resolve_path(script_arg)
        if target is not None and target.exists() and target.suffix == ".py":
            src = target.read_text(encoding="utf-8", errors="replace")
            missing = [tok for tok in ("save_every_n_epochs", "best_model.pth",
                                       "log_metrics")
                       if tok not in src]
            if missing:
                return json.dumps({
                    "error": f"training script {target.name} is missing required "
                             f"structure ({', '.join(missing)}). "
                             "The workspace has pre-seeded template copies "
                             "(train_template.py / train.py); edit the TODO regions "
                             "(model / data loading / training loop) on top of the "
                             "template. log_metrics is the metric-output contract and "
                             "must NOT be deleted.",
                    "experiment_launched": False,
                })

        # ── DRY-RUN 模式：系统执行干跑,系统写权威记录 ──
        # 旧流程让 agent 用 run_shell 手动干跑 + 脚本自己写 marker →
        # 解释器可能不一致、marker 可伪造/遗漏。新流程:
        #   同解释器 + 同 argv 绑定 + 追加 --dry-run(模板契约,模板硬校验保证支持)
        #   成功 → 系统写 dry_run_log.json(interpreter + script_hash + 依赖指纹)
        if dry_run:
            dry_argv = list(argv) + ["--dry-run"]
            dry_result = _tool_backend.run_command(dry_argv, timeout=600)
            if dry_result.get("returncode") != 0:
                err = (dry_result.get("stderr")
                       or dry_result.get("stdout") or "").strip()
                return json.dumps({
                    "error": f"dry-run 失败(退出码 {dry_result.get('returncode')}):\n{err[-600:]}",
                    "dry_run": "failed",
                    "experiment_launched": False,
                }, ensure_ascii=False)
            marker_data = {
                "dry_run": True, "ok": True, "time": time.time(),
                "interpreter": python_exe or argv[0],
                "script_hash": script_hash(_tool_workspace, script_arg),
                "fingerprint": python_fingerprint(python_exe) if python_exe else {},
            }
            try:
                (_tool_workspace / "dry_run_log.json").write_text(
                    json.dumps(marker_data, ensure_ascii=False), encoding="utf-8")
            except OSError as exc:
                return json.dumps({
                    "error": f"dry-run record write failed: {exc}",
                    "dry_run": "failed", "experiment_launched": False,
                })
            return json.dumps({
                "dry_run": "passed",
                "interpreter": marker_data["interpreter"],
                "script_hash": marker_data["script_hash"][:12],
                "experiment_launched": False,
            }, ensure_ascii=False)

        # 5. 干跑/训练一致性指纹校验（解释器 + 脚本内容 + 依赖版本）——
        #    干跑记录由系统在 dry_run 模式写入;这里校验三者与当前环境一致,
        #    任何一项漂移(换解释器/改脚本/pip 升级)都强制重新干跑。
        if python_exe:
            try:
                dry_run_data = json.loads(dry_run_marker.read_text(encoding="utf-8"))
                mismatch = dryrun_interpreter_error(dry_run_data, python_exe)
                if mismatch:
                    return json.dumps({
                        "error": mismatch,
                        "experiment_launched": False,
                    }, ensure_ascii=False)

                recorded_hash = str(dry_run_data.get("script_hash", "") or "")
                if recorded_hash:
                    current_hash = script_hash(_tool_workspace, script_arg)
                    if current_hash and current_hash != recorded_hash:
                        return json.dumps({
                            "error": "training script was modified after the dry run "
                                     "(content fingerprint mismatch). Re-run the dry run "
                                     "(launch_experiment dry_run=true) before launching real training.",
                            "experiment_launched": False,
                        }, ensure_ascii=False)

                recorded_torch = str(
                    (dry_run_data.get("fingerprint") or {}).get("torch", "") or "")
                if recorded_torch:
                    current_torch = str(
                        python_fingerprint(python_exe).get("torch", "") or "")
                    if current_torch and current_torch != recorded_torch:
                        return json.dumps({
                            "error": f"dependency fingerprint mismatch: dry run used "
                                     f"torch={recorded_torch}, current torch={current_torch}. "
                                     "Environment changed; re-run the dry run.",
                            "experiment_launched": False,
                        }, ensure_ascii=False)
            except (OSError, json.JSONDecodeError):
                pass  # 干跑记录不可读 → 前面的存在性检查已兜底

        env = {"CUDA_VISIBLE_DEVICES": gpu} if gpu else {}
        result = _tool_backend.launch_command(argv=argv, log_file=log_file_rel, env=env)
        result["experiment_launched"] = True
        # 记录最近一次 launch：崩溃重启后用于孤儿训练检测
        try:
            (_tool_workspace / ".last_launch.json").write_text(
                json.dumps({"pid": result.get("pid"), "ts": time.time(),
                            "log_file": log_file_rel}, ensure_ascii=False),
                encoding="utf-8")
        except OSError:
            pass
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc), "experiment_launched": False})


@tool
def git_clone(repo_url: str, dest: str = "") -> str:
    """Clone a public code repository into the workspace (repos/ directory).
    Args: repo_url — https URL; dest — optional subdirectory.

    Security: https only, host whitelist (github/gitee/gitlab, configurable),
    target fixed to workspace/repos/, no injection — this tool is the ONLY
    entry point for fetching code (vs bare git via run_shell).
    """
    import shlex as _shlex
    import subprocess as _subprocess

    if not repo_url.startswith("https://"):
        return f"(https URLs only: {repo_url[:60]})"
    allowed_hosts = set(
        os.environ.get("AR_GIT_ALLOWED_HOSTS",
                       "github.com,gitee.com,gitlab.com").split(","))
    from urllib.parse import urlparse
    host = urlparse(repo_url).netloc.lower()
    if host not in allowed_hosts:
        return f"(host not whitelisted: {host} — set AR_GIT_ALLOWED_HOSTS)"
    if " " in repo_url or any(ch in repo_url for ch in (";", "|", "&", "$", "`")):
        return "(URL contains illegal characters)"

    repos_dir = _tool_workspace / "repos"
    target = repos_dir / (dest.strip() if dest.strip() else
                          Path(urlparse(repo_url).path).stem or "repo")
    # 目标必须在 workspace/repos 内(防路径逃逸)
    resolved = _resolve_path(str(target.relative_to(_tool_workspace)))
    if resolved is None:
        return "(target path out of bounds)"
    if (target / ".git").exists():
        return f"(already exists: {target})"

    repos_dir.mkdir(parents=True, exist_ok=True)
    try:
        proc = _subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(target)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=300,
        )
    except FileNotFoundError:
        return "(git not installed or not on PATH)"
    if proc.returncode != 0:
        return f"(clone failed: {(proc.stderr or proc.stdout or '')[-300:]})"
    return f"(cloned to {target.relative_to(_tool_workspace).as_posix()})"


# ── git 工具（只读：复现报告的代码快照证据）──

def _git(argv: list[str]) -> str:
    """只读 git 调用,固定 `-C workspace` 执行,无参数注入面。"""
    import subprocess
    try:
        proc = subprocess.run(
            ["git", "-C", str(_tool_workspace), *argv],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30,
        )
    except FileNotFoundError:
        return "(git not installed or not on PATH)"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return f"(git failed: {err[:300]})"
    return proc.stdout.strip() or "(空)"


@tool
def git_status() -> str:
    """Show workspace code version status (read-only).

    Returns current branch + uncommitted changes summary (porcelain). Use before
    launch to confirm code state, and after experiments to record the reproduction
    snapshot (works with the reproducibility ledger).
    """
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _git(["status", "--porcelain"])
    return f"branch: {branch}\n{status}"


@tool
def git_diff(path: str = ".") -> str:
    """Show uncommitted code changes (read-only).

    Args: path — workspace-relative path, default '.'. Returns a diff --stat
    summary + the actual diff (truncated to 200 lines to avoid context explosion).
    """
    if not path.strip():
        path = "."
    resolved = _resolve_path(path)
    if resolved is None:
        return f"(path out of workspace bounds: {path})"
    stat = _git(["diff", "--stat", "--", path])
    diff = _git(["diff", "--", path])
    lines = diff.splitlines()
    if len(lines) > 200:
        diff = "\n".join(lines[:200]) + f"\n... [diff 共 {len(lines)} 行,已截断]"
    return f"{stat}\n\n{diff}"


# ── 论文搜索工具 ──

@tool
def search_papers(query: str, limit: int = 10, year: str = "") -> str:
    """Search academic papers (Semantic Scholar API). Args: query — search terms;
    limit — max results; year — year filter like '2024-2026'."""
    limit = max(1, min(int(limit), 50))
    params = {"query": query, "limit": limit, "fields": "title,year,authors,abstract,citationCount,url"}
    if year:
        params["year"] = year
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AutoResearcher/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return json.dumps({"papers": data.get("data", [])[:limit]}, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Search failed: {e}"})


@tool
def search_arxiv(query: str, limit: int = 10, category: str = "") -> str:
    """Search recent arXiv preprints. Args: query — search terms; limit — max results;
    category — category filter like 'cs.CV'."""
    limit = max(1, min(int(limit), 50))
    search_query = f"all:{query}"
    if category:
        search_query = f"cat:{category} AND ({search_query})"
    params = {"search_query": search_query, "start": 0, "max_results": limit,
              "sortBy": "submittedDate", "sortOrder": "descending"}
    url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AutoResearcher/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            root = ET.fromstring(resp.read())
    except Exception as e:
        return json.dumps({"error": f"arXiv search failed: {e}"})
    ns = {"a": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("a:entry", ns):
        arxiv_url = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
        papers.append({
            "arxiv_id": arxiv_url.rsplit("/", 1)[-1],
            "title": " ".join((entry.findtext("a:title", default="", namespaces=ns) or "").split()),
            "published": (entry.findtext("a:published", default="", namespaces=ns) or "").strip(),
            "authors": [(a.findtext("a:name", default="", namespaces=ns) or "").strip()
                        for a in entry.findall("a:author", ns)],
            "abstract": " ".join((entry.findtext("a:summary", default="", namespaces=ns) or "").split()),
            "url": arxiv_url,
        })
    return json.dumps({"papers": papers[:limit]}, ensure_ascii=False, indent=2)


@tool
def get_paper(paper_id: str, include_references: bool = True,
              include_citations: bool = True) -> str:
    """Get details of a single paper (with references/citations).
    Args: paper_id — 'arXiv:2401.01234' or a Semantic Scholar ID."""
    if not paper_id.strip():
        return json.dumps({"error": "paper_id must not be empty"})
    fields = ["title", "year", "authors", "abstract", "citationCount", "venue", "url"]
    if include_references:
        fields += ["references.title", "references.year", "references.externalIds"]
    if include_citations:
        fields += ["citations.title", "citations.year", "citations.externalIds"]
    quoted = urllib.parse.quote(paper_id.strip(), safe=":/")
    url = f"https://api.semanticscholar.org/graph/v1/paper/{quoted}?{urllib.parse.urlencode({'fields': ','.join(fields)})}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AutoResearcher/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        return json.dumps({"error": f"get_paper failed: {e}"})
    for key in ("references", "citations"):
        if isinstance(data.get(key), list):
            data[key] = data[key][:25]
    return json.dumps(data, ensure_ascii=False, indent=2)


# ── 记忆工具（Leader 专用）──

@tool
def log_memory(type: str, entry: str) -> str:
    """Record to the memory system. Args: type — 'milestone' or 'decision'; entry — content."""
    if _tool_memory is not None:
        if type == "milestone":
            _tool_memory.log_milestone(entry)
        elif type == "decision":
            _tool_memory.log_decision(entry)
    return json.dumps({"status": "logged", "type": type, "entry": entry[:200]})


# ── Worker 工具注册表（单步循环用）──
# 对齐 create_agent 的工具列表；通过 .func 直接调用，保留安全守卫。
TOOL_FUNCTIONS: dict[str, list] = {
    "code": [
        ("run_shell", run_shell), ("launch_experiment", launch_experiment),
        ("write_file", write_file), ("read_file", read_file),
        ("list_files", list_files), ("list_tree", list_tree),
        ("search_code", search_code),
        ("git_status", git_status), ("git_diff", git_diff),
        ("git_clone", git_clone),
    ],
    "idea": [
        ("search_papers", search_papers), ("search_arxiv", search_arxiv),
        ("get_paper", get_paper), ("write_file", write_file),
        ("read_file", read_file),
        # T6 实测:idea_agent 需要盘点本地论文库(文献在 workspace/literature/),
        # 缺 list_files/list_tree 时它只能 read_file 猜路径(版本号猜错多轮)
        ("list_files", list_files), ("list_tree", list_tree),
    ],
    "writing": [
        ("write_file", write_file), ("read_file", read_file),
        ("list_files", list_files), ("search_code", search_code),
    ],
    # review：只读工具（无 launch/write/run_shell 之外的能力），
    # 职责是审查 code agent 写的训练脚本，降低干跑失败率
    "review": [
        ("read_file", read_file), ("search_code", search_code),
        ("list_files", list_files), ("list_tree", list_tree),
        ("run_shell", run_shell),  # 用于 python -m py_compile 语法检查
    ],
}
WORKER_MAX_TURNS = {"code": 60, "idea": 30, "writing": 12, "review": 15}


# ═══════════════════════════════════════════════════════════════════
# 3. System Prompts
# ═══════════════════════════════════════════════════════════════════

SUMMARY_PROMPT = """
你是 supervisor agent。根据研究进展和关键结果，用中文做简短总结。
""".strip()

from .prompts import (  # noqa: E402 提示词单一事实源 = agents/*.md
    CODE_AGENT_PROMPT,
    IDEA_AGENT_PROMPT,
    LEADER_REFLECT_PROMPT,
    LEADER_THINK_PROMPT,
    REVIEW_AGENT_PROMPT,
    WRITING_AGENT_PROMPT,
    _AGENTS_DIR,
    _WORKER_PROMPTS,
    _load_worker_prompt,
)


# ═══════════════════════════════════════════════════════════════════
# Provider 配置解析（纯函数，无 IO — 可独立单元测试）
# ═══════════════════════════════════════════════════════════════════

# 国内厂商 preset：只补 base_url + 默认 key 环境变量名，其余走 OpenAI 兼容路径
_PROVIDER_PRESETS = {
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "dashscope": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "moonshot": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
    "kimi": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
    "zhipu": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPUAI_API_KEY"),
    "glm": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPUAI_API_KEY"),
}

_BASE_PROVIDERS = {"openai", "anthropic", "claude_cli", "codex_cli"}


def resolve_provider_config(provider: str, model: str = "", base_url: str = "",
                            api_key_env: str = "", auth_token_env: str = "") -> dict:
    """把 provider 名解析为具体连接参数（纯函数，不读环境变量）。

    规则：
    - 国内 preset（deepseek/qwen/kimi/glm/...）→ 补 base_url + 默认 key 环境变量名，
      显式传入的 base_url / api_key_env 优先（支持自建代理或自定义 key 变量）。
    - 未知 provider → 立即 ValueError（fail-fast，避免静默连到错误端点）。
    - model 原样透传（不替用户决定型号）。
    """
    label = provider
    if provider in _PROVIDER_PRESETS:
        preset_url, preset_env = _PROVIDER_PRESETS[provider]
        base_url = base_url or preset_url
        api_key_env = api_key_env or preset_env
    elif provider not in _BASE_PROVIDERS:
        known = sorted(_PROVIDER_PRESETS) + sorted(_BASE_PROVIDERS)
        raise ValueError(
            f"Unknown provider: {provider!r} — expected one of {', '.join(known)}"
        )
    return {
        "provider_label": label,
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "auth_token_env": auth_token_env,
    }


# ═══════════════════════════════════════════════════════════════════
# 4. ResearchGraph — 完整复刻 ResearchLoop 全部功能
# ═══════════════════════════════════════════════════════════════════

class ResearchGraph:
    """节点式科研循环。控制流用 StateGraph，其他组件全部保留原版。"""

    def _seed_workspace_templates(self) -> None:
        """预置训练模板到 workspace(幂等,不覆盖已有文件)。

        冒烟实测暴露:agent 自写 train.py 常缺 log_metrics/checkpoint 契约,
        干跑通过但真实启动被拒,返工烧光 max_turns;且 Windows 无 cp 命令,
        agent 用 run_shell 复制模板必然失败。预置模板副本后,agent 只需
        编辑 TODO 区域,结构校验天然通过。
        崩溃恢复/续跑时文件已存在(可能已被 agent 修改)→ 绝不覆盖。
        """
        try:
            import shutil as _shutil
            template_src = Path(__file__).resolve().parent / "train_template.py"
            if not template_src.exists():
                return
            for name in ("train_template.py", "train.py"):
                dst = self.workspace / name
                if not dst.exists():
                    _shutil.copy2(template_src, dst)
                    logger.info("模板种子: workspace/%s (模板副本,编辑 TODO 区域)", name)
        except OSError as exc:
            logger.warning("模板种子写入失败(不阻塞启动): %s", exc)

    def __init__(self, config: dict, project_dir: str):
        self.config = config
        self.project_dir = Path(project_dir).resolve()
        self.workspace = self.project_dir / config.get("project", {}).get("workspace", "workspace")
        self.workspace.mkdir(exist_ok=True)
        self._seed_workspace_templates()
        self.state_path = self.workspace / "state.json"

        # ── 执行后端 ──
        self.execution_backend = build_execution_backend(
            config=config, controller_workspace=self.workspace
        )
        self.execution_backend.validate()

        # ── 训练解释器绑定（环境一致性的事实源）──
        # 干跑和训练必须用同一个 python。四层解析：config 指定 → 探测本机
        # 现成环境 → 项目 venv。**构造阶段不触发创建**（auto_create=False,
        # 避免解析环境就触发 2GB torch 下载）;真正 launch 时的惰性重解析
        # 才允许创建。launch 命令里的裸 `python` 会被系统层替换成这个
        # 绝对路径,LLM 写错也错不了。
        self.execution_python = ensure_project_python(config, self.workspace,
                                                      auto_create=False)
        if self.execution_python:
            logger.info("训练解释器绑定: %s", self.execution_python)
        else:
            logger.warning(
                "未解析到可用训练解释器（torch+torchvision 齐全的 python）。"
                "可在 config.yaml 的 execution.python 显式指定绝对路径。")

        # ── 记忆管理（原版 MemoryManager，双写：文件 + state 字段）──
        from .memory import MemoryManager
        self.memory = MemoryManager(
            project_dir=self.project_dir,
            brief_max=config.get("memory", {}).get("brief_max_chars", 3000),
            log_max=config.get("memory", {}).get("log_max_chars", 2000),
            milestone_max=config.get("memory", {}).get("milestone_max_chars", 1200),
            max_recent=config.get("memory", {}).get("max_recent_entries", 15),
            workspace_name=config.get("project", {}).get("workspace", "workspace"),
        )

        # ── 监控器 ──
        monitor_cfg = config.get("monitor", {}) or {}
        self.monitor = ExperimentMonitor(
            poll_interval=monitor_cfg.get("poll_interval", 900),
            zero_llm=monitor_cfg.get("zero_llm", True),
            backend=self.execution_backend,
            divergence_detection=monitor_cfg.get("divergence_detection", True),
            divergence_rise_streak=monitor_cfg.get("divergence_rise_streak", 3),
        )

        # ── 工具上下文 ──
        self.sandbox = resolve_sandbox(config.get("sandbox", {}))
        self.approval = ApprovalGate(self.workspace, config.get("approval", {}))
        set_tool_context(self.workspace, self.execution_backend, self.memory,
                         python_exe=self.execution_python, config=config,
                         sandbox=self.sandbox, approval=self.approval)

        # ── LLM（LangChain ChatOpenAI，支持 OpenAI 兼容 API + 国内 preset）──
        agent_cfg = config.get("agent", {}) or {}
        load_dotenv(self.project_dir / ".env")
        load_dotenv()

        # 纯函数解析 provider（preset 补 base_url/key 环境变量名，未知即报错）
        resolved = resolve_provider_config(
            provider=agent_cfg.get("provider", "openai"),
            model=agent_cfg.get("model", "qwen-plus"),
            base_url=agent_cfg.get("base_url", ""),
            api_key_env=agent_cfg.get("api_key_env", ""),
            auth_token_env=agent_cfg.get("auth_token_env", ""),
        )
        base_url = resolved["base_url"]
        api_key_env = resolved["api_key_env"]
        self.provider_label = resolved["provider_label"]

        # 离线回归开关:agent.allow_missing_key=true 时,无 API key 也允许构造
        # (langchain-openai 对空 key 直接抛错)。此时任何真实调用都会失败,
        # 由 retry/降级兜底 —— 供 ScriptedLLM 确定性回归 / 离线测试使用。
        self._allow_missing_key = bool(agent_cfg.get("allow_missing_key", False))
        api_key = os.getenv(api_key_env.strip() or "OPENAI_API_KEY")
        if not api_key and self._allow_missing_key:
            api_key = "missing-key-placeholder"

        self.llm = ChatOpenAI(
            model=resolved["model"],
            base_url=base_url or None,
            api_key=api_key,
            temperature=0,
        )

        # ── 分层模型路由 ──
        tier = agent_cfg.get("tiered_models", {}) or {}
        self._llm_think = self._make_llm(tier.get("think"), base_url, api_key_env) or self.llm
        self._llm_reflect = self._make_llm(tier.get("reflect"), base_url, api_key_env) or self.llm
        self._llm_worker = self._make_llm(tier.get("worker"), base_url, api_key_env) or self.llm
        # 降级 fallback 模型(可选:agent.fallback_model):主模型全失败后接管,
        # 配合 _safe_llm_call 的熔断统计,形成"主 → 备 → 结构化降级"三级链
        self._fallback_llm = self._make_llm(
            agent_cfg.get("fallback_model"), base_url, api_key_env)
        self._fallback_failures = 0

        # ── v2 模块（全部保留）──
        self._ledger_cfg = config.get("ledger", {}) or {}
        self._stagnation_cfg = config.get("stagnation", {}) or {}
        self._direction_forced = False  # 创新度约束:已强制换方向标记(见 _maybe_force_direction_switch)
        self._last_innovation = ("", "")  # 上轮 reflect 创新度评价 (verdict, reason)
        self._low_innovation_streak = 0  # 连续无创新点轮数(创新度门信号 2)
        self._current_directive = ""  # 创新度门指令保护(think_node 每轮写入)
        self._journal_cfg = config.get("journal", {}) or {}
        self._safety_cfg = config.get("safety", {}) or {}
        self._gates_cfg = config.get("gates", {}) or {}

        self.ledger = (
            ExperimentLedger(self.workspace)
            if self._ledger_cfg.get("enabled", True) else None
        )
        self.journal = (
            ResearchJournal(self.workspace, max_chars=self._journal_cfg.get("max_chars", 4000))
            if self._journal_cfg.get("enabled", True) else None
        )
        self.obsidian = ObsidianExporter(
            config=config, project_dir=self.project_dir, backend=self.execution_backend
        )
        self.audit = AuditLogger(self.workspace)
        self.user_profile = UserProfileStore(self.workspace).load()
        self.cost_tracker = CostTracker(self.workspace, config.get("cost", {}))
        # 预算封顶（cost.daily_budget > 0 时启用;0 = 不限制）
        self._daily_budget = float(config.get("cost", {}).get("daily_budget", 0) or 0)

        # ── 状态变量 ──
        self.max_cycles = agent_cfg.get("max_cycles", -1)
        self.cooldown = agent_cfg.get("cooldown_interval", 300)
        self.no_progress_fallback_threshold = agent_cfg.get("no_progress_fallback_threshold", 3)
        self.max_cycles_per_hour = agent_cfg.get("max_cycles_per_hour", 0)
        # 工具循环熔断:连续 N 次同工具同参数 → 中断该轮(0 = 关闭)
        self._tool_loop_fuse = int(agent_cfg.get("tool_loop_fuse", 3))
        # worker 上下文预算(可配置;超预算触发三级压缩)
        self._worker_max_context_tokens = int(
            agent_cfg.get("worker_max_context_tokens", 8000))
        self._no_progress_streak = 0
        self._last_no_progress_signature = ""
        self._cycle_times_path = self.workspace / ".cycle_times"
        self._cycle_counter_path = self.workspace / ".cycle_counter"
        self._running = True

        # ── 长期记忆（LangGraph Store，4 层分类，SQLite 持久化）──
        # 对齐 LangGraph 生产铁律：绝不使用 InMemoryStore 上生产。
        # SqliteStore 实现 BaseStore 接口，重启不丢数据。
        # 降级不是静默的：_store_degraded 显式暴露，写入失败可见（不假装成功）。
        self._store_degraded = False
        try:
            self.store = SqliteStore(self.workspace / "langgraph_store.db")
            self.store.setup()
            logger.info("Store: SqliteStore (persistent) — %s", self.workspace / "langgraph_store.db")
        except Exception as exc:
            self._store_degraded = True
            logger.error(
                "SqliteStore init failed (%s) — 持久化记忆降级为 InMemoryStore，"
                "本轮数据重启后将丢失。请检查磁盘权限/损坏。", exc)
            self.store = InMemoryStore()
        # 项目名可能含 '.' 等 BaseStore 拒绝的字符 → sanitize 后作为 namespace 段
        # （cross_store 的 project 标识仍用原始名，互不影响）
        self._project_name = self.project_dir.name
        self._store_project = re.sub(r"[^A-Za-z0-9_\-]", "_", self._project_name) or "project"

        # ── 跨项目语义记忆（SQLite + sentence-transformers）──
        self.cross_store = CrossProjectStore(self.workspace / "memory.db")

        # ── 假设生命周期状态机(G1):提出→验证→证实/否证,防重复实验 ──
        from .hypotheses import HypothesisStore
        self.hypotheses = HypothesisStore(self.workspace / "hypotheses.db")

        # ── 论文/文档知识库 RAG（复用 CrossProjectStore，namespace="rag"）──
        self.rag = RagKnowledgeBase(
            self.cross_store,
            project=f"rag_{self._project_name[:40]}",
        )
        self._rag_enabled = bool(self.config.get("rag", {}).get("enabled", True))
        if self._rag_enabled:
            try:
                self._ingest_user_literature()
            except Exception as exc:
                logger.warning(f"RAG literature ingestion failed: {exc}")

        # ── 回退系统：持久化 checkpoint + workspace 快照 ──
        self.checkpointer = SqliteCheckpointer(self.workspace / "checkpoints.db")
        self.snapshotter = WorkspaceSnapshot(self.workspace, keep=10)
        self.rollback_handler = RollbackHandler(
            self.workspace, self.snapshotter, self._cycle_counter_path
        )

        # ── 生产级容错组件 ──
        self.bad_case_collector = BadCaseCollector(self.workspace)
        self.stream_guard = StreamOutputGuard()

        # ── 跨进程事件日志（dashboard 观测通道，append-only）──
        # 注意：变量名用 event_log，避免覆盖 self.journal（ResearchJournal）。
        # run_id 含启动时间戳，区分多轮运行；同 run 续写 seq 单调
        self.event_log = EventJournal(self.workspace / "events.jsonl")
        self._run_id = f"{self._project_name}@{time.strftime('%Y%m%d_%H%M%S')}"
        self.event_log.start(self._run_id)
        self._journal_enabled = bool(self.config.get("journal", {}).get("enabled", True))

        # ── Agent Eval 录制（config eval.enabled 时启用）──
        # 录制决策轨迹到 workspace/eval/recording.jsonl，
        # 离线用 core.eval.evaluate_recording 对照 golden dataset 出报告
        if self.config.get("eval", {}).get("enabled", False):
            self._recorder = AgentRecorder(self.workspace / "eval" / "recording.jsonl")
        else:
            self._recorder = None

        # ── Leader 对话历史（替代 AgentDispatcher._leader_history）──
        self._leader_history: list = []
        self._last_plan: str = ""   # 最近一次修订后的 plan（供 episodic 持久化）
        self._review_reject_streak: int = 0  # review 连续拒绝计数（熔断用）
        # retry 熔断：LLM 故障/解析失败 → 临时跳过（retry），连续达上限才 finish
        self._retry_streak: int = 0
        self._retry_limit: int = int(config.get("agent", {}).get("retry_limit", 3))

        # ── 信号处理 ──
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        # ── 任务 ──
        self._task_content = ""
        brief_path = self.project_dir / "PROJECT_BRIEF.md"
        if brief_path.exists():
            self._task_content = brief_path.read_text(encoding="utf-8")[:3000]

        # ── 实例锁（防两个 agent 实例并发跑同一项目 → 同时 launch 多个训练）──
        self._lock_path = self.workspace / ".agent.lock"
        self._acquire_agent_lock()
        self._check_orphan_training()

    def _acquire_agent_lock(self):
        """实例锁：同项目只允许一个 agent 实例(原子创建)。

        锁文件记录 pid；已存在且 pid 存活 → 拒绝启动(防双实例并发
        launch 导致多个训练同时占 GPU)。
        O_CREAT|O_EXCL 原子创建消除 TOCTOU:两个实例同时启动时
        只有一个能创建成功,另一个必然看到锁并检查持有者。
        """
        import os as _os
        try:
            fd = _os.open(str(self._lock_path),
                          _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY, 0o644)
            try:
                _os.write(fd, str(_os.getpid()).encode())
            finally:
                _os.close(fd)
            logger.info("Agent lock acquired: %s", self._lock_path)
        except FileExistsError:
            # 锁已存在 → 判断持有者是否存活(存活探测统一走 pid_alive:
            # Windows 上裸 os.kill(pid, 0) 会误报/抛 SystemError)
            try:
                old_pid = int(self._lock_path.read_text(encoding="utf-8").strip())
                alive = pid_alive(old_pid)
            except (ValueError, OSError):
                alive = False  # 锁文件残留(内容非法)→ 可接管
            if alive:
                raise RuntimeError(
                    f"另一个 agent 实例正在运行 (pid={old_pid})。"
                    f"请先停止它，或确认无残留后删除 {self._lock_path}")
            # 持有者已死(残留锁)→ 接管
            try:
                self._lock_path.write_text(str(_os.getpid()), encoding="utf-8")
                logger.info("Agent lock taken over (stale): %s", self._lock_path)
            except OSError as exc:
                logger.warning("Agent lock takeover failed (non-fatal): %s", exc)
        except OSError as exc:
            logger.warning("Agent lock write failed (non-fatal): %s", exc)

    def _release_agent_lock(self):
        """释放实例锁（仅当锁归本进程所有）。"""
        try:
            if self._lock_path.exists():
                pid = int(self._lock_path.read_text(encoding="utf-8").strip())
                if pid == os.getpid():
                    self._lock_path.unlink()
        except (OSError, ValueError):
            pass

    def _check_orphan_training(self):
        """孤儿训练检测：上次 launch 的训练进程是否还在跑（崩溃残留）。

        仅在本地后端下可探测（SSH/Slurm 的 pid 是远端/slurm job id）。
        检测到 → 警告（不自动 kill——可能是用户手动启动的训练）。
        """
        try:
            last = self.workspace / ".last_launch.json"
            if not last.exists():
                return
            data = json.loads(last.read_text(encoding="utf-8"))
            pid = int(data.get("pid", 0))
            if pid <= 0:
                return
            from .execution import LocalExecutionBackend
            if not isinstance(self.monitor.backend, LocalExecutionBackend):
                return
            # 存活探测统一走 pid_alive（Windows 上 os.kill 会误报/SystemError）
            alive = pid_alive(pid)
            if alive:
                logger.warning(
                    "检测到上次运行残留的训练进程 pid=%d（log=%s）——"
                    "请确认它是否应继续运行；若不需要请手动终止，"
                    "避免与新一轮训练冲突",
                    pid, data.get("log_file", "?"))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    def _make_llm(self, model: str, base_url: str, api_key_env: str):
        """创建分层模型实例。如果 model 为空则返回 None（fallback 到 self.llm）。"""
        if not model:
            return None
        api_key = os.getenv(api_key_env.strip() or "OPENAI_API_KEY")
        if not api_key and getattr(self, "_allow_missing_key", False):
            api_key = "missing-key-placeholder"
        return ChatOpenAI(
            model=model,
            base_url=base_url or None,
            api_key=api_key,
            temperature=0,
        )

    def _safe_llm_call(self, llm, system: str, messages: list, *,
                        actor: str = "", action: str = "",
                        max_retries: int = 2) -> tuple:
        """
        带降级/重试/输出护栏的 LLM 调用。

        瞬态错误（429/529/timeout）→ 指数退避重试
        所有重试失败 → 返回降级 JSON（不抛异常）
        成功 → OutputGuard 扫描 + BadCase 记录违规

        Returns
        -------
        (response_text: str, degraded: bool, token_usage: dict)
        """
        last_error = ""
        for attempt in range(1 + max_retries):
            try:
                response = llm.invoke(messages)
                text = str(response.content)
                usage = getattr(response, "response_metadata", {}).get("token_usage", {})

                # 输出护栏扫描
                is_safe, violations = OutputGuard.validate(text)
                if not is_safe:
                    logger.warning("[%s/%s] Output guard violations: %s", actor, action, violations)
                    self.bad_case_collector.record(
                        stage="output", rule="output_guard",
                        content_snippet=text[:200],
                        model=getattr(llm, 'model_name', 'unknown'),
                        action="flagged",
                        metadata={"violations": violations, "actor": actor, "action": action},
                    )
                    # 标记为 degraded（有安全违规）
                    return scan_full_output(text)[0], True, usage

                return text, False, usage

            except Exception as exc:
                last_error = str(exc)
                msg_lower = last_error.lower()
                is_transient = any(kw in msg_lower for kw in (
                    "rate_limit", "429", "too many requests", "quota",
                    "overloaded", "529", "server error", "timeout",
                    "connection", "reset", "temporarily",
                ))
                model_label = getattr(llm, 'model_name', 'unknown')

                if is_transient and attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "[%s/%s] %s transient error (attempt %d/%d), retrying in %ds: %.100s",
                        actor, action, model_label, attempt + 1, max_retries + 1,
                        wait, last_error[:100],
                    )
                    time.sleep(wait)
                    continue

                logger.error(
                    "[%s/%s] %s failed after %d attempts: %.150s",
                    actor, action, model_label, attempt + 1, last_error[:150],
                )
                break

        # 主模型全失败 → 尝试 fallback 模型(可选配置 agent.fallback_model)。
        # 三级降级链:主模型 → fallback → 结构化降级 JSON(绝不抛异常)。
        fallback_llm = getattr(self, "_fallback_llm", None)
        if fallback_llm is not None:
            try:
                fb_response = fallback_llm.invoke(messages)
                fb_text = str(fb_response.content)
                fb_usage = getattr(fb_response, "response_metadata", {}).get("token_usage", {})
                try:
                    self.cost_tracker.record_call(
                        model=getattr(fallback_llm, "model_name", "fallback"),
                        input_tokens=fb_usage.get("prompt_tokens", 0),
                        output_tokens=fb_usage.get("completion_tokens", 0),
                        actor=actor, action=f"{action}:fallback",
                    )
                except Exception:
                    pass
                try:
                    self._emit_event("llm_fallback",
                                     payload={"actor": actor, "action": action})
                except Exception:
                    pass
                logger.warning(
                    "[%s/%s] fallback model used after %d attempts",
                    actor, action, max_retries + 1,
                )
                return fb_text, True, fb_usage
            except Exception as fb_exc:
                getattr(self, "_fallback_failures", 0)
                try:
                    self._fallback_failures += 1
                except Exception:
                    pass
                logger.error(
                    "[%s/%s] fallback also failed: %.150s",
                    actor, action, str(fb_exc)[:150],
                )

        # 所有重试失败 → 结构化降级
        self.bad_case_collector.record(
            stage="llm_call", rule="all_retries_failed",
            content_snippet=last_error[:300],
            model=getattr(llm, 'model_name', 'unknown'),
            action="degraded",
            metadata={"actor": actor, "action": action, "error": last_error[:200]},
        )

        degraded_response = json.dumps({
            "error": "llm_unavailable",
            "message": f"LLM call failed after {max_retries + 1} attempts",
            "last_error": last_error[:200],
            "actor": actor,
            "action": "retry",
            "reason": "LLM service temporarily unavailable, retrying next cycle",
        }, ensure_ascii=False)
        return degraded_response, True, {}

    def _budget_verdict(self) -> str:
        """预算状态判定:''(未启用/正常) / 'warning'(已用 80%) / 'exceeded'(已超限)。"""
        if self._daily_budget <= 0:
            return ""
        alert, message = self.cost_tracker.budget_alert(self._daily_budget)
        if alert:
            return "exceeded" if "exceeded" in message else "warning"
        return ""

    # ═══════════════════════════════════════════════════════════════
    # supervisor_node
    # ═══════════════════════════════════════════════════════════════

    def supervisor_node(self, state: ResearchState) -> ResearchState:
        self._emit_event("node_start", phase="supervisor")
        """纯规则路由（不再调用 LLM）。决策真相源 = _deterministic_next。

        合并方案：leader（think）承担唯一 LLM 决策，supervisor 只是机械转发。
        消除"LLM 反复选 think"导致的死循环，且省掉每次路由的 API 调用。
        """
        print("\n[supervisor] 规则路由")
        self._update_state({"phase": "supervisor", "ts": time.time()})
        cycle = state.get("cycle", 0)

        decision = self._deterministic_next(state)

        # think 的 next_stage 作为"下一阶段"提示，但只用于 think 刚执行完、
        # 且规则路由也指向 think 起点时的首次决策（避免覆盖 execute/monitor 后的推进）。
        # 关键：不能用 next_stage 无条件覆盖规则 —— 那会导致 idea_agent 死循环
        # （think 反复说 execute，execute 完成后规则应走 reflect 却被覆盖回 execute）。
        think_raw = state.get("think_result", "")
        if think_raw:
            think = _safe_json(think_raw)
            ns = think.get("next_stage")
            if ns in ("execute", "monitor", "reflect", "finish"):
                # 只有规则说"该 think"（新一轮起点）时，才让 think 的 next_stage 生效；
                # 且要求 execute_result 为空——若已有执行记录（如 monitor 刚结束），
                # think 的 next_stage 是旧建议，不允许覆盖路由，否则 monitor→execute 死循环。
                if decision == "think" and not state.get("execute_result", ""):
                    decision = ns
                    print(f"  (think 指定 next_stage={ns})")

        # max_cycles 限制：但若存在要求执行的用户指令（HUMAN_DIRECTIVE），
        # 自动多给一轮 —— 用户指令优先于轮数上限。
        directive = state.get("directive", "")
        has_exec_directive = bool(directive) and any(
            kw in directive.lower() for kw in
            ("优化", "提升", "训练", "改进", "达到", "准确率", "继续",
             "optimize", "improve", "train", "accuracy"))
        if (decision == "think" and self.max_cycles > 0
                and cycle >= self.max_cycles and not has_exec_directive):
            decision = "finish"

        print(f"  → {decision}")
        # 实时状态：supervisor 路由后立即写"下一步做什么"，UI 能即时看到
        self._update_state({"phase": "supervisor", "next": decision, "ts": time.time()})

        if decision == "finish":
            try:
                final = self.llm.invoke([
                    SystemMessage(content=SUMMARY_PROMPT),
                    HumanMessage(content=(
                        f"memory log:\n{self.memory.get_log()}"
                    )),
                ]).content
            except Exception:
                final = f"共完成 {cycle} 轮实验。"
            return {**state, "next_agent": "finish", "final_answer": str(final)}

        return {**state, "next_agent": decision}

    def _deterministic_next(self, state: ResearchState) -> str:
        think_raw = state.get("think_result", "")
        exec_raw = state.get("execute_result", "")
        refl_raw = state.get("reflect_result", "")

        if not think_raw:
            return "think"
        think = _safe_json(think_raw)
        action = think.get("action", "")
        # 语义区分（关键）：
        #   wait    — Leader 主动停止（目标达成/等人类介入）→ finish
        #   retry   — 系统异常（LLM 故障/解析失败）→ 临时跳过本轮，下轮重试
        #             有上限防死循环：连续 retry 达阈值 → finish（避免无限烧钱）
        if action == "wait":
            return "finish"
        if action == "retry":
            self._retry_streak += 1
            logger.warning(
                "think returned retry (%d/%d), skipping cycle",
                self._retry_streak, self._retry_limit)
            if self._retry_streak >= self._retry_limit:
                logger.error("retry limit reached; finishing")
                return "finish"
            return "think"
        self._retry_streak = 0  # 正常决策 → 重置
        if not exec_raw:
            return "execute"
        exec_data = _safe_json(exec_raw)
        if exec_data.get("experiment_launched"):
            if "training_logs" not in exec_data and "experiment_status" not in exec_data:
                return "monitor"
        if not refl_raw:
            return "reflect"
        return "think"

    # ═══════════════════════════════════════════════════════════════
    # think_node — Leader THINK（带对话历史）
    # ═══════════════════════════════════════════════════════════════

    def think_node(self, state: ResearchState) -> ResearchState:
        print("\n[think_node] Leader THINK...")
        self._update_state({"phase": "think", "ts": time.time()})
        self._emit_event("node_start", phase="think",
                         payload={"cycle": state.get("cycle", 0)})

        # 限速检查
        self._throttle_if_needed()
        if not self._running:
            return {**state, "next_agent": "finish", "final_answer": "收到退出信号。"}

        directive = state.get("directive", "") or self._consume_directive()
        self._current_directive = directive  # 创新度门的指令保护读取

        # 拼装 context（含 v2 信号）
        context = {
            "brief": state.get("task", ""),
            "memory_log": self.memory.get_log(),
            "cycle": state.get("cycle", 0),
            "directive": directive,
        }

        # ── 预算告警（可选：cost.daily_budget > 0 时启用）──
        # 超限 → 本轮终止（封顶模式,不再发起新实验）;80% → 警告注入 context
        budget_verdict = self._budget_verdict()
        if budget_verdict == "exceeded":
            alert_msg = f"今日预算已超限 (${self.cost_tracker.project_total():.2f})"
            print(f"[budget] {alert_msg} — 停止新实验")
            self._emit_event("budget_exceeded", payload={"message": alert_msg})
            return {**state, "next_agent": "finish",
                    "final_answer": f"{alert_msg}。可提高 config.cost.daily_budget 后重试。"}
        if budget_verdict == "warning":
            context["💰 Budget"] = (
                "今日预算已用 80%+,优先小实验,避免昂贵探索。"
            )

        # G1:假设状态注入 —— 决策只能选"待验证"假设,已否证的不再提出
        try:
            hyp_text = self.hypotheses.to_context()
            if hyp_text:
                context["🧪 Hypotheses"] = hyp_text
        except Exception as exc:
            logger.warning(f"hypothesis context failed: {exc}")

        # G3:崩溃恢复上下文(一次性) —— 上一轮训练崩溃的事实 + 续训建议
        try:
            crash_path = self.workspace / ".crash_context.json"
            if crash_path.exists():
                crash = json.loads(crash_path.read_text(encoding="utf-8"))
                crash_path.unlink()  # 一次性:读取后删除
                context["🚨 Crash Context"] = json.dumps(
                    crash, ensure_ascii=False, indent=1)
        except Exception as exc:
            logger.warning(f"crash context read failed: {exc}")

        self._enrich_context(context)

        # ── Recall-before-reason：从 Store 检索相关记忆 ──
        store_context = self._recall_from_store(state)
        context.update(store_context)

        # ── 多步计划注入（Plan-then-Execute with Replan）──
        plan = self._parse_plan(state.get("plan", ""))
        if plan:
            plan_view = self._format_plan(plan)
            context["📋 Experiment Plan"] = plan_view
            # 提示 Leader：有计划 → 引用下一个 pending 步骤（轻量推进），
            # 仅在计划完成/失败时才重新规划
            context["Plan Directive"] = (
                "当前实验计划存在。若计划未完成：引用下一个 pending 步骤作为本轮的 "
                "task（不要重新发明方向）；若全部 done/failed：制定新计划。"
                "注意：若 HUMAN_DIRECTIVE 存在，用户指令优先于计划。"
            )

        # ── 结构化 prompt（PromptBuilder：XML 隔离 + 优先级 + 折叠）──
        pb = PromptBuilder()
        pb.add_section("identity",
                       "你是 Leader agent，分析研究进展并制定下一步实验计划。",
                       priority="critical")
        if directive:
            pb.add_section("human_directive", directive,
                           priority="critical")   # 用户指令最高优先，永不折叠
        pb.add_section("project_brief", context["brief"], priority="critical")
        pb.add_section("memory_log", context["memory_log"], priority="high")
        signal_text = "\n\n".join(
            f"## {k}\n{v}" for k, v in context.items()
            if k in ("Active Violations", "Phase Gate", "Progress Signal",
                     "Recent Experiments", "Dead Ends (do NOT retry these)",
                     "Durable Insights", "💰 Budget", "🧪 Hypotheses",
                     "📖 Related Knowledge (Store)", "📜 Recent Episodes (Store)",
                     "📋 Experiment Plan", "Plan Directive"))
        if signal_text:
            pb.add_section("state_signals", signal_text,
                           priority="medium", collapse_when=4000)
        pb.add_section("cycle", f"当前 cycle: {context['cycle']}", priority="low")
        prompt = pb.build(max_tokens=3000)

        try:
            # 用户画像注入
            profile_prompt = self.user_profile.to_prompt()
            messages = list(self._leader_history)
            messages.append(HumanMessage(content=profile_prompt + "\n\n" + prompt))
            think_llm = self._llm_think or self.llm
            # ── 结构化输出（对齐 OpenAI with_structured_output）──
            # 若 pydantic 可用，用 LeaderDecision 约束 LLM 输出 JSON schema，
            # 消除正则模糊解析风险。不可用时降级到文本解析。
            structured_llm = None
            if LeaderDecision is not None:
                try:
                    structured_llm = think_llm.with_structured_output(LeaderDecision)
                except Exception:
                    logger.debug("with_structured_output not supported by this model, falling back to text")
            # 输入护栏
            safe, _, sanitized_prompt = InputGuard.validate(messages[-1].content)
            if not safe:
                messages[-1] = HumanMessage(content=sanitized_prompt)
            if structured_llm is not None:
                # 结构化路径：LLM → LeaderDecision 对象 → 字典
                try:
                    decision: LeaderDecision = structured_llm.invoke([  # type: ignore[no-untyped-call]
                        SystemMessage(content=LEADER_THINK_PROMPT),
                        *messages,
                    ])
                    raw = decision.model_dump_json()
                    degraded = False
                    usage = {}  # 结构化输出暂不返回 token usage
                    logger.debug("[think_node] structured output: %s", decision.action)
                    # 构造兼容 __raw_response 以便审计/日志
                    if not hasattr(decision, '__raw_response'):
                        pass  # structured output 不保留原始 response metadata
                except Exception as exc:
                    logger.warning(
                        "structured output failed (%s), falling back to text mode",
                        str(exc)[:100],
                    )
                    raw, degraded, usage = self._safe_llm_call(
                        think_llm,
                        system=LEADER_THINK_PROMPT,
                        messages=[SystemMessage(content=LEADER_THINK_PROMPT), *messages],
                        actor="leader", action="think",
                    )
            else:
                raw, degraded, usage = self._safe_llm_call(
                    think_llm,
                    system=LEADER_THINK_PROMPT,
                    messages=[SystemMessage(content=LEADER_THINK_PROMPT), *messages],
                    actor="leader", action="think",
                )
            # 费用追踪
            if usage:
                self.cost_tracker.record_call(
                    model=think_llm.model_name,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    actor="leader", action="think"
                )
            self._leader_history = messages + [type(messages[-1])(content=str(raw))]
            result = _parse_json_response(str(raw))
            self.audit.record(actor="leader", action="llm:think",
                            target=think_llm.model_name,
                            result="success" if (result.get("action") != "wait" and not degraded) else ("degraded" if degraded else "wait"))
        except Exception as exc:
            logger.error(f"THINK 失败: {exc}")
            result = {"action": "retry", "reason": f"THINK error, retrying next cycle: {str(exc)[:200]}"}
            self.audit.record(actor="leader", action="llm:think", result="failed",
                            detail={"error": str(exc)[:200]})

        # 反卡死
        result = self._apply_no_progress_fallback(result, directive)

        # ── 硬约束：Human Directive 必须被强制执行（代码级，非 prompt 软提示）──
        # 若用户指令含"优化/提升/训练/改进"等执行语义，禁止 think 选择
        # writing/report（用户要的是继续干，不是写总结）。
        if directive:
            d = directive.lower()
            wants_execution = any(kw in d for kw in
                                  ("优化", "提升", "训练", "改进", "达到", "准确率",
                                   "accuracy", "optimize", "improve", "train"))
            if wants_execution and result.get("agent") in ("writing", "report"):
                logger.warning(
                    "Directive requires execution but think chose %s; forcing to code",
                    result.get("agent"))
                result["agent"] = "code"
                result["action"] = "experiment"
                result["task"] = f"(USER DIRECTIVE PRIORITY) {directive}\n" \
                                 f"Original plan: {result.get('task', '')}"

        # ── 创新度硬约束:方向多样性(代码级,零 LLM)──
        # 用户审查:think 沿决策树机械叠加「加层/dropout/lr」,全部落在同一
        # 方向维度 —— 有停滞检测(advisory)但无方向多样性约束,开放式任务
        # 会困在局部。命中「最近 3 轮同维度 + 指标停滞」→ 强制换维度;
        # 强制过一次仍同维度 → 升级 idea agent(开放式探索)。
        result = self._maybe_force_direction_switch(result)

        # ── Eval 录制：Leader think 决策轨迹 ──
        if self._recorder is not None:
            try:
                self._recorder.record_llm(
                    "leader", "think",
                    prompt_snippet=prompt, output_snippet=str(raw),
                    chosen_action=str(result.get("action", "")),
                    chosen_agent=str(result.get("agent", "")),
                    cycle=state.get("cycle", 0),
                )
            except Exception:
                pass

        print(f"  action={result.get('action')}, agent={result.get('agent', 'N/A')}")

        # ── Plan 合并：首次生成（Leader 返回 plan 且当前无 plan）→ 采纳；否则沿用 ──
        new_plan = self._merge_plan(state.get("plan", ""), result.get("plan", []))

        # G4 计划评审:与账本最近实验明显重复的任务 → 打回(不重复执行已做过的)
        if result.get("action") == "experiment":
            blocked = self._plan_duplicate_check(result, new_plan)
            if blocked:
                logger.warning("[G4] plan blocked: %s", blocked)
                result["action"] = "wait"
                result["reason"] = blocked
                result["plan"] = []
                new_plan = ""

        # 实时状态：把最新 plan 写入 state.json，UI 能显示"当前计划"
        self._update_state({
            "phase": "plan",
            "plan_task": str(result.get("task", ""))[:200],
            "plan_hypothesis": str(result.get("hypothesis", ""))[:200],
            "plan_agent": result.get("agent", ""),
            "ts": time.time(),
        })
        return {
            **state,
            "directive": directive,
            "plan": new_plan,
            "think_result": json.dumps(result, ensure_ascii=False),
            # 新轮次起点：清空上一轮的 execute/reflect 结果。
            # reflect 的 return 已清 think/execute（见 reflect_node），但 think 的
            # return 没清 reflect_result —— 残留会让 supervisor 在 monitor 结束后
            # 误判「本轮已反思」跳过 reflect，再被 next_stage 覆盖打回 execute，
            # 形成 monitor→execute 无限重试（真实事故，2026-08-13 实测）。
            "execute_result": "",
            "reflect_result": "",
        }

    # ═══════════════════════════════════════════════════════════════
    # execute_node — Worker EXECUTE（create_agent 工具调用）
    # ═══════════════════════════════════════════════════════════════

    def execute_node(self, state: ResearchState) -> ResearchState:
        print("\n[execute_node] Worker EXECUTE...")
        self._update_state({"phase": "execute", "ts": time.time()})
        think = _safe_json(state.get("think_result", "{}"))
        self._emit_event("node_start", phase="execute",
                         payload={"agent": think.get("agent", ""),
                                  "cycle": state.get("cycle", 0)})
        think = _safe_json(state.get("think_result", "{}"))
        agent_type = think.get("agent", "code")
        task_description = think.get("task", "")

        if not task_description:
            return {**state, "execute_result": json.dumps(
                {"error": "think_result 中没有 task", "experiment_launched": False}, ensure_ascii=False)}

        # ── 执行前快照（rollback 恢复点）──
        cycle = state.get("cycle", 0)
        try:
            snap_path = self.snapshotter.create(label=f"cycle{cycle}", cycle=cycle)
            self.snapshotter.cleanup()  # 同时清理旧快照
        except Exception as exc:
            logger.warning("Pre-execute snapshot failed (non-fatal): %s", exc)

        print(f"  agent={agent_type}, task={task_description[:100]}...")

        # ── HITL 审批门控 ──
        needs, reason = self.approval.needs_approval("agent_execute", {"task": task_description})
        if needs:
            req = self.approval.create_request("agent_execute", {"task": task_description})
            print(f"  [approval] pending: [{req.id}] {reason}")
            result = self.approval.wait_for_approval(req.id, poll_interval=10, timeout=600)
            self.audit.record(actor="approval_gate", action="hitl:agent_execute",
                            target=req.id, result=result)
            if result == "denied":
                return {**state, "execute_result": json.dumps(
                    {"error": f"HITL denied: {reason}", "experiment_launched": False,
                     "approval_id": req.id}, ensure_ascii=False)}

        try:
            # idea_agent 必须真的做文献调研并写 IDEA_NOTES（硬约束，非可选）。
            # 若用户提供了文献（USER_LITERATURE.md）→ 优先分析它（绕过 429 限流）；
            # 否则才调 search_papers / search_arxiv。
            extra_instr = ""
            if agent_type == "idea":
                lit_path = self.workspace / "USER_LITERATURE.md"
                if lit_path.exists():
                    lit = lit_path.read_text(encoding="utf-8", errors="replace").strip()
                    extra_instr = (
                        f"\n[MANDATORY] The user provided the literature index below; "
                        f"analyze it directly and extract actionable methods "
                        f"(do NOT call search_papers):\n{lit}\n"
                        f"Write your findings to IDEA_NOTES.md (use the write_file tool)."
                    )
                else:
                    extra_instr = (
                        "\n[MANDATORY] Call search_papers / search_arxiv to research "
                        "relevant literature, extract actionable methods, and write "
                        "findings to IDEA_NOTES.md (use write_file). If external APIs "
                        "fail, base your suggestions on domain knowledge and continue."
                    )
            # ── 单步工具循环（对齐原版 dispatch_worker，替代 create_agent 一次性 invoke）──
            # 每轮单次 LLM + 执行工具，max_turns 硬上限，上下文可控不卡死。
            task_text = (
                f"User task: {task_description}\n"
                f"Working directory: {self.workspace}\n"
                f"You are the {agent_type}_agent; use tools to complete the task."
                f"{extra_instr}\n"
                f"Tool budget: at most {WORKER_MAX_TURNS.get(agent_type, 20)} LLM "
                f"decisions (multiple tool calls per turn are allowed). Budget is "
                f"limited — act efficiently: confirm what you need, then execute; "
                f"avoid re-reading and unrelated exploration."
            )
            # 流式模式：config worker.stream_mode 开启；token 增量节流写事件日志
            # （每 30 个 token 一条，避免日志膨胀），tool 生命周期事件在循环内发。
            stream_mode = bool(self.config.get("worker", {}).get("stream_mode", False))
            token_counter = {"n": 0}

            def _on_token(text: str):
                token_counter["n"] += 1
                if token_counter["n"] % 30 == 0:
                    self._emit_event("stream_text_delta", phase="execute",
                                     payload={"agent": agent_type,
                                              "delta": text[:200]})

            result = self._run_worker_single_step(
                agent_type, task_text, system_prompt=_WORKER_PROMPTS.get(agent_type, ""),
                stream_mode=stream_mode, on_token=_on_token if stream_mode else None)

            # ── RAG 摄取：idea agent 分析文献后，把发现摄入知识库 ──
            if agent_type == "idea" and self._rag_enabled:
                try:
                    self._ingest_idea_notes()
                except Exception as exc:
                    logger.warning(f"RAG idea ingestion failed: {exc}")

            # ── Review Agent：code 未 launch 时审查训练脚本，通过才补 launch ──
            # 实测评估（2026-08）：阻塞式 review 的边际价值低于成本——
            #   1. mandatory_dry_run 已是最强防线（真实执行 > 模型读代码）
            #   2. 拉锯循环（拒绝→修改→拒绝）实测耗掉整个 cycle，训练未启动
            #   3. 模型审查质量不稳定（空响应/拒绝无理由/探索爆炸）
            # 因此默认关闭（review.enabled: false）；需要深度审查时手动开启。
            if (agent_type == "code" and not result.get("experiment_launched")
                    and self.config.get("review", {}).get("enabled", False)
                    and not result.get("error")
                    and _is_training_task(task_description)):
                result = self._review_and_launch(
                    result, task_description, state.get("cycle", 0))

            # ── Eval 录制：worker 工具循环轨迹 ──
            if self._recorder is not None:
                try:
                    self._recorder.record_worker(
                        agent_type, task_text,
                        tools_used=result.get("_tool_log", []),
                        result=result, cycle=state.get("cycle", 0),
                    )
                except Exception:
                    pass
            self.audit.record(actor=f"{agent_type}_agent", action="agent:execute",
                            target=task_description[:100],
                            result="success" if not result.get("error") else "failed")
        except Exception as exc:
            logger.error(f"EXECUTE 失败: {exc}")
            result = {"agent": agent_type, "error": str(exc)[:500], "experiment_launched": False}
            self.audit.record(actor=f"{agent_type}_agent", action="agent:execute",
                            result="failed", detail={"error": str(exc)[:200]})

        # ── B1 HITL 等待:launch 需要审批 → 零 LLM 轮询 PENDING_APPROVALS.md ──
        # 等待期间无 LLM 调用(与 monitor 同一哲学);获批 → 缓存 + 重试 launch;
        # 拒绝/超时 → 标记 skipped(truthful,不误报 completed)。
        if isinstance(result, dict) and result.get("approval_pending"):
            try:
                approval = getattr(self, "approval", None)
                if approval is not None:
                    timeout = int(self.config.get("approval", {}).get(
                        "timeout_seconds", 1800))
                    decision = approval.wait_for_approval(
                        result["approval_id"], poll_interval=10, timeout=timeout)
                    if decision == "approved":
                        approval.cache_decision(result.get("cache_key", ""), "approved")
                        # 重试 launch(缓存已批 → 直接执行)
                        from .nodes import launch_experiment as _launch
                        retry = json.loads(_launch.func(
                            command=result.get("command", ""),
                            log_file=result.get("log_file", "")))
                        result.update(retry)
                        result["approval_waited"] = True
                    else:
                        result["experiment_status"] = "skipped"
                        result["approval_denied"] = True
                        result["experiment_launched"] = False
                        logger.warning(
                            "approval %s for %s — experiment skipped",
                            decision, result.get("approval_id"))
            except Exception as exc:
                logger.warning(f"approval wait failed: {exc}")
                result["experiment_status"] = "skipped"

        return {**state, "execute_result": json.dumps(result, ensure_ascii=False)}

    def _extract_execute_result(self, agent_result: dict, agent_type: str, last_msg: str) -> dict:
        """从 worker 的 messages 里提取执行结果。

        关键：训练是否启动看 pid + log_file（launch_experiment 是唯一返回这两者的工具）。
        优先找 ToolMessage 里的 launch JSON（权威），fallback 到任何含 pid+log_file 的
        消息。不依赖 "experiment_launched" 字符串 —— agent 散文可能不提它，但 pid 是硬证据。
        """
        result = {"agent": agent_type, "response": last_msg}
        messages = agent_result.get("messages", []) if isinstance(agent_result, dict) else []

        # 先找含 pid + log_file 的消息（launch 证据）
        for msg in messages:
            content = str(getattr(msg, 'content', ''))
            if not content:
                continue
            # 尝试解析整个 content 为 JSON
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and parsed.get("pid") is not None:
                    result.update(parsed)
                    result["experiment_launched"] = True
                    return result
            except (json.JSONDecodeError, TypeError):
                pass
            # 从嵌套 JSON 提取 pid
            if '"pid"' in content:
                m = re.search(r'\{[^{}]*"pid"[^{}]*\}', content, re.DOTALL)
                if m:
                    try:
                        parsed = json.loads(m.group())
                        if parsed.get("pid") is not None:
                            result.update(parsed)
                            result["experiment_launched"] = True
                            return result
                    except json.JSONDecodeError:
                        pass

        if not result.get("experiment_launched"):
            result["experiment_launched"] = False
        return result

    # ═══════════════════════════════════════════════════════════════════
    # 上下文压缩流水线（模块级纯函数，可单元测试）
    # 对齐 Anthropic Claude Code 的多级压缩策略：
    #   工具结果截断 → 早期摘要 → 硬截断（按完整轮）
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _estimate_tokens(msgs: list) -> int:
        """粗略估算 messages 的 token 数（chars/4）。

        精确计数需要 tiktoken，但此近似在 ±15% 范围内足够用于预算控制。
        """
        total = 0
        for m in msgs:
            content = str(getattr(m, 'content', ''))
            total += len(content)
        return total // 4

    @staticmethod
    def _stream_collect_for_test(chunks_iter, on_token=None) -> "AIMessage":
        """完整迭代 stream chunks，累积文本与 tool_call 分片，返回最终 AIMessage。

        提交屏障语义：tool_call 参数分片累积后做完整 JSON 解析，
        解析失败 → 参数标记 _error，绝不执行部分参数（对齐 OpenAI
        buffer_streamed_tool_calls 思路）。
        """
        text_parts: list[str] = []
        tool_chunks: dict[tuple, dict] = {}
        for chunk in chunks_iter:
            content = chunk.content
            if isinstance(content, str):
                if content:
                    text_parts.append(content)
                    if on_token is not None:
                        on_token(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        t = str(block.get("text", ""))
                        if t:
                            text_parts.append(t)
                            if on_token is not None:
                                on_token(t)
            for tchunk in getattr(chunk, "tool_call_chunks", None) or []:
                key = (tchunk.get("index", 0), tchunk.get("id", ""))
                entry = tool_chunks.setdefault(key, {
                    "name": "", "args": "", "id": tchunk.get("id", ""),
                    "index": tchunk.get("index", 0),
                })
                if tchunk.get("name"):
                    entry["name"] = tchunk["name"]
                entry["args"] += tchunk.get("args", "")

        text = "".join(text_parts)
        if tool_chunks:
            tool_calls = []
            for _k in sorted(tool_chunks):
                e = tool_chunks[_k]
                try:
                    args = json.loads(e["args"]) if e["args"].strip() else {}
                except json.JSONDecodeError:
                    args = {"_error": "invalid JSON args, tool not executed"}
                tool_calls.append({
                    "name": e["name"], "args": args, "id": e["id"],
                    "type": "tool_call",
                })
            return AIMessage(content=text, tool_calls=tool_calls)
        return AIMessage(content=text)

    # ═══════════════════════════════════════════════════════════════════
    # 多步实验计划辅助（Plan-then-Execute with Replan，纯函数可测试）
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_plan(raw: str) -> list:
        """解析 state.plan（JSON 字符串）→ 步骤列表。容错：坏数据返回 []。"""
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(data, list):
            return []
        out = []
        for step in data:
            if isinstance(step, dict) and step.get("step_id") and step.get("status"):
                out.append(step)
        return out

    @staticmethod
    def _format_plan(plan: list) -> str:
        """渲染计划进度视图（注入 think prompt）。"""
        if not plan:
            return ""
        lines = []
        for i, step in enumerate(plan, 1):
            marker = {"pending": "⬜", "running": "🔄", "done": "✅", "failed": "❌"}.get(
                step.get("status", "pending"), "⬜")
            agent = step.get("agent", "")
            title = str(step.get("title", ""))[:80]
            result = str(step.get("result", ""))[:60]
            line = f"{i}. {marker} [{step.get('status', 'pending')}] ({agent}) {title}"
            if result:
                line += f" — {result}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _merge_plan(current_raw: str, leader_plan: list) -> str:
        """Think 后合并计划：无 plan 且 Leader 给出 → 采纳（首个步骤标 running）；
        已有 plan → 沿用（Leader 返回 [] 或当前非空）。返回 JSON 字符串。"""
        current = ResearchGraph._parse_plan(current_raw)
        if not current and leader_plan:
            steps = []
            for i, s in enumerate(leader_plan):
                steps.append({
                    "step_id": str(s.get("step_id") or f"s{i+1}"),
                    "title": str(s.get("title", "")),
                    "agent": str(s.get("agent", "code")),
                    "status": "running" if i == 0 else "pending",
                    "result": "",
                })
            return json.dumps(steps, ensure_ascii=False)
        if not current:
            return ""  # Leader 也没给 → 维持无计划
        # 已有 plan：本轮 Think 决策对应下一个 pending 步骤 → 标 running
        for step in current:
            if step.get("status") == "pending":
                step["status"] = "running"
                break
        return json.dumps(current, ensure_ascii=False)

    @staticmethod
    def _replan(plan: list, exec_data: dict, reflect: dict) -> list:
        """Reflect 后修订计划：
        1. 当前 running 步骤 → 按实验结果标 done/failed（附结果摘要）
        2. Leader 返回新 plan（重新规划）→ 整体替换
        3. 计划完成（无 pending/running）→ 清空，下轮 Think 重新规划
        """
        # Leader 显式重新规划优先
        new_plan = reflect.get("plan")
        if new_plan:
            steps = []
            for i, s in enumerate(new_plan):
                steps.append({
                    "step_id": str(s.get("step_id") or f"s{i+1}"),
                    "title": str(s.get("title", "")),
                    "agent": str(s.get("agent", "code")),
                    "status": "pending",
                    "result": "",
                })
            return steps

        if not plan:
            return []

        status = exec_data.get("experiment_status", "") or ""
        terminal = str(exec_data.get("terminal_state", ""))
        outcome = "failed" if (status == "failed" or "error" in status) else "done"
        result_note = ""
        final_metrics = exec_data.get("final_metrics") or {}
        if isinstance(final_metrics, dict) and final_metrics:
            acc = final_metrics.get("accuracy") or final_metrics.get("acc")
            loss = final_metrics.get("loss")
            if acc is not None:
                result_note = f"acc={acc}"
            elif loss is not None:
                result_note = f"loss={loss}"
        if terminal:
            result_note = (result_note + f" [{terminal}]").strip()

        revised = []
        marked = False
        for step in plan:
            s = dict(step)
            if not marked and s.get("status") == "running":
                s["status"] = outcome
                if result_note:
                    s["result"] = result_note
                marked = True
            revised.append(s)

        # 计划全部完成/失败 → 清空，下轮 Think 重新规划
        if all(s.get("status") in ("done", "failed") for s in revised):
            return []
        return revised

    @staticmethod
    def _round_safe_start(msgs: list, start_idx: int) -> int:
        """找到安全的截断起点：跳过开头的孤立 ToolMessage。

        Agent Loop 中 ToolMessage 永远紧跟其 AIMessage（tool_call），
        所以只要窗口起点不是 ToolMessage，窗口内所有 ToolMessage 都配对完整。
        否则 OpenAI API 会报 "ToolMessage with id X not found in conversation"。
        """
        i = max(0, start_idx)
        while i < len(msgs) and isinstance(msgs[i], ToolMessage):
            i += 1
        return i

    @staticmethod
    def _manage_context_window(msgs: list, agent_type: str,
                               max_tokens: int = 8000,
                               tool_result_max_chars: int = 500) -> list:
        """上下文压缩流水线（对齐 Anthropic Claude Code 多级压缩）。

        三级压缩，由轻到重：
          1. 工具结果截断 — ToolMessage 内容 > tool_result_max_chars 字符时截断
          2. 早期摘要 — 前 1/3 消息用一句话摘要替代
          3. 硬截断 — 保留 SystemMessage + 最近 8 条完整轮

        每级截断都保证 ToolMessage ↔ AIMessage 配对完整（不产生孤立 ToolMessage）。

        只在超预算时触发，未超预算则原样返回。
        """
        est = ResearchGraph._estimate_tokens(msgs)
        if est <= max_tokens:
            return msgs

        logger.info(
            "[%s] context %d tokens exceeds budget %d, compacting...",
            agent_type, est, max_tokens,
        )

        # ── 第 1 级：工具结果截断 ──
        truncated = []
        for m in msgs:
            if isinstance(m, ToolMessage):
                content = str(m.content)
                if len(content) > tool_result_max_chars:
                    truncated.append(ToolMessage(
                        content=content[:300] + f"\n... [truncated, {len(content)} total chars]",
                        tool_call_id=getattr(m, 'tool_call_id', ''),
                    ))
                else:
                    truncated.append(m)
            else:
                truncated.append(m)

        est2 = ResearchGraph._estimate_tokens(truncated)
        if est2 <= max_tokens:
            logger.info("[%s] level-1 (tool truncation): %d→%d tokens", agent_type, est, est2)
            return truncated

        # ── 第 2 级：早期摘要 → SystemMessage ──
        if len(truncated) > 6:
            # 从第 2 条消息开始的前 1/3 替换为摘要；late 起点跳过孤立 ToolMessage
            split = max(2, len(truncated) // 3)
            late_start = ResearchGraph._round_safe_start(truncated, split)
            early = truncated[1:late_start]
            late = truncated[late_start:]
            summary_parts = []
            for m in early:
                content = str(getattr(m, 'content', ''))[:100]
                if content.strip():
                    summary_parts.append(content.strip()[:80])
            if summary_parts:
                summary = "[对话早期摘要] " + " | ".join(summary_parts[:5]) + " ..."
            else:
                summary = "[对话早期摘要] (no text content)"
            result = [truncated[0]]  # SystemMessage
            result.append(HumanMessage(content=summary))
            result.extend(late)
            est3 = ResearchGraph._estimate_tokens(result)
            if est3 <= max_tokens:
                logger.info("[%s] level-2 (early summarization): %d→%d tokens", agent_type, est2, est3)
                return result
            # fall through to level 3
            truncated = result
            est2 = est3

        # ── 第 3 级：硬截断 — 保留 System + 最近 8 条完整轮 ──
        # 起点必须跳过 ToolMessage，保证与 AIMessage 配对完整
        tail_start = ResearchGraph._round_safe_start(truncated, len(truncated) - 8)
        keep = [truncated[0]] if isinstance(truncated[0], SystemMessage) else []
        keep.extend(truncated[tail_start:])
        est3 = ResearchGraph._estimate_tokens(keep)
        logger.warning(
            "[%s] level-3 (hard truncation): %d→%d tokens (%d→%d messages)",
            agent_type, est2, est3, len(truncated), len(keep),
        )
        return keep

    def _run_worker_single_step(self, agent_type: str, task_text: str,
                                system_prompt: str = "", *,
                                stream_mode: bool = False,
                                on_token=None) -> dict:
        """LangChain 原生工具循环：bind_tools + 原生 tool_calls + ToolMessage。

        每轮一次 invoke → 读原生 .tool_calls → 执行工具 → 追加 ToolMessage →
        直到 AIMessage 无 tool_calls（最终答案）。max_turns 硬上限防卡死。

        关键：必须 bind_tools（否则模型走文本协议，拿不到原生 tool_calls，
        也无法用 ToolMessage）。工具经 @tool.func 直接调用（保留安全守卫）；
        PID 从 tool_results_log 权威提取。

        流式模式（stream_mode=True，对齐 OpenAI buffer_streamed_tool_calls 思路）：
        - worker_llm.stream() 迭代 AIMessageChunk，文本增量实时回调 on_token
        - tool_call 参数分片按 (index, id) 累积，**完整 JSON 校验通过后才执行工具**，
          绝不执行部分参数（提交屏障：参数不完整 = 工具不执行）
        - 流式错误同样走分级重试；不传 stream_mode 时行为与旧版完全一致

        智能重试（对齐 Anthropic 分级退避）：
        - 瞬时错误 → 指数退避 + 随机抖动
        - 上下文溢出 → 自动缩减早期消息
        - 致命错误 → 立即中断不浪费配额
        """
        # ── 内联 _trim_context：上下文溢出时缩减早期消息 ──
        def _trim_context(msgs: list) -> list:
            """上下文溢出时：保留 SystemMessage + 截掉最早的 HumanMessage 之后的冗余，
            只保留最近 3 轮完整对话。"""
            if len(msgs) <= 3:
                return msgs
            # 取最后 6 条,但向前扩展到最近的非 ToolMessage —— 直接取 [-6:]
            # 可能从孤立 ToolMessage 开始(无配对的 AI tool_call),LangChain 会报错
            start = max(0, len(msgs) - 6)
            while start > 0 and isinstance(msgs[start], ToolMessage):
                start -= 1
            if start == 0:
                return msgs  # 全保留(截断不会更省,且避免破坏配对)
            # 保留 system（如果第一个是 SystemMessage）
            trimmed = [msgs[0]] if isinstance(msgs[0], SystemMessage) else []
            trimmed.extend(msgs[start:])
            logger.warning(
                "trimmed context from %d to %d messages (overflow mitigation)",
                len(msgs), len(trimmed),
            )
            return trimmed

        # ── Token 预算（实例可覆盖）──
        max_context_tokens = getattr(self, '_worker_max_context_tokens', 8000)

        tool_fns: dict[str, callable] = {}
        tool_schemas: dict[str, dict] = {}
        tools: list = []
        for name, fn in TOOL_FUNCTIONS.get(agent_type, []):
            tool_fns[name] = fn.func
            tools.append(fn)   # @tool 对象列表，传给 bind_tools
            try:
                tool_schemas[name] = fn.args_schema.schema()
            except Exception:
                tool_schemas[name] = {}

        max_turns = WORKER_MAX_TURNS.get(agent_type, 20)
        worker_llm = (self._llm_worker or self.llm).bind_tools(tools)  # ← 关键

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=task_text)]
        tool_results_log: list[dict] = []
        last_response = ""
        # 工具熔断跨 turn 计数(闭包列表:内层循环需要读写)
        _fuse_last_sig: list = [("", "")]
        _fuse_streak: list = [0]

        def _coerce_args(name: str, args) -> dict:
            """按 @tool 的 args_schema 做类型强转（模型可能输出字符串数字）。

            非 dict 参数(模型输出畸形)→ 返回空 dict:该轮工具不执行,
            而不是 dict(args) 抛 ValueError 杀死整个 worker 循环。
            """
            if not isinstance(args, dict):
                logger.warning("[%s] tool args not a dict: %r", agent_type, args)
                return {}
            schema = tool_schemas.get(name, {})
            props = schema.get("properties", {}) or {}
            coerced = dict(args)
            for k, v in list(coerced.items()):
                ptype = props.get(k, {}).get("type")
                if ptype == "integer" and isinstance(v, str) and v.strip().lstrip("-").isdigit():
                    coerced[k] = int(v)
                elif ptype == "number" and isinstance(v, str):
                    try:
                        coerced[k] = float(v)
                    except ValueError:
                        pass
                elif ptype == "boolean" and isinstance(v, str):
                    coerced[k] = v.lower() in ("true", "1", "yes")
            return coerced

        for turn in range(max_turns):
            # ── 上下文压缩：超过 Token 预算时自动触发（对齐 Anthropic 多级压缩流水线）──
            messages = self._manage_context_window(
                messages, agent_type, max_tokens=max_context_tokens)

            # ── 智能重试（对齐 Anthropic 分级退避策略）──
            # 瞬时错误（rate_limit/overloaded/timeout/connection）→ 指数退避重试
            # 致命错误（auth/bad_request）→ 立即中断
            # 上下文溢出 → 自动缩减早期消息 + 重试 1 次
            # 注意：on_context_overflow 必须返回「重试后的 LLM 响应」而非缩减后的
            # 输入——retry_llm_call 会直接把它作为 fn 的结果返回。
            try:
                if stream_mode:
                    # 流式：整轮收集放入重试包装，保证错误语义与 invoke 一致
                    last_msg = retry_llm_call(
                        lambda: self._stream_collect_for_test(
                            worker_llm.stream(messages), on_token),
                        max_retries=2,
                        actor=agent_type,
                        action=f"stream_turn_{turn}",
                        on_context_overflow=lambda: self._stream_collect_for_test(
                            worker_llm.stream(_trim_context(messages)), on_token),
                    )
                else:
                    last_msg = retry_llm_call(
                        lambda: worker_llm.invoke(messages),
                        max_retries=2,
                        actor=agent_type,
                        action=f"tool_loop_turn_{turn}",
                        on_context_overflow=lambda: worker_llm.invoke(
                            _trim_context(messages)),
                    )
            except FatalLLMError as exc:
                logger.error(
                    "[%s] tool_loop fatal error (turn=%d): %s",
                    agent_type, turn, str(exc)[:200],
                )
                break
            except LLMRetryError as exc:
                logger.error(
                    "[%s] tool_loop all retries exhausted (turn=%d): %s",
                    agent_type, turn, str(exc)[:200],
                )
                break
            except Exception as exc:
                logger.error(
                    "[%s] tool_loop unexpected error (turn=%d): %s",
                    agent_type, turn, str(exc)[:200],
                )
                break

            # ── 费用追踪:worker 工具循环是主要成本来源,逐轮记账 ──
            # (冒烟实测:leader 只有 6 次调用,worker 循环 80+ 次,不记账则
            #  costs.jsonl 严重低估真实成本,违背"成本逐 cycle 可审计")
            try:
                usage = (getattr(last_msg, "response_metadata", {}) or {}).get(
                    "token_usage", {}) or {}
                if usage:
                    self.cost_tracker.record_call(
                        model=getattr(worker_llm, "model_name", "worker"),
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        actor=agent_type, action="tool_loop",
                    )
            except Exception:
                pass

            tool_calls = getattr(last_msg, "tool_calls", None) or []
            if not tool_calls:
                last_response = str(last_msg.content)  # 无工具 → 最终答案
                break

            messages.append(last_msg)
            # 工具循环熔断:连续 N 次同工具+同参数 → 中断该轮并提示
            # (防 LLM 卡死在同一个调用上反复烧 token;launch 幂等门只覆盖 launch)
            fuse = getattr(self, "_tool_loop_fuse", 3)
            fuse_triggered = False
            for call in tool_calls:
                name = call.get("name", "")
                args = _coerce_args(name, call.get("args", {}) or {})
                cid = call.get("id", "")
                sig = (name, json.dumps(args, sort_keys=True, ensure_ascii=False)[:200])
                if sig == _fuse_last_sig[0]:
                    _fuse_streak[0] += 1
                else:
                    _fuse_streak[0] = 1
                    _fuse_last_sig[0] = sig
                if fuse > 0 and _fuse_streak[0] >= fuse:
                    tool_output = json.dumps({
                        "error": f"called the same tool {name} with identical arguments {_fuse_streak[0]} times in a row. "
                                 "Stop repeating; change strategy or summarize and finish."})
                    tool_results_log.append({"name": name, "args": args, "output": tool_output})
                    messages.append(ToolMessage(content=tool_output, tool_call_id=cid))
                    fuse_triggered = True
                    break
                # 提交屏障：流式分片参数解析失败（args 含 _error 标记）→ 工具不执行
                if args.get("_error"):
                    tool_output = json.dumps({"error": args["_error"]})
                elif name in tool_fns:
                    self._emit_event("tool_call", phase="execute",
                                     payload={"agent": agent_type, "tool": name,
                                              "args": str(args)[:200]})
                    try:
                        tool_output = tool_fns[name](**args)
                    except Exception as exc:
                        tool_output = json.dumps({"error": f"tool {name} failed: {str(exc)[:200]}"})
                else:
                    tool_output = json.dumps({"error": f"unknown tool: {name}"})
                tool_results_log.append({"name": name, "args": args, "output": tool_output})
                self._emit_event("tool_result", phase="execute",
                                 payload={"agent": agent_type, "tool": name,
                                          "output": tool_output[:200]})
                # 用 ToolMessage 结构化回喂（tool_call_id 关联原生调用）
                # 工具结果即时代总结（对齐 Claude Code tool summary 思路）：
                # 超长输出在回灌前截断——保留头部关键信息 + 尾部行 + 截断提示，
                # 避免大段内容膨胀上下文（也减少模型"忘了读过什么"重复读取）
                # read_file 例外:整文件可见,防 agent 用 shell 转储文件
                limit = (_READ_FILE_SUMMARY_CHARS
                         if name == "read_file" else _TOOL_RESULT_SUMMARY_CHARS)
                if len(tool_output) > limit:
                    tool_output = _summarize_tool_output(
                        tool_output, name, head_chars=limit)
                messages.append(ToolMessage(content=tool_output, tool_call_id=cid))
            if fuse_triggered:
                # 熔断:本轮 worker 结束(下轮 think 会看到错误提示并改变策略)
                last_response = f"tool loop fuse triggered after {_fuse_streak[0]} identical calls"
                break
        else:
            logger.warning(f"{agent_type} hit max_turns={max_turns}; returning last response")

        # 从 tool_results_log 权威提取 PID/log_file
        result = {"agent": agent_type, "response": last_response}
        for call in tool_results_log:
            if call["name"] == "launch_experiment":
                try:
                    out = json.loads(call["output"])
                    if out.get("pid") is not None:
                        result.update(out)
                        result["experiment_launched"] = True
                        break
                except (json.JSONDecodeError, TypeError):
                    pass
        result.setdefault("experiment_launched", False)
        # Eval 录制用：工具调用名列表（小体积，不进 state 主流程）
        result["_tool_log"] = [{"name": c["name"]} for c in tool_results_log]
        # 空响应占位：耗尽轮数/异常退出时下游不拿空串当结果
        if not str(last_response).strip():
            last_response = (
                f"(worker 达到 max_turns={max_turns} 轮上限或异常退出，"
                f"未产出最终答案；已执行 {len(tool_results_log)} 次工具调用)")
            result["response"] = last_response
        return result

    def _review_and_launch(self, code_result: dict, task_description: str,
                           cycle: int) -> dict:
        """Review Agent 流程：审查 code 写的训练脚本，通过才补 launch。

        1. review worker（只读工具）审查 workspace 中的训练脚本
        2. 输出 JSON {approved, issues}
        3. approved → code worker 第二段补 launch_experiment
        4. not approved → 返回 review_issues，不 launch（reflect 下一轮修代码）
        """
        # ── 连续拒绝熔断：连续 2 次拒绝 → 强制放行（打破"修改→拒绝"循环）──
        # 第 2 次起直接尝试 launch，用真实训练结果（成功或干跑失败）打破空转。
        if self._review_reject_streak >= 2:
            logger.warning(
                "Review rejected %d consecutive times; force-launching to break loop",
                self._review_reject_streak)
            self._review_reject_streak = 0
            launch_task = (
                f"用户任务：{task_description[:200]}\n"
                f"工作目录：{self.workspace}\n"
                f"请立即用 launch_experiment 启动训练（先用 list_files 确认脚本路径）。"
            )
            launch_result = self._run_worker_single_step(
                "code", launch_task, system_prompt=_WORKER_PROMPTS.get("code", ""))
            merged = dict(code_result)
            merged.update(launch_result)
            merged["_tool_log"] = code_result.get("_tool_log", []) + \
                launch_result.get("_tool_log", [])
            merged["review"] = {"approved": "force", "issues": [],
                                "summary": "连续拒绝后强制放行（熔断）"}
            return merged

        print("  [review] 审查训练脚本...")
        review_task = (
            f"审查以下任务生成的训练脚本：{task_description[:200]}\n"
            f"工作目录：{self.workspace}\n"
            f"请先 list_files / list_tree 找到刚写的训练脚本，"
            f"然后按审查清单逐项检查，最后输出 JSON 结论。"
        )
        try:
            review_result = self._run_worker_single_step(
                "review", review_task, system_prompt=REVIEW_AGENT_PROMPT)
        except Exception as exc:
            logger.warning(f"Review agent failed (non-fatal): {exc}")
            return code_result

        # 解析 review 结论（容错：非 JSON/空响应 → 视为不通过且有明确说明，
        # 绝不产生 approved=False + issues=[] 的"空拒绝"——那会让下一轮无从修改）
        review_output = str(review_result.get("response", "")).strip()
        approved = False
        issues: list = []
        parsed = _parse_json_response(review_output)
        if parsed:
            approved = bool(parsed.get("approved", False))
            issues = parsed.get("issues", []) if isinstance(parsed.get("issues"), list) else []
            if not issues and not approved:
                # 拒绝但未列问题 → 补充默认说明（带审查输出摘录供下轮参考）
                issues = [{"severity": "high", "file": "?",
                           "message": "review rejected but listed no specific issues; "
                                      "recheck the script and resubmit. Review output excerpt: "
                                      f"{review_output[:150]}"}]
        else:
            issues = [{"severity": "medium", "file": "?", "message": "review did not return JSON"}]
            if not review_output:
                issues = [{"severity": "high", "file": "?",
                           "message": "review agent returned no content (possibly hit the turn limit)"}]

        self._emit_event("review_result", phase="execute",
                         payload={"approved": approved,
                                  "issues": [i.get("message", "")[:100] for i in issues[:5]]})

        if not approved:
            logger.info("Review rejected: %d issues", len(issues))
            self._review_reject_streak += 1
            code_result["review"] = {
                "approved": False,
                "issues": issues,
                "summary": f"训练脚本未通过审查（{len(issues)} 个问题），本轮未启动训练",
            }
            return code_result

        # 审查通过 → 重置熔断计数
        self._review_reject_streak = 0

        # ── approved → code worker 补 launch ──
        launch_task = (
            f"用户任务：{task_description[:200]}\n"
            f"工作目录：{self.workspace}\n"
            f"你的训练脚本已通过审查。请立即用 launch_experiment 启动训练"
            f"（先用 list_files 确认脚本路径）。"
        )
        logger.info("Review approved; launching experiment")
        launch_result = self._run_worker_single_step(
            "code", launch_task, system_prompt=_WORKER_PROMPTS.get("code", ""))
        merged = dict(code_result)
        merged.update(launch_result)
        merged["_tool_log"] = code_result.get("_tool_log", []) + \
            launch_result.get("_tool_log", [])
        merged["review"] = {"approved": True, "issues": issues,
                            "summary": f"审查通过（{len(issues)} 个提示）"}
        return merged

    # ═══════════════════════════════════════════════════════════════
    # monitor_node
    # ═══════════════════════════════════════════════════════════════

    def monitor_node(self, state: ResearchState) -> ResearchState:
        print("\n[monitor_node] MONITOR (zero LLM)...")
        self._update_state({"phase": "monitor", "ts": time.time()})
        self._emit_event("node_start", phase="monitor",
                         payload={"cycle": state.get("cycle", 0)})
        exec_data = _safe_json(state.get("execute_result", "{}"))
        pid = exec_data.get("pid")
        log_file = exec_data.get("log_file")

        if not pid:
            return {**state, "execute_result": json.dumps(
                {**exec_data, "experiment_status": "no_pid"}, ensure_ascii=False)}

        # state 标记 running + 时间戳:激活 safety.scan_violations 的 stale 检测
        # (历史 bug:从不写 status="running",stale 检查恒不触发)
        self._update_state({
            "phase": "monitor", "status": "running",
            "updated_at": time.time(), "ts": time.time(),
            "pid": pid, "log_file": log_file,
        })

        print(f"  PID={pid}, log={log_file}")
        # 工具层 launch 不经过 monitor.launch_experiment → 手动登记,
        # 否则 wait_for_completion 的耗时/状态统计全为空(历史 bug)
        try:
            self.monitor.track_experiment(
                pid=int(pid), log_file=str(log_file or ""),
                command=str(exec_data.get("command", "")))
        except Exception as exc:
            logger.warning(f"track_experiment failed: {exc}")
        monitor_result = self.monitor.wait_for_completion(
            pid=pid, log_file=log_file,
            notify=self.config.get("monitor", {}).get("notify_on_complete", True),
            on_progress=self._update_monitor_progress,   # 训练中周期更新 epoch 进度
            should_stop=lambda: not self._running,       # 退出信号 → 提前终止监控
            max_wait_hours=float(self.config.get("monitor", {}).get("max_wait_hours", 0)),
        )
        exec_data["training_logs"] = monitor_result.get("log_tail", "")
        exec_data["final_metrics"] = monitor_result.get("metrics", {})
        exec_data["experiment_status"] = monitor_result.get("status", "completed")
        exec_data["terminal_state"] = monitor_result.get("terminal_state", "")
        exec_data["interrupted"] = monitor_result.get("interrupted", False)

        # G3 崩溃恢复上下文:训练崩溃(非正常结束)时,把"恢复决策所需的
        # 事实"(最后 checkpoint / 日志尾部 / 建议)写成一次性上下文,
        # 下一轮 think 读取 —— 替代"让 LLM 自己翻文件找 checkpoint"。
        if exec_data["experiment_status"] == "failed":
            try:
                self._write_crash_context(exec_data)
            except Exception as exc:
                logger.warning(f"crash context write failed: {exc}")

        print(f"  status={exec_data['experiment_status']}")
        return {**state, "execute_result": json.dumps(exec_data, ensure_ascii=False)}

    def _write_crash_context(self, exec_data: dict) -> None:
        """G3:崩溃后写一次性恢复上下文(.crash_context.json,think 读取后删除)。"""
        ckpt_dir = self.workspace / "checkpoints"
        best = ckpt_dir / "best_model.pth"
        ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth")) if ckpt_dir.is_dir() else []
        resume_hint = (
            f"python train.py --resume {best.name}"
            if best.exists() else
            (f"python train.py --resume {ckpts[-1].name}" if ckpts else
             "无 checkpoint —— 从头训练"))
        context = {
            "status": "failed",
            "terminal_state": str(exec_data.get("terminal_state", "") or ""),
            "log_tail": str(exec_data.get("training_logs", ""))[-1500:],
            "has_best_checkpoint": best.exists(),
            "checkpoints": [c.name for c in ckpts[-3:]],
            "resume_hint": resume_hint,
            "recommendation": (
                "优先续训(resume_hint);仅当无 checkpoint 或连续多次崩溃时才从头。"
                "在 task 里明确写出 resume 命令。"),
        }
        (self.workspace / ".crash_context.json").write_text(
            json.dumps(context, ensure_ascii=False), encoding="utf-8")

    # ═══════════════════════════════════════════════════════════════
    # reflect_node — Leader REFLECT（带对话历史 + 完整持久化）
    # ═══════════════════════════════════════════════════════════════

    def reflect_node(self, state: ResearchState) -> ResearchState:
        print("\n[reflect_node] Leader REFLECT...")
        self._update_state({"phase": "reflect", "ts": time.time()})
        self._emit_event("node_start", phase="reflect",
                         payload={"cycle": state.get("cycle", 0)})
        exec_data = _safe_json(state.get("execute_result", "{}"))
        think_data = _safe_json(state.get("think_result", "{}"))
        cycle = state.get("cycle", 0)

        # 省 token(monitor 之后的第二大消耗点):训练日志不进 LLM。
        # 指标已由 monitor 结构化提取(final_metrics),这里只保留日志尾部
        # 1500 字符作为证据,避免几万 token 的日志塞进 reflect prompt。
        if isinstance(exec_data, dict) and exec_data.get("training_logs"):
            log_tail = str(exec_data["training_logs"])
            if len(log_tail) > 1500:
                exec_data = {**exec_data,
                             "training_logs": "...[日志已截断]...\n" + log_tail[-1500:]}

        context = {
            "brief": state.get("task", ""),
            "memory_log": self.memory.get_log(),
            "experiment_result": exec_data,
            "cycle": cycle,
        }
        self._enrich_context(context)

        prompt = (
            f"## Task: REFLECT\n\n"
            f"## Project Brief\n{context['brief']}\n\n"
            f"## Memory Log\n{context['memory_log']}\n\n"
            f"## Experiment Result\n{json.dumps(context['experiment_result'], ensure_ascii=False, indent=2)}\n\n"
            + "".join(f"## {k}\n{v}\n\n" for k, v in context.items()
                      if k in ("Active Violations", "Phase Gate", "Progress Signal",
                               "Recent Experiments", "Dead Ends (do NOT retry these)",
                               "Durable Insights"))
            + f"## Cycle: {cycle}\n\n"
            "请分析结果，输出 JSON。"
        )

        try:
            messages = list(self._leader_history)
            messages.append(HumanMessage(content=prompt))
            reflect_llm = self._llm_reflect or self.llm
            # ── 结构化输出 ──
            structured_llm = None
            if LeaderDecision is not None:
                try:
                    structured_llm = reflect_llm.with_structured_output(LeaderDecision)
                except Exception:
                    logger.debug("with_structured_output not supported for reflect, falling back to text")
            if structured_llm is not None:
                try:
                    decision: LeaderDecision = structured_llm.invoke([  # type: ignore[no-untyped-call]
                        SystemMessage(content=LEADER_REFLECT_PROMPT),
                        *messages,
                    ])
                    raw = decision.model_dump_json()
                    degraded = False
                    usage = {}
                except Exception as exc:
                    logger.warning(
                        "structured output failed for reflect (%s), falling back to text mode",
                        str(exc)[:100],
                    )
                    raw, degraded, usage = self._safe_llm_call(
                        reflect_llm,
                        system=LEADER_REFLECT_PROMPT,
                        messages=[SystemMessage(content=LEADER_REFLECT_PROMPT), *messages],
                        actor="leader", action="reflect",
                    )
            else:
                raw, degraded, usage = self._safe_llm_call(
                    reflect_llm,
                    system=LEADER_REFLECT_PROMPT,
                    messages=[SystemMessage(content=LEADER_REFLECT_PROMPT), *messages],
                    actor="leader", action="reflect",
                )
            # 费用追踪
            if usage:
                self.cost_tracker.record_call(
                    model=reflect_llm.model_name,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    actor="leader", action="reflect"
                )
            # REFLECT 后清空历史，下一轮重新开始
            self._leader_history = []
            result = _parse_json_response(str(raw))
            self.audit.record(actor="leader", action="llm:reflect",
                            target=reflect_llm.model_name,
                            result="success" if (result.get("decision") and not degraded) else ("degraded" if degraded else "empty"))
            # ── 创新度评估(用户审查:reflect 评判本轮想法质量,联动创新度门)──
            innovation = str(result.get("innovation", "") or "").strip().lower()
            if innovation not in ("high", "medium", "low"):
                innovation = "medium"
            self._last_innovation = (
                innovation, str(result.get("innovation_reason", "") or "")[:160])
            # 连续无创新点计数(信号 2):low 连续累计,high/medium 清零
            self._low_innovation_streak = (
                self._low_innovation_streak + 1 if innovation == "low" else 0)
            logger.info("reflect 创新度: %s — %s (连续 low: %d)",
                        innovation, self._last_innovation[1][:80],
                        self._low_innovation_streak)
            # 画像反馈：如果执行失败，记录到用户画像的避坑列表
            if exec_data.get("experiment_status") == "failed" and exec_data.get("terminal_state"):
                dodge = f"[{exec_data.get('terminal_state', '')}] {result.get('decision', '')}"[:200]
                try:
                    UserProfileStore(self.workspace).update_dodge(dodge)
                except Exception:
                    pass
        except Exception as exc:
            logger.error(f"REFLECT 失败: {exc}")
            self._leader_history = []
            self._last_innovation = ("medium", "")
            self._low_innovation_streak = 0
            result = {"decision": f"REFLECT error: {str(exc)[:200]}", "milestone": ""}
            self.audit.record(actor="leader", action="llm:reflect", result="failed",
                            detail={"error": str(exc)[:200]})

        # ── 持久化：MemoryManager 文件 + LangGraph Store ──
        # epoch 秒：跨年/同分钟排序正确（%m-%d 字符串排序在跨年时错乱）
        timestamp = time.time()

        if result.get("milestone"):
            self.memory.log_milestone(result["milestone"])
            print(f"  milestone: {result['milestone'][:80]}")
        if result.get("decision"):
            self.memory.log_decision(result["decision"])
            print(f"  decision: {result['decision'][:80]}")

        # ── Plan 修订（Plan-then-Execute with Replan）──
        # 本轮执行结果 → 标记当前 running 步骤 done/failed；若 Leader 返回了
        # 新 plan（重新规划），则整体替换
        plan = self._parse_plan(state.get("plan", ""))
        revised_plan = self._replan(plan, exec_data, result)
        if revised_plan != plan:
            logger.info("Plan revised: %d steps", len(revised_plan))
            self._emit_event("plan_revised", phase="reflect",
                             payload={"steps": len(revised_plan)})
        self._last_plan = json.dumps(revised_plan, ensure_ascii=False)

        # ── Store 写入：Episodic（完整记录）+ Semantic（知识点）──
        new_cycle = cycle + 1
        self._persist_to_store(new_cycle, think_data, exec_data, result, timestamp)

        # ── Consolidation：每 5 轮去重合并 ──
        if new_cycle % 5 == 0:
            self._consolidate_memories()

        # ── 账本 + 日志 + Obsidian（原版全部保留）──
        self._record_cycle_outcome(think_data, exec_data, result)
        self._record_to_ledger(new_cycle, think_data, exec_data, result)
        self._settle_hypothesis(think_data, exec_data, result, cycle=cycle)
        self._refresh_obsidian(result)

        # ── 训练后快照（post）：此时 checkpoints/ 权重已生成，manifest 能记录模型文件。
        #    回退优先用 post 快照，才能带权重续训。
        try:
            if exec_data.get("experiment_status") == "completed":
                self.snapshotter.create(label=f"cycle{cycle}_post", cycle=cycle)
                self.snapshotter.cleanup()
        except Exception as exc:
            logger.warning("Post-snapshot failed (non-fatal): %s", exc)

        return {
            **state,
            "cycle": new_cycle,
            # 清空本轮 think/execute，避免状态泄漏到下轮导致重复执行上一轮任务
            # （旧 think_result 的 next_stage 会覆盖路由，把新 cycle 打回 execute）
            "think_result": "",
            "execute_result": "",
            "reflect_result": json.dumps(result, ensure_ascii=False),
            "directive": "",   # 指令已在本轮执行，清空避免永久禁用反卡死
            "plan": json.dumps(revised_plan, ensure_ascii=False),
        }

    # ═══════════════════════════════════════════════════════════════
    # route_next
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def route_next(state: ResearchState) -> str:
        agent = state.get("next_agent", "think")
        if agent in ("think", "execute", "monitor", "reflect"):
            return agent
        return END

    # ═══════════════════════════════════════════════════════════════
    # build + run
    # ═══════════════════════════════════════════════════════════════

    def build(self) -> StateGraph:
        graph = StateGraph(ResearchState)
        graph.add_node("supervisor", self.supervisor_node)
        graph.add_node("think", self.think_node)
        graph.add_node("execute", self.execute_node)
        graph.add_node("monitor", self.monitor_node)
        graph.add_node("reflect", self.reflect_node)
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges("supervisor", self.route_next)
        graph.add_edge("think", "supervisor")
        graph.add_edge("execute", "supervisor")
        graph.add_edge("monitor", "supervisor")
        graph.add_edge("reflect", "supervisor")
        return graph

    def run(self):
        logger.info(f"ResearchGraph 启动 | project={self.project_dir}")

        # ── Rollback 指令检测（graph 启动前）──
        rollback_result = self.rollback_handler.check_directive()
        if rollback_result:
            print(f"\n[rollback] {rollback_result}")
            logger.info("Rollback executed: %s", rollback_result)
            if "can RESUME training" in rollback_result:
                print("  → Agent 将从模型 checkpoint 续训")
            elif "start from scratch" in rollback_result:
                print("  → 模型文件丢失，Agent 将从头训练")

        graph = self.build().compile(checkpointer=self.checkpointer, store=self.store)

        start_cycle = self._load_cycle_counter()
        initial_state: ResearchState = {
            "task": self._task_content,
            "cycle": start_cycle,
            "max_cycles": self.max_cycles,
            "cooldown": self.cooldown,
            "next_agent": "think",
            "directive": "",
            "think_result": "",
            "execute_result": "",
            "reflect_result": "",
            "final_answer": "",
            "plan": "",
        }

        # 用 thread_id 支持跨 run 恢复
        config = {"configurable": {"thread_id": str(self.project_dir)}}
        # 0.5-B 崩溃恢复语义:该 thread 已有 checkpoint 时只传增量输入
        # (外部信号 directive),保留 checkpoint 里的执行状态 —— 否则
        # initial_state 全量覆盖 checkpoint 的 cycle/think/execute 字段,
        # 崩溃恢复会回退 cycle / 跳过节点(审查项 30 实测证实)。
        try:
            has_ckpt = self.checkpointer.get_tuple(config) is not None
        except Exception:
            has_ckpt = False
        invoke_input = initial_state
        if has_ckpt:
            invoke_input = {"directive": initial_state.get("directive", "")}
            logger.info("检测到既有 checkpoint:增量恢复(仅外部信号,保留执行状态)")
        try:
            result = graph.invoke(invoke_input, config)
        except Exception as exc:
            # ── Checkpoint 损坏自愈（LangGraph 已知限制：Checkpoints ≠
            #    Durable Execution，崩溃残留的半成品 checkpoint 会导致启动失败）──
            # 重建 checkpointer（丢弃损坏状态，从 cycle_counter 续跑）重试一次。
            if "pending_writes" in str(exc) or "unpack" in str(exc):
                logger.error(
                    "Checkpoint 损坏（%s）— 删除损坏状态，从当前 cycle 重新开始",
                    str(exc)[:150])
                try:
                    # 自愈前先备份损坏库(0.5-C):serde 升级/半写损坏时
                    # 保留现场供排查,不静默丢历史
                    ckpt_path = self.workspace / "checkpoints.db"
                    try:
                        import shutil as _shutil
                        _shutil.copy2(
                            ckpt_path,
                            self.workspace /
                            f"checkpoints.db.corrupt-{int(time.time())}.bak")
                        logger.warning("已备份损坏 checkpoint -> checkpoints.db.corrupt-*.bak")
                    except OSError:
                        pass  # 备份失败不阻塞自愈
                    # 删除损坏的 checkpoints.db（含 WAL/SHM 残留）后重建
                    for suffix in ("", "-wal", "-shm"):
                        p = Path(str(ckpt_path) + suffix)
                        if p.exists():
                            p.unlink()
                    self.checkpointer = SqliteCheckpointer(ckpt_path)
                    graph = self.build().compile(
                        checkpointer=self.checkpointer, store=self.store)
                    result = graph.invoke(initial_state, config)
                except Exception as exc2:
                    logger.error(f"Checkpoint 重建后仍失败: {exc2}")
                    raise
            else:
                raise
        finally:
            # 生命周期收尾：关闭持久化 Store（幂等），确保 WAL 落盘
            try:
                self.store.close()
            except Exception:
                pass
            # 释放实例锁（正常结束/异常退出都释放）
            self._release_agent_lock()
            # Windows 关键修复：sqlite3 的 `with` 块退出只 commit、不 close 连接，
            # 句柄释放依赖完整 GC。不显式 collect 会锁住 workspace 下的 .db 文件
            # （checkpoints.db / memory.db），导致整个项目目录无法删除/清理。
            gc.collect()

        self._save_cycle_counter(result.get("cycle", 0))

        print("\n[supervisor] 最终回答:")
        print(result.get("final_answer", "(无总结)"))
        logger.info("ResearchGraph 停止。")
        return result

    # ═══════════════════════════════════════════════════════════════
    # v2 信号注入（原版 _enrich_context，完整保留）
    # ═══════════════════════════════════════════════════════════════

    def _enrich_context(self, context: dict):
        if self.ledger is not None:
            try:
                summary = self.ledger.summary(self._ledger_cfg.get("recent_in_context", 5))
                if summary:
                    context["Recent Experiments"] = summary
            except Exception as exc:
                logger.warning(f"ledger summary failed: {exc}")

            metric_key = self._ledger_cfg.get("metric_key", "")
            direction = self._ledger_cfg.get("metric_direction", "higher_better")
            if metric_key and self._stagnation_cfg.get("enabled", True):
                try:
                    verdict = detect_stagnation(
                        self.ledger.all(), metric_key, direction=direction,
                        threshold_cycles=self._stagnation_cfg.get("threshold_cycles", 3),
                        min_delta=self._stagnation_min_delta(metric_key),
                    )
                    context["Progress Signal"] = self._format_stagnation(verdict)
                except Exception as exc:
                    logger.warning(f"stagnation detection failed: {exc}")

            # 上轮 reflect 的创新度评价(想法质量信号注入决策;仅 low/high
            # 注入,medium 是常态噪声;low 会联动创新度门直接升级 idea agent)
            last_inn = getattr(self, "_last_innovation", None)
            if last_inn and last_inn[0] in ("low", "high"):
                context["Innovation Signal"] = (
                    f"上轮 reflect 创新度评价: {last_inn[0]}"
                    + (f" — {last_inn[1]}" if last_inn[1] else ""))

            if metric_key and self._gates_cfg.get("enabled", False):
                try:
                    gate = check_phase_gate(
                        self.ledger.all(), metric_key,
                        threshold=self._gates_cfg.get("threshold", 0.0),
                        direction=self._gates_cfg.get("direction", direction),
                    )
                    context["Phase Gate"] = self._format_gate(gate)
                except Exception as exc:
                    logger.warning(f"phase gate check failed: {exc}")

        if self.journal is not None:
            try:
                tail_chars = int(self._journal_cfg.get("tail_in_context", 1500))
                dead_ends = self.journal.dead_ends_tail(tail_chars)
                if "- [" in dead_ends:
                    context["Dead Ends (do NOT retry these)"] = dead_ends.strip()
                insights = self.journal.insights_tail(tail_chars)
                if "- [" in insights:
                    context["Durable Insights"] = insights.strip()
            except Exception as exc:
                logger.warning(f"journal tail failed: {exc}")

        if self._safety_cfg.get("enabled", True):
            try:
                violations = safety.scan_violations(
                    self._load_state(), self._no_progress_streak, time.time(),
                    fail_threshold=self._safety_cfg.get("fail_threshold", 3),
                    stale_state_hours=self._safety_cfg.get("stale_state_hours", 6),
                )
                if violations:
                    context["Active Violations"] = "\n".join(f"- {v}" for v in violations)
            except Exception as exc:
                logger.warning(f"violation scan failed: {exc}")

    # ═══════════════════════════════════════════════════════════════
    # 记忆检索与持久化（Store）
    # ═══════════════════════════════════════════════════════════════

    def _tried_arxiv_ids(self) -> set:
        """从实验账本提取已尝试过的论文 id(假设/结论里的 [arXiv:xxx] 引用)。

        RAG 注入的新鲜度闸:已实验过的论文不再喂给 idea/leader,
        防止反复找同一创新点 → 重复实验(重复扣 GPU 时长)。
        """
        tried: set = set()
        ledger = getattr(self, "ledger", None)
        if ledger is None:
            return tried
        try:
            for entry in ledger.all()[-20:]:
                text = " ".join([
                    str(entry.get("hypothesis", "") or ""),
                    str(entry.get("conclusion", "") or ""),
                ])
                for m in re.finditer(r"\[arXiv:([0-9.]+(?:v\d+)?)\]",
                                     text, re.IGNORECASE):
                    tried.add(m.group(1).split("v")[0])
        except Exception:
            pass
        return tried

    def _ingest_user_literature(self):
        """启动时摄取 USER_LITERATURE.md（用户提供的文献）到知识库。

        幂等：同一来源且内容哈希未变化时不重复摄取。
        """
        lit_path = self.workspace / "USER_LITERATURE.md"
        if not lit_path.exists():
            return
        content = lit_path.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            return
        import hashlib
        digest = hashlib.md5(content.encode("utf-8")).hexdigest()[:12]
        existing = self.rag.retrieve("", top_k=50,
                                     source_filter="USER_LITERATURE.md")
        if any((e.get("metadata") or {}).get("content_hash") == digest
               for e in existing):
            return  # 内容未变化，跳过
        self.rag.add_document(
            content, source="USER_LITERATURE.md",
            metadata={"content_hash": digest})

    def _ingest_idea_notes(self):
        """idea agent 分析完文献后，把 IDEA_NOTES.md 摄入知识库（幂等）。"""
        notes_path = self.workspace / "IDEA_NOTES.md"
        if not notes_path.exists():
            return
        content = notes_path.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            return
        import hashlib
        digest = hashlib.md5(content.encode("utf-8")).hexdigest()[:12]
        existing = self.rag.retrieve("", top_k=50, source_filter="IDEA_NOTES.md")
        if any((e.get("metadata") or {}).get("content_hash") == digest
               for e in existing):
            return
        self.rag.add_document(
            content, source="IDEA_NOTES.md",
            metadata={"content_hash": digest})

    def _recall_from_store(self, state: ResearchState) -> dict:
        """Recall-before-reason：从 Store 检索相关记忆，不塞全量。
        返回 context 增量 dict，key 用 emoji 前缀方便 LLM 区分。"""
        result = {}
        ns_ep = ("project", self._store_project, "episodes")
        ns_sm = ("project", self._store_project, "semantic")
        ns_pr = ("project", self._store_project, "procedural")

        # ── RAG 知识检索（论文/文档知识库 → 注入实验方向上下文）──
        if self._rag_enabled:
            try:
                rag_query = str(state.get("task", ""))[:200]
                rag_hits = self.rag.retrieve(rag_query, top_k=3)
                if rag_hits:
                    # 新鲜度过滤:从实验账本提取已尝试过的论文(假设/结论里的
                    # [arXiv:id] 引用)→ 不再注入,防反复找同一创新点 → 重复实验
                    tried_ids = self._tried_arxiv_ids()
                    lines = []
                    # 场景感知注入:方法/实验段优先(idea agent 决策假设靠
                    # 「怎么做」而非「讲了什么」),并给更长展示额度
                    ordered = sorted(
                        rag_hits,
                        key=lambda h: 0 if (
                            (h.get("metadata") or {}).get("section") == "methods")
                        else 1,
                    )
                    for h in ordered:
                        section = (h.get("metadata") or {}).get("section", "")
                        budget = 300 if section == "methods" else 150
                        src = str(h.get("source", ""))[:60]
                        if tried_ids and any(tid in src for tid in tried_ids):
                            continue  # 已实验过该论文 → 跳过
                        text = str(h.get("text", ""))[:budget]
                        # 外部论文内容注入防护:文献文本可能夹带 prompt injection
                        # 指令 → 检测命中直接丢弃该 chunk(不清洗后放行)
                        try:
                            is_attack, _reason = InputGuard.detect_injection(text)
                        except Exception:
                            is_attack = False
                        if is_attack:
                            continue
                        sim = h.get("similarity", 0)
                        lines.append(f"- [{src}] (sim={sim:.2f}) {text}")
                    if lines:
                        result["📚 RAG Knowledge"] = "\n".join(lines)
            except Exception:
                pass

        # ── 实验模式记忆（procedural）：成功/失败配置组合，注入避坑上下文 ──
        try:
            patterns = list(self.store.search(ns_pr, limit=8))
            if patterns:
                lines = []
                for item in patterns:
                    val = item.value if isinstance(item.value, dict) else {}
                    outcome = val.get("outcome", "")
                    mark = "✅" if outcome == "success" else "❌"
                    config = val.get("config", {})
                    config_str = ", ".join(f"{k}={v}" for k, v in config.items())[:120]
                    metric = val.get("metric", "")
                    note = str(val.get("note", ""))[:80]
                    line = f"- {mark} [{outcome}] {config_str}"
                    if metric:
                        line += f" → {metric}"
                    if note:
                        line += f" ({note})"
                    lines.append(line)
                if lines:
                    result["🧪 Known Patterns (Store)"] = "\n".join(lines)
        except Exception:
            pass

        try:
            episodes = list(self.store.search(ns_ep, limit=5))
            if episodes:
                lines = []
                for item in episodes:
                    val = item.value if isinstance(item.value, dict) else {}
                    cycle_tag = f"[cycle {val.get('cycle', '?')}]"
                    milestone = str(val.get("milestone", ""))[:120]
                    if milestone:
                        lines.append(f"- {cycle_tag} {milestone}")
                if lines:
                    result["📜 Recent Episodes (Store)"] = "\n".join(lines)
        except Exception:
            pass

        try:
            insights = list(self.store.search(ns_sm, limit=5))
            if insights:
                lines = []
                for item in insights:
                    val = item.value if isinstance(item.value, dict) else {}
                    text = str(val.get("text", ""))[:150]
                    tags = val.get("tags", [])
                    tag_str = f" [{', '.join(tags)}]" if tags else ""
                    if text:
                        lines.append(f"- {text}{tag_str}")
                if lines:
                    result["📖 Related Knowledge (Store)"] = "\n".join(lines)
        except Exception:
            pass

        # ── 跨项目语义检索（CrossProjectStore）──
        try:
            query = str(state.get("reflect_result", "") or state.get("task", ""))
            cross_results = self.cross_store.search_cross_project(
                query, exclude_project=self._project_name, limit=3
            )
            if cross_results:
                lines = []
                for r in cross_results:
                    proj_tag = f"[{r['project']}]"
                    text = r["text"][:150]
                    sim = r.get("similarity", 0)
                    lines.append(f"- {proj_tag} (sim={sim:.2f}) {text}")
                if lines:
                    result["🌐 Cross-Project Insights"] = "\n".join(lines)
        except Exception:
            pass

        return result

    def _persist_to_store(self, cycle: int, think: dict, execute: dict,
                          reflect: dict, timestamp: float):
        """REFLECT 后写入 Store：Episodic（完整记录）+ Semantic（知识点）。"""
        try:
            # Episodic：本轮完整记录
            self.store.put(
                ("project", self._store_project, "episodes"),
                f"cycle_{cycle}",
                {
                    "cycle": cycle,
                    "think": think,
                    "exec_summary": {
                        "agent": execute.get("agent", ""),
                        "experiment_launched": execute.get("experiment_launched"),
                        "status": execute.get("experiment_status"),
                    },
                    "milestone": reflect.get("milestone", ""),
                    "decision": reflect.get("decision", ""),
                    "ts": timestamp,
                    "plan": self._parse_plan(self._last_plan or ""),
                },
            )
        except Exception as exc:
            logger.warning(f"Store episodic write failed: {exc}")

        try:
            # Semantic：提取知识点（只有 milestone 才存）
            milestone = reflect.get("milestone", "")
            if milestone:
                self.store.put(
                    ("project", self._store_project, "semantic"),
                    f"insight_{cycle}",
                    {
                        "text": milestone,
                        "ts": timestamp,
                        "tags": self._extract_tags(reflect),
                        "cycle": cycle,
                    },
                )
        except Exception as exc:
            logger.warning(f"Store semantic write failed: {exc}")

        # ── 实验模式记忆（procedural）：结构化记录配置组合成功/失败 ──
        try:
            pattern = self._extract_pattern(think, execute, reflect)
            if pattern:
                self.store.put(
                    ("project", self._store_project, "procedural"),
                    f"pattern_{cycle}",
                    pattern,
                )
        except Exception as exc:
            logger.warning(f"Store procedural write failed: {exc}")

        # ── 跨项目持久化（CrossProjectStore，SQLite 持久化）──
        try:
            milestone = reflect.get("milestone", "")
            decision = reflect.get("decision", "")
            if milestone:
                self.cross_store.add(
                    text=milestone,
                    project=self._project_name,
                    namespace="semantic",
                    metadata={
                        "cycle": cycle,
                        "tags": self._extract_tags(reflect),
                        "decision": decision,
                    },
                )
        except Exception as exc:
            logger.warning(f"CrossProjectStore write failed: {exc}")

    @staticmethod
    def _extract_pattern(think: dict, execute: dict, reflect: dict) -> dict:
        """从本轮实验中提取结构化实验模式（procedural memory 条目）。

        返回 {config, outcome, metric, terminal_state, note}；无可用信息返回 {}。
        """
        outcome_raw = execute.get("experiment_status", "")
        if not outcome_raw or outcome_raw == "no_pid":
            return {}
        outcome = "success" if outcome_raw == "completed" else "failed"

        # config：从 think 的 hypothesis/task 中提取关键参数（复用标签关键词）
        tags = ResearchGraph._extract_tags(think)
        config = {t: "?" for t in tags[:6]} if tags else {}

        metric = ""
        final_metrics = execute.get("final_metrics") or {}
        if isinstance(final_metrics, dict):
            for key in ("accuracy", "acc", "loss", "f1", "mAP", "FID"):
                if final_metrics.get(key) is not None:
                    metric = f"{key}={final_metrics[key]}"
                    break

        return {
            "config": config,
            "outcome": outcome,
            "metric": metric,
            "terminal_state": str(execute.get("terminal_state", "")),
            "note": str(reflect.get("decision", ""))[:80],
            "ts": time.time(),
        }

    @staticmethod
    def _extract_tags(reflect: dict) -> list:
        """从反思结果中提取标签（简单关键词匹配）。"""
        text = json.dumps(reflect, ensure_ascii=False).lower()
        keywords = ["lr", "learning_rate", "batch", "optimizer", "sgd", "adam",
                    "loss", "accuracy", "precision", "recall", "f1", "mAP",
                    "dropout", "normalization", "bn", "batch_norm", "layer_norm",
                    "augmentation", "scheduler", "cosine", "warmup",
                    "overfitting", "underfitting", "gradient", "nan", "inf",
                    "vit", "resnet", "transformer", "cnn"]
        return sorted(set(kw for kw in keywords if kw in text))[:8]

    def _consolidate_memories(self):
        """每 N 轮合并重复知识点 + 实验模式，淘汰矛盾的旧结论。
        简单策略：文本相似度 > 0.7 的条目合并为一条，保留最新的。"""
        ns = ("project", self._store_project, "semantic")
        try:
            items = list(self.store.search(ns, limit=50))
            if len(items) < 2:
                # semantic 不足两条时仍尝试 procedural 去重
                self._consolidate_procedural()
                return
        except Exception:
            return

        # 按 ts（epoch 秒）数值排序，最新在前；老数据（字符串 ts）解析失败排最后
        def _ts_key(item) -> float:
            ts = item.value.get("ts", 0) if isinstance(item.value, dict) else 0
            try:
                return float(ts)
            except (TypeError, ValueError):
                return 0.0

        # 简单去重：按 text 前缀相似度分组
        merged = []
        seen_texts = set()
        for item in sorted(items, key=_ts_key, reverse=True):
            text = str(item.value.get("text", ""))[:100] if isinstance(item.value, dict) else ""
            if not text:
                continue
            # 检查是否和已有条目相似
            is_dup = False
            for existing in seen_texts:
                if _text_similarity(text, existing) > 0.7:
                    is_dup = True
                    break
            if not is_dup:
                seen_texts.add(text)
                merged.append(item)

        if len(merged) < len(items):
            logger.info(f"Consolidation: {len(items)} → {len(merged)} semantic memories")
            # 重建：删旧写新
            try:
                for item in items:
                    try:
                        self.store.delete(ns, item.key)
                    except Exception:
                        pass
                for item in merged:
                    self.store.put(ns, item.key, item.value)
            except Exception as exc:
                logger.warning(f"Store consolidation write failed: {exc}")

        # ── procedural 去重：同配置组合只保留最新一条（避免模式重复膨胀）──
        self._consolidate_procedural()

    def _consolidate_procedural(self):
        """实验模式去重：同 outcome + 相似 config 的模式只保留最新一条。"""
        ns_pr = ("project", self._store_project, "procedural")
        try:
            items = list(self.store.search(ns_pr, limit=50))
            if len(items) < 2:
                return
        except Exception:
            return

        def _ts_key(item) -> float:
            ts = item.value.get("ts", 0) if isinstance(item.value, dict) else 0
            try:
                return float(ts)
            except (TypeError, ValueError):
                return 0.0

        def _config_key(item) -> str:
            cfg = item.value.get("config", {}) if isinstance(item.value, dict) else {}
            return json.dumps(sorted(cfg.items()), ensure_ascii=False)

        merged = []
        seen_configs = set()
        for item in sorted(items, key=_ts_key, reverse=True):
            cfg_key = _config_key(item)
            if cfg_key in seen_configs:
                continue
            seen_configs.add(cfg_key)
            merged.append(item)

        if len(merged) < len(items):
            logger.info(f"Procedural consolidation: {len(items)} → {len(merged)} patterns")
            try:
                for item in items:
                    try:
                        self.store.delete(ns_pr, item.key)
                    except Exception:
                        pass
                for item in merged:
                    self.store.put(ns_pr, item.key, item.value)
            except Exception as exc:
                logger.warning(f"Store procedural consolidation failed: {exc}")

    @staticmethod
    def _format_stagnation(verdict: dict) -> str:
        if verdict.get("reason"):
            return f"{verdict['reason']} (metric={verdict.get('metric_key', '')})"
        flag = "STAGNATING" if verdict.get("stagnating") else "improving"
        return (f"{flag}: best {verdict.get('metric_key')}={verdict.get('best')}, "
                f"{verdict.get('cycles_since_improvement')} cycle(s) since last improvement "
                f"over {verdict.get('n_points')} measured runs.")

    @staticmethod
    def _format_gate(gate: dict) -> str:
        if gate.get("gate_met"):
            return f"Phase gate MET (best metric={gate.get('best_metric')}). OK to pursue innovation."
        return f"Phase gate NOT met: {gate.get('blocker_reason', 'baseline quality not reached')}."

    # ═══════════════════════════════════════════════════════════════
    # 反卡死 + 账本 + 指令 + Obsidian（原版全部保留）
    # ═══════════════════════════════════════════════════════════════

    def _plan_signature(self, plan: dict) -> str:
        normalized = {
            "action": plan.get("action", ""),
            "agent": plan.get("agent", ""),
            "task": " ".join(plan.get("task", "").split())[:300],
            "hypothesis": " ".join(plan.get("hypothesis", "").split())[:200],
        }
        return json.dumps(normalized, sort_keys=True, ensure_ascii=True)

    def _stagnation_min_delta(self, metric_key: str) -> float:
        """停滞判定的自适应 min_delta。

        固定 min_delta 的问题(用户审查):离目标很远时 +0.03pp 也算
        "改进",机械叠加永不触发创新度门;离目标很近时固定 0.1pp 又过严。
        规则(用户提议):
          min_delta = clamp(  max(floor, ratio × 离目标距离),  ≤ 剩余距离 )
        - ratio 默认 0.4(离目标越远,要求单轮提升越大)
        - floor 默认 0.3pp(metric 0.003):低于 0.3pp 的提升视为噪声
          (MNIST 10k 测试样本,0.3pp ≈ 30 样本)
        - 上限 = 剩余距离:达标那一跳(哪怕 < 0.3pp)永远算实质进展,
          否则门会在冲线前把获胜实验改掉
        未配置 target → 退回固定 min_delta。
        """
        st_cfg = self._stagnation_cfg or {}
        try:
            target = float(st_cfg.get("target", 0) or 0)
            ratio = float(st_cfg.get("min_delta_ratio", 0.4))
            floor = float(st_cfg.get("min_delta_floor", 0.0) or 0.0)
        except (TypeError, ValueError):
            target, ratio, floor = 0.0, 0.4, 0.0
        if target > 0 and ratio > 0 and self.ledger is not None:
            try:
                direction = (self._ledger_cfg or {}).get(
                    "metric_direction", "higher_better")
                best = self.ledger.best_metric(metric_key, direction)
                if best is not None:
                    distance = (target - best) if direction == "higher_better" \
                        else (best - target)
                    if distance > 0:
                        raw = max(floor, ratio * distance)
                        return min(raw, distance)
            except Exception:
                pass
        return float(st_cfg.get("min_delta", 0.0) or 0.0)

    @staticmethod
    def _plan_has_innovation(text: str) -> bool:
        """计划文本是否含创新点信号(零 LLM 关键词)。

        用户澄清:「当前没有提供创新点 → 就去找创新点」。创新点 = 具体的
        方法/论文/新思路(如 mixup/CutMix/蒸馏/对比学习),而非机械调参
        (加层/dropout/lr/通道/epochs)。用于任务要求创新时的前置检查。
        """
        t = (text or "").lower()
        return any(kw in t for kw in (
            "论文", "paper", "文献", "创新", "借鉴", "新思路", "新方法",
            "mixup", "cutmix", "cutout", "sgdr", "重启", "蒸馏",
            "对比学习", "标签平滑", "label smooth", "数据混合",
            "meta-learning", "元学习", "正则化方法", "损失函数改造"))

    def _escalate_to_idea(self, think_result: dict, reason: str) -> dict:
        """Escalate to the idea agent: open-ended exploration (writes IDEA_NOTES)."""
        task = str(think_result.get("task", "") or "")
        think_result["agent"] = "idea"
        think_result["task"] = (
            f"(FORCED DIRECTION CHANGE - ESCALATED) {reason}. Plain tuning has "
            f"hit its ceiling; break out of the current approach and propose "
            f"different-dimension / borrowable new methods (use tools to research "
            f"/ read literature), and write your findings to IDEA_NOTES.md.\n"
            f"Original task: {task}")
        self._direction_forced = True
        logger.warning("[innovation] %s -> escalate to idea agent", reason)
        return think_result

    def _maybe_force_direction_switch(self, think_result: dict) -> dict:
        """任务级创新兜底(最小契约约束,非行为监控)。

        用户审查:行为类硬约束(收益递减门/连续低创新门/同维度门)冗余且
        不自然 —— 已删除,路由交由提示词驱动(agents/leader_think_agent.md
        的 Tune vs. Innovate 规程 + few-shot)。仅保留**任务契约兜底**:
        任务要求创新(T6 requires_innovation)且 IDEA_NOTES.md 尚未产出、
        计划又不含创新点 → 升级 idea agent。这是任务成功标准的一部分,
        可一键移除(innovation_required 配置)。
        """
        if not think_result or think_result.get("action") != "experiment":
            return think_result
        # 指令保护:用户明确禁止换方向时不干预
        try:
            directive = str(self._current_directive or "") or ""
        except Exception:
            directive = ""
        if any(kw in directive for kw in ("保持当前", "保持方向", "别换方向",
                                          "不要换", "禁止换", "继续当前")):
            return think_result
        try:
            st_cfg = self._stagnation_cfg or {}
            innovation_required = bool(st_cfg.get("innovation_required", False))
        except Exception:
            return think_result
        if not innovation_required:
            return think_result
        try:
            has_notes = (self.workspace / "IDEA_NOTES.md").exists()
        except Exception:
            has_notes = False
        if not has_notes and not self._plan_has_innovation(
                f"{think_result.get('hypothesis', '')} "
                f"{think_result.get('task', '')}"):
            return self._escalate_to_idea(
                think_result,
                "task requires innovation, but the current plan contains "
                "no innovation point (tuning-only wording)")
        return think_result

    def _apply_no_progress_fallback(self, think_result: dict, directive: str) -> dict:
        # 反卡死:同一计划连续无进展 → 终止本轮研究(wait = finish),
        # 不再空转烧 token/GPU。恢复只能靠 HUMAN_DIRECTIVE 或人工介入
        # —— 语义是「停止空转」而非「临时退避后自恢复」。
        # 有 directive 时不直接跳过反卡死，但仍允许执行指令（不强制 wait）
        if self.no_progress_fallback_threshold <= 0:
            return think_result
        if think_result.get("action") != "experiment":
            return think_result
        signature = self._plan_signature(think_result)
        if (self._no_progress_streak >= self.no_progress_fallback_threshold
                and signature == self._last_no_progress_signature
                and not directive):  # directive 在时不强制 wait，避免掐断用户指令
            reason = (f"Fallback triggered after {self._no_progress_streak} no-progress cycles. "
                      "Stopping to avoid burning tokens/GPU on empty loops.")
            logger.warning(reason)
            self.memory.log_decision(reason)
            if self.journal is not None:
                task_text = " ".join(think_result.get("task", "").split())[:160]
                self.journal.append_dead_end(f"Cycle: repeated with no progress — {task_text}")
            return {"action": "wait", "reason": reason, "decision": reason}
        return think_result

    def _record_cycle_outcome(self, think_result: dict, execute_result: dict, reflect_result: dict):
        if think_result.get("action") != "experiment":
            if think_result.get("action") != "wait":
                self._no_progress_streak = 0
                self._last_no_progress_signature = ""
            return
        signature = self._plan_signature(think_result)
        made_progress = bool(
            execute_result.get("experiment_launched")
            or execute_result.get("final_metrics")
            or reflect_result.get("milestone")
        )
        if made_progress:
            self._no_progress_streak = 0
            self._last_no_progress_signature = ""
            return
        if signature == self._last_no_progress_signature:
            self._no_progress_streak += 1
        else:
            self._last_no_progress_signature = signature
            self._no_progress_streak = 1

    def _plan_duplicate_check(self, think_result: dict, plan: str) -> str:
        """G4:计划与账本最近实验重复检查。返回打回原因;"" = 放行。

        与账本最近 N 条实验的 hypothesis/task 文本相似度 > 阈值 → 重复
        (同一任务已被执行过,继续做只会重复扣 GPU 时长)。
        阈值 0.5(Jaccard 词级:完全复述的实验句通常在 0.5+;措辞完全
        不同的新方向在 0.3 以下,误伤面小)。
        """
        ledger = getattr(self, "ledger", None)
        if ledger is None:
            return ""
        candidates = [str(think_result.get("task", "") or "")]
        try:
            parsed = self._parse_plan(plan)
            for step in parsed:
                candidates.append(str(step.get("title", "") or ""))
                candidates.append(str(step.get("task", "") or ""))
        except Exception:
            pass
        try:
            recent = ledger.all()[-8:]
        except Exception:
            return ""
        for cand in candidates:
            if len(cand.strip()) < 20:
                continue
            for entry in recent:
                prev = " ".join([
                    str(entry.get("hypothesis", "") or ""),
                    str(entry.get("conclusion", "") or ""),
                ])
                if len(prev.strip()) < 20:
                    continue
                try:
                    if _text_similarity(cand, prev) > 0.5:
                        return (
                            f"计划与账本实验重复(相似度>0.5):「{cand[:60]}」"
                            f"与之前实验结论相近。请参考账本(Recent Experiments) "
                            f"与 DEAD_ENDS 换一个未验证的方向,不要重复执行。")
                except Exception:
                    continue
        return ""

    # 假设结算的证据强度阈值(用户审查:单次小幅负结果 ≠ 否证)
    # 指标增量 ≥ REFUTE_DELTA(1pp)的明确下降才算「否证」;
    # 小幅负结果 → inconclusive(可能架构/超参混淆,不冤枉不轻信)。
    _REFUTE_DELTA = 0.01

    def _run_metric_delta(self, execute_result: dict, cycle: int) -> float | None:
        """本轮最佳指标 − 本轮之前账本最佳指标。无可比数据 → None。"""
        metrics = execute_result.get("final_metrics") or {}
        if not isinstance(metrics, dict):
            return None
        best_this = None
        for k in ("accuracy", "acc", "test_acc", "metric"):
            v = metrics.get(k)
            if v is not None:
                try:
                    best_this = max(best_this or 0.0, float(v))
                except (TypeError, ValueError):
                    pass
        if best_this is None:
            return None
        prev_best = None
        try:
            entries = (self.ledger.all() if self.ledger is not None else []) or []
        except Exception:
            entries = []
        for e in entries:
            try:
                if int(e.get("cycle", 0) or 0) >= int(cycle or 0):
                    continue
            except (TypeError, ValueError):
                continue
            m = e.get("metrics") or {}
            if not isinstance(m, dict):
                continue
            for k in ("accuracy", "acc", "test_acc"):
                v = m.get(k)
                if v is not None:
                    try:
                        prev_best = max(prev_best or 0.0, float(v))
                    except (TypeError, ValueError):
                        pass
        if prev_best is None:
            return None
        return best_this - prev_best

    def _settle_hypothesis(self, think_result: dict, execute_result: dict,
                           reflect_result: dict, cycle: int = 0) -> None:
        """G1 假设结算(证据强度感知,用户审查修复)。

        旧规则的两个误沉淀(T6 实测):
        - 「completed + 非空 milestone → confirmed」把负结果(-0.6pp)
          沉淀成 confirmed,证据文本却写着 refuted;
        - 「failed → refuted」把发散误判的健康 run 的**已证实假设**
          沉淀成否证(污染「已否证假设,禁止再次提出」列表)。
        新规则看指标增量(本轮最佳 − 此前账本最佳):
        - failed                → refuted(证据带 terminal_state 真实原因)
        - completed, 增量 ≥ 0   → confirmed
        - completed, -1pp<增量<0 → inconclusive(单次小幅负结果,可能
          架构/超参混淆,证据注明;绝不轻率否证)
        - completed, 增量 ≤ -1pp → refuted(明确下降才否证)
        - 无指标可比/无里程碑    → inconclusive
        同一文本假设自动去重(重复提出 = 复用旧条目,不再新建)。
        """
        try:
            hypotheses = getattr(self, "hypotheses", None)
            if hypotheses is None:
                return
            text = str(think_result.get("hypothesis", "") or "").strip()
            if not text:
                return
            # 元陈述过滤:think 在"目标已达成"时可能把 meta 文本放进 hypothesis
            # 字段(如「无需新假设,目标已达成。」),那不是可验证假设,不入账本。
            if _META_HYPOTHESIS_RE.search(text):
                return
            hid = hypotheses.add(text)
            exp_id = str(execute_result.get("pid", "") or "")
            status = str(execute_result.get("experiment_status", "") or "")
            evidence = str(reflect_result.get("decision", "") or "")[:200]
            delta = self._run_metric_delta(execute_result, cycle)
            if status == "failed":
                terminal = str(execute_result.get("terminal_state", "") or "")
                ev = evidence or "experiment failed"
                if terminal and terminal not in ev:
                    ev = f"{ev} [{terminal}]"
                hypotheses.resolve(hid, "refuted", experiment_id=exp_id,
                                   evidence=ev[:200])
            elif status in ("completed", "launched") and reflect_result.get("milestone"):
                if delta is None:
                    # 无指标可比:以里程碑为准(保守 confirmed)
                    hypotheses.resolve(hid, "confirmed", experiment_id=exp_id,
                                       evidence=evidence or reflect_result.get("milestone", "")[:200])
                elif delta >= 0:
                    hypotheses.resolve(hid, "confirmed", experiment_id=exp_id,
                                       evidence=evidence or reflect_result.get("milestone", "")[:200])
                elif delta > -self._REFUTE_DELTA:
                    # 单次小幅负结果:不轻率否证(可能架构/超参混淆)
                    hypotheses.resolve(
                        hid, "inconclusive", experiment_id=exp_id,
                        evidence=(f"{evidence} [single-run small negative delta "
                                  f"{delta:+.4f}; not enough to refute — possible "
                                  f"architecture/hyperparameter confound]")[:200])
                else:
                    hypotheses.resolve(
                        hid, "refuted", experiment_id=exp_id,
                        evidence=(f"{evidence} [clear drop {delta:+.4f}]")[:200])
            elif status in ("completed", "launched"):
                hypotheses.resolve(hid, "inconclusive", experiment_id=exp_id,
                                   evidence=evidence or "no milestone")
            else:
                hypotheses.mark_testing(hid, experiment_id=exp_id)
        except Exception as exc:
            logger.warning(f"hypothesis settle failed: {exc}")

    def _record_to_ledger(self, cycle: int, think_result: dict, execute_result: dict, reflect_result: dict):
        if self.ledger is None:
            return
        metrics = execute_result.get("final_metrics") or {}
        if not isinstance(metrics, dict):
            metrics = {}
        if execute_result.get("experiment_launched"):
            status = execute_result.get("experiment_status") or "launched"
        else:
            status = think_result.get("action", "") or "no_experiment"
        terminal_state = execute_result.get("terminal_state", "")
        conclusion = reflect_result.get("milestone") or reflect_result.get("decision", "")
        if status == "failed" and terminal_state:
            conclusion = (f"[{terminal_state}] " + conclusion).strip()
        try:
            self.ledger.record(
                cycle=cycle,
                hypothesis=think_result.get("hypothesis") or think_result.get("task", ""),
                action=think_result.get("action", ""),
                status=status,
                metrics=metrics,
                pid=execute_result.get("pid"),
                log_file=execute_result.get("log_file", ""),
                conclusion=conclusion,
            )
        except Exception as exc:
            logger.warning(f"ledger record failed: {exc}")
        if self.journal is not None and reflect_result.get("milestone"):
            self.journal.append_insight(reflect_result["milestone"])

    def _consume_directive(self) -> str:
        directive_path = self.workspace / "HUMAN_DIRECTIVE.md"
        parts = []
        if directive_path.exists():
            content = directive_path.read_text(encoding="utf-8").strip()
            if content:
                archive_dir = self.workspace / "directive_archive"
                archive_dir.mkdir(exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                # Windows 下文件可能被占用（杀毒/索引），rename 失败不崩溃，
                # 直接留文件（下轮会再读）——内容已在 parts，不丢指令。
                try:
                    directive_path.rename(archive_dir / f"directive_{ts}.md")
                except OSError:
                    logger.warning("Failed to archive directive (file in use); will retry")
                logger.info(f"消费人工指令: {content[:100]}...")
                parts.append(content)
        # RESUME_DIRECTIVE.md：回退后的续训指令（rollback_handler 写入）
        resume_path = self.workspace / "RESUME_DIRECTIVE.md"
        if resume_path.exists():
            content = resume_path.read_text(encoding="utf-8").strip()
            if content:
                parts.append(f"## RESUME 续训指令（最高优先）\n{content}")
                try:
                    resume_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Failed to unlink RESUME_DIRECTIVE.md (file in use)")
                logger.info("消费续训指令")
        return "\n\n".join(parts)

    def _refresh_obsidian(self, reflect_result: dict):
        if not self.obsidian.is_enabled():
            return
        self.obsidian.refresh_dashboard(memory=self.memory, cycle_count=self._load_cycle_counter())
        self.obsidian.append_daily_entry(
            memory=self.memory, cycle_count=self._load_cycle_counter(),
            event_type="cycle_complete", reflection=reflect_result, directive="",
        )

    # ═══════════════════════════════════════════════════════════════
    # 基础设施：限速 / cooldown / 退避 / 信号处理
    # ═══════════════════════════════════════════════════════════════

    def _throttle_if_needed(self):
        if not self.max_cycles_per_hour or self.max_cycles_per_hour <= 0:
            return
        now = time.time()
        timestamps = self._load_cycle_times()
        wait = safety.seconds_until_allowed(timestamps, now, self.max_cycles_per_hour)
        if wait > 0:
            logger.warning(f"Anti-burn: {self.max_cycles_per_hour} cycles/hour; throttling {int(wait)}s")
            elapsed = 0.0
            while elapsed < wait and self._running:
                time.sleep(min(30.0, wait - elapsed))
                elapsed += 30.0
            now = time.time()
        timestamps = safety.prune_timestamps(timestamps, now)
        timestamps.append(now)
        self._save_cycle_times(timestamps)

    def _handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}. Graceful shutdown.")
        self._running = False

    # ═══════════════════════════════════════════════════════════════
    # 持久化辅助
    # ═══════════════════════════════════════════════════════════════

    def _load_cycle_counter(self) -> int:
        if self._cycle_counter_path.exists():
            try:
                return int(self._cycle_counter_path.read_text().strip())
            except ValueError:
                return 0
        return 0

    def _save_cycle_counter(self, value: int):
        self._cycle_counter_path.write_text(str(value))

    def _load_cycle_times(self) -> list:
        if self._cycle_times_path.exists():
            try:
                data = json.loads(self._cycle_times_path.read_text())
                return [float(t) for t in data] if isinstance(data, list) else []
            except (json.JSONDecodeError, ValueError, TypeError):
                return []
        return []

    def _save_cycle_times(self, timestamps: list):
        try:
            self._cycle_times_path.write_text(json.dumps(timestamps))
        except OSError as exc:
            logger.warning(f"failed to persist cycle times: {exc}")

    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _update_state(self, updates: dict):
        """原子写 state.json：临时文件 + os.replace，消除读改写并发损坏。

        写失败降级为日志（dashboard 失去实时性，但实验主流程不受影响）——
        磁盘瞬时问题不应中断节点执行。
        """
        try:
            state = self._load_state()
            state.update(updates)
            tmp = self.state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            os.replace(tmp, self.state_path)  # 原子替换
        except OSError as exc:
            logger.warning("state.json write failed (degraded, non-fatal): %s", exc)

    def _emit_event(self, type: str, phase: str = "", payload: Optional[dict] = None):
        """写事件日志（观测通道）。失败不中断主流程。"""
        if not self._journal_enabled:
            return
        try:
            self.event_log.emit(type, phase=phase, payload=payload or {}, run_id=self._run_id)
        except Exception:
            pass

    def _update_monitor_progress(self, detail: dict):
        """训练中周期更新 state.json 的进度（供 dashboard 显示）+ 事件日志。"""
        try:
            self._update_state({"phase": "monitor", **detail, "ts": time.time()})
        except Exception:
            pass
        self._emit_event("monitor_progress", phase="monitor", payload=detail)


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

# 工具结果即时代总结阈值（对齐 Claude Code tool summary：回灌前压缩，
# 不等上下文压缩阶段——减少每轮膨胀 + 模型重复读取）
_TOOL_RESULT_SUMMARY_CHARS = 800

# read_file 例外阈值：read_file 是 agent 的"眼睛"。冒烟实测 worker 因
# 结果被截断(只看得到前 800 字符)被迫用 run_shell python -c 转储文件到
# _temp_*.txt 再分块读,60 轮全耗在侦察上。≤16K 字符(≈400 行训练脚本)
# 的读取结果应整体回灌;更大文件仍截断防上下文爆炸。
_READ_FILE_SUMMARY_CHARS = 16000


def _summarize_tool_output(output: str, tool_name: str,
                           head_chars: int = _TOOL_RESULT_SUMMARY_CHARS) -> str:
    """工具结果即时代总结：超长输出保留头部关键信息 + 尾部 + 截断提示。

    对齐 Claude Code 的 tool summary（把工具调用结果压缩成更短总结，
    帮助后续轮次减少冗余上下文）。read_file 用更大的 head_chars
    （整文件可见,避免逼 agent 用 shell 转储文件）。
    """
    head = output[:head_chars]
    # 尾部保留最后 1 行（常含结果/错误关键信息）
    tail_lines = [l for l in output.splitlines() if l.strip()]
    tail = tail_lines[-1][:200] if tail_lines else ""
    # 只有真正发生截断才附加提示(输入未超阈值时不加,避免误导 agent)
    if len(output) <= head_chars:
        return output
    note = (f"\n... [tool {tool_name} result {len(output)} chars, "
            f"truncated keeping first {head_chars} chars]")
    if tail and tail not in head:
        note += f"\n[result tail] {tail}"
    return head + note


def _is_training_task(text: str) -> bool:
    """判断任务是否含训练语义（review 只审查训练类任务）。

    只用强信号词（训练/train/launch/跑模型），避免"整理实验日志"这类
    含"实验"但不含训练语义的任务误触发审查。
    """
    t = (text or "").lower()
    return any(kw in t for kw in (
        "训练", "train", "launch", "跑模型", "模型训练", "训练脚本"))


def _safe_json(raw: str) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _text_similarity(a: str, b: str) -> float:
    """简单的 Jaccard 词级相似度，用于去重。"""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _parse_json_response(raw: str) -> dict:
    # 1. 整个字符串直接 json.loads（原生支持嵌套 JSON）
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. 括号计数：找到第一个 '{'，匹配其配对的 '}'（支持嵌套），提取外层对象
    start = raw.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[start:i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break
                    break

    # 安全兜底：解析失败 → action="retry"（下轮重试，有上限防死循环），
    # 绝不把散文当执行指令，也绝不映射到 wait（wait 语义 = 永久停止，
    # 会让一次格式错误停摆整个 agent）。
    return {"action": "retry", "reason": f"JSON parse failed, skipping this cycle: {raw[:200]}"}


# ═══════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════

def main():
    # Windows GBK 控制台打印 emoji/中文混合文本(final_answer 含 ✅ 等)
    # 会抛 UnicodeEncodeError —— 统一按 UTF-8 输出,编码错误替换为 '?' 不崩溃。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="AutoResearcher — LangGraph 完整改造版")
    parser.add_argument("--project", type=str, required=True, help="项目目录路径")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件")
    parser.add_argument("--max-cycles", type=int, default=None, help="最大轮数")
    parser.add_argument("--gpu", type=str, default=None, help="GPU 设备")
    parser.add_argument("--goal", type=str, default="",
                        help="一句话研究目标(如 \"把 MNIST 训练到 99%\")。"
                             "brief 缺失时自动生成 PROJECT_BRIEF.md")
    parser.add_argument("--check", action="store_true", help="环境检查")

    args = parser.parse_args()

    if args.check:
        print("LangGraph 完整改造版环境检查:")
        print(f"  Python: {sys.version}")
        print(f"  Project: {args.project}")
        try:
            import langgraph, langchain
            print(f"  LangGraph: OK")
            print(f"  LangChain: {langchain.__version__}")
        except ImportError:
            pass
        print("  Status: OK")
        return

    import yaml
    config_path = Path(args.project) / args.config
    config = yaml.safe_load(open(config_path, encoding="utf-8")) if config_path.exists() else {}

    if args.max_cycles is not None:
        config.setdefault("agent", {})["max_cycles"] = args.max_cycles
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(args.project) / "autoresearcher_nodes.log"),
        ],
    )

    # ── 用户最小输入路径:--goal 自动生成 brief(F8 友好检查)──
    from .brief_draft import ensure_brief
    created, message = ensure_brief(Path(args.project), args.goal)
    print(f"[brief] {message}")
    if not created and not (Path(args.project) / "PROJECT_BRIEF.md").exists():
        # brief 缺失且生成失败 → 明确报错,不进入空 context 瞎跑
        print("[FATAL] 无法启动:缺少 PROJECT_BRIEF.md。"
              "请编写 brief 或检查 API key 配置(--goal 自动生成需要 LLM key)。")
        return

    try:
        graph = ResearchGraph(config=config, project_dir=args.project)
    except Exception as exc:
        # F8:启动友好检查 —— 缺 key/配置错误给出可操作提示,不堆栈崩溃
        msg = str(exc)
        if "api_key" in msg.lower() or "credentials" in msg.lower() or "auth" in msg.lower():
            print("[FATAL] LLM API key 未配置或无效。"
                  "请在 config.yaml 设置 agent.provider / agent.api_key_env,"
                  "或 export 对应环境变量(如 DEEPSEEK_API_KEY)后重试。")
        else:
            print(f"[FATAL] 启动失败: {msg}")
        return
    graph.run()


if __name__ == "__main__":
    main()
