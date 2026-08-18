"""
沙箱策略层 — 工具权限分级 + 敏感环境变量剥离。

与"规则沙箱"正交(机制层 vs 策略层):
- 规则沙箱(已有):无 shell 执行、路径越界拦截、危险命令黑名单、符号链接防泄漏
- 策略沙箱(本模块):工具在什么权限级别下可用 + 子进程拿到什么环境

权限分级(对齐主流 agent 沙箱模型):
- read-only        只读工具可用;write_file / run_shell / launch_experiment 全部拒绝
- workspace-write  默认;读写工作区 + 受限命令执行
- full             全开(信任场景/调试)

环境剥离:
run_shell 子进程默认继承完整环境 —— agent 的 API key 就在 env 里,黑名单
只拦 `echo $KEY` 这类形式,`python -c` 等有绕过面。sanitize_env() 从子进程
环境里剥离 *API_KEY*/*TOKEN*/*SECRET*/*PASSWORD* 模式的变量;确需传递时
通过 sandbox.keep_env 显式放行(训练脚本一般不需要 key)。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("autoresearcher.sandbox")

VALID_MODES = ("read-only", "workspace-write", "full")


class Sandbox:
    """工具执行沙箱策略(纯策略,无 IO;挂接在工具层执行)。"""

    def __init__(self, mode: str = "workspace-write",
                 keep_env: Optional[list] = None):
        if mode not in VALID_MODES:
            raise ValueError(
                f"Unknown sandbox mode: {mode!r} — expected one of {', '.join(VALID_MODES)}"
            )
        self.mode = mode
        self.keep_env = set(keep_env or [])

    # ── 权限判断 ──

    @property
    def allow_write(self) -> bool:
        return self.mode != "read-only"

    @property
    def allow_exec(self) -> bool:
        return self.mode != "read-only"

    @property
    def allow_full(self) -> bool:
        return self.mode == "full"

    def reject_reason(self) -> Optional[str]:
        if self.mode == "read-only":
            return "sandbox read-only mode: operation denied by policy"
        return None

    # ── 环境剥离 ──

    def environment(self, base_env: Optional[dict] = None) -> dict:
        """返回子进程可用的环境副本。

        复用 execution._scrub_env 的硬剥离(剔除 KEY/TOKEN/SECRET/PASSWORD/AUTH
        变量、保留 PATH/HOME 等必需变量),再按 keep_env 显式放行。
        注意:_scrub_env 从 os.environ 读取,base_env 参数保留给未来注入。
        """
        from .execution import _scrub_env

        env = _scrub_env()
        for name in self.keep_env:
            if name in os.environ and name not in env:
                env[name] = os.environ[name]
        return env

    def __repr__(self) -> str:  # pragma: no cover
        return f"Sandbox(mode={self.mode!r}, keep_env={sorted(self.keep_env)})"


def resolve_sandbox(cfg: Optional[dict]) -> Sandbox:
    """从 config['sandbox'] 解析沙箱策略(未知模式 fail-fast,与 provider 一致)。"""
    cfg = cfg or {}
    return Sandbox(
        mode=cfg.get("mode", "workspace-write"),
        keep_env=cfg.get("keep_env", []),
    )
