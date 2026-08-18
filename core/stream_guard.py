"""
流式输出安全网关。

在每个 SSE chunk 到达时做增量正则匹配。
一旦命中敏感规则，立即发送 [CONTENT_BLOCKED] 并关闭流。

面试价值：流式场景下的实时内容安全是 2025-2026 最热门的话题之一。
"""

from __future__ import annotations

import re
import logging
from typing import Iterator, Optional

logger = logging.getLogger("autoresearcher.stream_guard")

# 默认敏感规则
DEFAULT_BLOCKED_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("api_key_leak", re.compile(r"sk-(?:ant|proj)-[a-zA-Z0-9_-]{20,}", re.IGNORECASE)),
    ("system_path", re.compile(r"(?:/etc/(?:passwd|shadow)|C:\\Windows\\System32)", re.IGNORECASE)),
    ("pii_phone_cn", re.compile(r"1[3-9]\d{9}")),
    ("pii_email", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    ("shell_danger", re.compile(r"(?:rm\s+-rf\s+/|mkfs\.|dd\s+if=)", re.IGNORECASE)),
    ("sql_danger", re.compile(r"(?:DROP\s+TABLE|DELETE\s+FROM\s+\w+\s+WHERE)", re.IGNORECASE)),
]


class StreamOutputGuard:
    """流式输出安全网关。

    用法:
        guard = StreamOutputGuard()
        for chunk in llm.stream(messages):
            safe, ok = guard.feed(chunk.content)
            if safe:
                yield safe
            if not ok:
                break
        # 流结束后
        final = guard.finalize()
    """

    def __init__(self, blocked_patterns: Optional[list[tuple[str, re.Pattern]]] = None):
        self._patterns = blocked_patterns or DEFAULT_BLOCKED_PATTERNS
        self._accumulator = ""
        self._blocked = False
        self._block_reason = ""
        self._safe_prefix = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, chunk: str) -> tuple[str, bool]:
        """
        喂入一个 SSE chunk。

        Returns
        -------
        (output_chunk, should_continue)
            如果 should_continue=False，流已因安全原因被阻断。
            output_chunk 是应发送给用户的安全内容（可能包含阻断通知）。
        """
        if self._blocked:
            return "", False

        self._accumulator += chunk

        # 增量检查：只扫描最近 300 字符的滑动窗口
        check_window = self._accumulator[-300:]
        for rule_name, pattern in self._patterns:
            match = pattern.search(check_window)
            if match:
                self._blocked = True
                self._block_reason = rule_name
                self._safe_prefix = self._accumulator[: len(self._accumulator) - 300 + match.start()]

                logger.warning(
                    "Stream blocked by rule '%s': matched '%s'",
                    rule_name, match.group()[:60],
                )

                # 返回安全前缀 + 阻断通知
                blocked_msg = (
                    f"\n\n[CONTENT_BLOCKED by safety gateway: {rule_name}]"
                )
                return blocked_msg, False

        return chunk, True

    def finalize(self) -> str:
        """流结束时的最终检查。返回完整的安全输出。"""
        if self._blocked:
            return self._safe_prefix + f"\n[BLOCKED: {self._block_reason}]"
        return self._accumulator

    @property
    def was_blocked(self) -> bool:
        return self._blocked

    @property
    def block_reason(self) -> str:
        return self._block_reason

    @property
    def accumulated(self) -> str:
        """获取当前累积的安全内容。"""
        if self._blocked:
            return self._safe_prefix
        return self._accumulator


# ═══════════════════════════════════════════════════════════════════
# 批量输出（非流式）安全检查
# ═══════════════════════════════════════════════════════════════════

def scan_full_output(response: str, patterns: Optional[list[tuple[str, re.Pattern]]] = None) -> tuple[str, list[str]]:
    """
    对完整响应做安全扫描。用于非流式场景（和 guardrails.py 互补）。

    Returns
    -------
    (safe_response, violations)
        safe_response 是已 redact 的版本。
        violations 是命中的规则名列表。
    """
    patterns = patterns or DEFAULT_BLOCKED_PATTERNS
    violations = []
    redacted = response

    for rule_name, pattern in patterns:
        matches = pattern.findall(redacted)
        if matches:
            violations.append(f"{rule_name}: {len(matches)} match(es)")
            redacted = pattern.sub("[REDACTED]", redacted)

    return redacted, violations
