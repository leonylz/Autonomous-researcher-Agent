"""
输入输出护栏。

在 LLM 调用前后做安全检查：
  - InputGuard：PII 脱敏 + prompt injection 检测
  - OutputGuard：敏感信息泄露检测 + 危险输出拦截

面试价值：AI 安全是 2025-2026 最热的话题。
"""

from __future__ import annotations

import re
import logging
from typing import Optional

logger = logging.getLogger("autoresearcher.guardrails")

# Prompt injection 特征模式
INJECTION_PATTERNS = [
    r"ignore (all )?(previous|above|prior) (instructions?|prompts?|rules?)",
    r"forget (everything|all) (you know|you've learned|your training)",
    r"you are now (a |an )?(DAN|jailbreak|evil|unrestricted)",
    r"pretend (you are|to be) (a |an )?",
    r"act as (if )?(you are )?(a |an )?(unethical|evil|malicious)",
    r"bypass (your |the )?(safety|content|security) (filter|guard|check)",
    r"system:\s*you are",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
]

# PII 脱敏规则
PII_PATTERNS = {
    "email": (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL]"),
    "phone_cn": (r"1[3-9]\d{9}", "[PHONE]"),
    "ip": (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP]"),
    "api_key": (r"(?:sk|api[_-]?key|token)[=:]\s*['\"]?[\w-]{20,}['\"]?", "[API_KEY]"),
}

# 输出敏感模式
OUTPUT_SENSITIVE_PATTERNS = {
    "api_key_leak": r"(?:sk-ant-|sk-proj-|sk-)[a-zA-Z0-9_-]{20,}",
    "system_path": r"(?:/etc/(?:passwd|shadow)|C:\\Windows\\System32|/root/\.ssh)",
    "sql_injection": r"(?:DROP\s+TABLE|DELETE\s+FROM\s+\w+\s+WHERE|INSERT\s+INTO\s+\w+\s+VALUES)",
    "shell_danger": r"(?:rm\s+-rf\s+/|mkfs\.|dd\s+if=|>/\dev/\w+)",
}


class InputGuard:
    """输入护栏：检查进入 LLM 的文本。"""

    @staticmethod
    def detect_injection(text: str) -> tuple[bool, str]:
        """检测 prompt injection。返回 (is_attack, reason)。"""
        text_lower = text.lower()
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, text_lower):
                return True, f"injection pattern matched: {pattern[:60]}"
        return False, ""

    @staticmethod
    def sanitize(text: str) -> str:
        """脱敏 PII。"""
        for _name, (pattern, replacement) in PII_PATTERNS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    @classmethod
    def validate(cls, text: str, strict: bool = False) -> tuple[bool, str, str]:
        """完整输入校验。返回 (is_safe, reason, sanitized_text)。"""
        sanitized = cls.sanitize(text)
        is_attack, reason = cls.detect_injection(sanitized)
        if is_attack and strict:
            return False, f"BLOCKED: {reason}", sanitized
        if is_attack:
            logger.warning(f"Potential injection detected (non-strict, sanitizing): {reason}")
        return True, "", sanitized


class OutputGuard:
    """输出护栏：检查 LLM 返回的文本。"""

    @staticmethod
    def validate(response: str) -> tuple[bool, list[str]]:
        """检查 LLM 输出是否包含敏感信息。返回 (is_safe, violations)。"""
        violations = []
        for name, pattern in OUTPUT_SENSITIVE_PATTERNS.items():
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                violations.append(f"{name}: matched {len(matches)} pattern(s)")
        if violations:
            logger.warning(f"Output guard violations: {violations}")
        return len(violations) == 0, violations

    @staticmethod
    def redact(response: str) -> str:
        """直接从输出中删除敏感内容。"""
        for _name, pattern in OUTPUT_SENSITIVE_PATTERNS.items():
            response = re.sub(pattern, "[REDACTED]", response, flags=re.IGNORECASE)
        return response
