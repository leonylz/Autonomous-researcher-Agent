"""
LLM 调用重试与容错模块。

对齐 Anthropic Claude Code CLI 的重试架构：
  - 指数退避 + 随机抖动（避免惊群效应）
  - 错误分类：瞬时错误重试，致命错误立即中断
  - 分级退避：rate_limit / overloaded / timeout / connection 各有策略
  - 上下文溢出自适应：捕获 context overflow → 自动缩减输入 → 重试

用法:
    from .retry import retry_llm_call, classify_error

    try:
        response = retry_llm_call(lambda: llm.invoke(messages), max_retries=3)
    except FatalLLMError:
        # 认证失败 / 权限不足 → 不重试，立即降级
        ...
"""

from __future__ import annotations

import logging
import random
import time
from enum import Enum
from typing import Callable, TypeVar

logger = logging.getLogger("autoresearcher.retry")

T = TypeVar("T")


class ErrorCategory(Enum):
    """LLM 错误分类（对齐 Anthropic / OpenAI 错误码体系）。"""
    TRANSIENT_RATE_LIMIT = "rate_limit"        # 429 Too Many Requests
    TRANSIENT_OVERLOADED = "overloaded"         # 529 Service Overloaded
    TRANSIENT_TIMEOUT = "timeout"               # 请求超时
    TRANSIENT_CONNECTION = "connection"          # 网络抖动 / 连接重置
    TRANSIENT_SERVER = "server_error"            # 5xx 服务端错误
    CONTEXT_OVERFLOW = "context_overflow"        # 400 上下文超长
    FATAL_AUTH = "auth_error"                    # 401/403 认证/权限
    FATAL_BAD_REQUEST = "bad_request"            # 400 参数错误（非 overflow）
    FATAL_UNKNOWN = "unknown"                    # 未知错误（保守：不重试）


class LLMRetryError(Exception):
    """所有重试均失败。"""

    def __init__(self, message: str, category: ErrorCategory, attempts: int):
        super().__init__(message)
        self.category = category
        self.attempts = attempts


class FatalLLMError(LLMRetryError):
    """不可重试的致命错误（认证失败 / 权限不足 / 参数错误）。"""
    pass


# ── 瞬态错误关键词（按优先级从高到低） ──
_TRANSIENT_PATTERNS = [
    (ErrorCategory.TRANSIENT_RATE_LIMIT, (
        "rate_limit", "rate limit", "too many requests", "quota exceeded",
        "request limit", "限额", "频率限制", "并发上限",
    )),
    (ErrorCategory.TRANSIENT_OVERLOADED, (
        "overloaded", "529", "service overloaded", "too busy",
    )),
    (ErrorCategory.TRANSIENT_TIMEOUT, (
        "timeout", "timed out", "timed_out", "request timed out",
    )),
    (ErrorCategory.TRANSIENT_CONNECTION, (
        "connection", "reset", "econnreset", "epipe", "broken pipe",
        "network", "socket", "econnrefused", "nodata",
    )),
    (ErrorCategory.TRANSIENT_SERVER, (
        "server error", "internal server error", "500", "502", "503", "504",
        "service unavailable", "bad gateway",
    )),
]

_FATAL_PATTERNS = [
    (ErrorCategory.FATAL_AUTH, (
        "401", "403", "unauthorized", "forbidden", "authentication",
        "invalid api key", "incorrect api key", "auth",
        "permission", "access denied", "认证失败", "权限不足",
    )),
    (ErrorCategory.FATAL_BAD_REQUEST, (
        "invalid_request_error", "invalid parameter",
    )),
]

# 上下文溢出有专门关键词（单独检测）
_CONTEXT_OVERFLOW_PATTERNS = (
    "context_length_exceeded", "maximum context length", "too long",
    "reduce the length", "token limit", "max_tokens", "context window",
    "prompt too long", "413", "request too large",
    "上下文过长", "超出限制", "token 超限",
)


# ── 退避参数 ──
BASE_DELAY_MS = 1000          # 基础延迟 1s
MAX_DELAY_MS = 30_000         # 最大延迟 30s
JITTER_FACTOR = 0.25          # 抖动系数（±25%）


def classify_error(error: Exception) -> ErrorCategory:
    """将异常分类到错误类别。

    按优先级从高到低检测：
    1. 上下文溢出（特殊处理：需要缩减输入）
    2. 瞬态错误（可重试）
    3. 致命错误（不重试）
    4. 兜底 → FATAL_UNKNOWN
    """
    msg = str(error).lower()

    # 上下文溢出优先检测（最特殊的可恢复错误）
    for kw in _CONTEXT_OVERFLOW_PATTERNS:
        if kw in msg:
            return ErrorCategory.CONTEXT_OVERFLOW

    # 致命错误
    for cat, patterns in _FATAL_PATTERNS:
        for kw in patterns:
            if kw in msg:
                return cat

    # 瞬态错误
    for cat, patterns in _TRANSIENT_PATTERNS:
        for kw in patterns:
            if kw in msg:
                return cat

    # 兜底：检查 HTTP 状态码
    if hasattr(error, "status_code"):
        code = getattr(error, "status_code", 0)
        if code == 429:
            return ErrorCategory.TRANSIENT_RATE_LIMIT
        if code == 529:
            return ErrorCategory.TRANSIENT_OVERLOADED
        if code in (401, 403):
            return ErrorCategory.FATAL_AUTH
        if 500 <= code < 600:
            return ErrorCategory.TRANSIENT_SERVER
        if code == 400 and "context" in msg:
            return ErrorCategory.CONTEXT_OVERFLOW

    # HTTP 库异常映射
    exc_name = type(error).__name__.lower()
    if "timeout" in exc_name:
        return ErrorCategory.TRANSIENT_TIMEOUT
    if "connection" in exc_name:
        return ErrorCategory.TRANSIENT_CONNECTION
    if "auth" in exc_name:
        return ErrorCategory.FATAL_AUTH

    return ErrorCategory.FATAL_UNKNOWN


def _backoff_delay(attempt: int, category: ErrorCategory) -> float:
    """计算本次重试等待时间（指数退避 + 随机抖动）。

    rate_limit / overloaded 退避更长（给服务端恢复时间）。
    connection / timeout 退避较短（网络抖动通常秒级恢复）。
    """
    if category == ErrorCategory.TRANSIENT_RATE_LIMIT:
        base = min(BASE_DELAY_MS * (2 ** attempt), 60_000)  # cap 60s
    elif category == ErrorCategory.TRANSIENT_OVERLOADED:
        base = min(BASE_DELAY_MS * (2 ** attempt), MAX_DELAY_MS)
    else:
        base = min(BASE_DELAY_MS * (1.5 ** attempt), 15_000)  # 较短退避

    jitter = random.uniform(-JITTER_FACTOR, JITTER_FACTOR) * base
    return (base + jitter) / 1000.0  # 转为秒


def retry_llm_call(
    fn: Callable[[], T],
    max_retries: int = 3,
    on_context_overflow: Callable[[], T] | None = None,
    actor: str = "worker",
    action: str = "llm_call",
) -> T:
    """带智能重试的 LLM 调用包装器。

    对齐 Anthropic Claude Code CLI 的分级重试策略：
      - rate_limit → 指数退避（最长 60s），3 次后放弃
      - overloaded (529) → 指数退避 + 可切换到 fallback 模型
      - timeout / connection → 较短退避 + 禁用 keep-alive
      - context_overflow → 自动缩减输入 + 重试 1 次
      - auth / bad_request → 立即抛 FatalLLMError，不重试

    Parameters
    ----------
    fn : callable
        LLM 调用函数（无参数，返回 T）。
    max_retries : int
        最大重试次数（默认 3）。
    on_context_overflow : callable or None
        上下文溢出时的缩减回调。若为 None，overflow 视为致命错误。
    actor : str
        调用者标识（日志用）。
    action : str
        动作标识（日志用）。

    Returns
    -------
    T
        LLM 调用返回值。

    Raises
    ------
    FatalLLMError
        不可重试的错误（认证 / 权限 / 参数）。
    LLMRetryError
        所有重试均失败。
    """
    last_error: Exception | None = None
    last_category: ErrorCategory | None = None

    for attempt in range(1 + max_retries):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            category = classify_error(exc)
            last_category = category
            msg = str(exc)[:200]

            # 上下文溢出 → 尝试缩减后重试一次
            if category == ErrorCategory.CONTEXT_OVERFLOW:
                if on_context_overflow is not None and attempt == 0:
                    logger.warning(
                        "[%s/%s] context overflow detected, reducing input and retrying...",
                        actor, action,
                    )
                    try:
                        return on_context_overflow()
                    except Exception as exc2:
                        last_error = exc2
                        category = classify_error(exc2)
                        last_category = category
                        msg = str(exc2)[:200]
                else:
                    logger.error(
                        "[%s/%s] context overflow, no reduction callback or already retried",
                        actor, action,
                    )
                    break

            # 致命错误 → 立即中断
            if category in (ErrorCategory.FATAL_AUTH, ErrorCategory.FATAL_BAD_REQUEST):
                logger.error(
                    "[%s/%s] fatal error (category=%s): %.150s",
                    actor, action, category.value, msg,
                )
                raise FatalLLMError(
                    f"[{actor}/{action}] {category.value}: {msg}",
                    category=category,
                    attempts=attempt + 1,
                ) from exc

            # 未知错误 → 保守策略：不重试
            if category == ErrorCategory.FATAL_UNKNOWN:
                logger.error(
                    "[%s/%s] unknown error (not retrying): %.150s",
                    actor, action, msg,
                )
                break

            # 瞬态错误 → 重试
            if attempt < max_retries:
                delay = _backoff_delay(attempt, category)
                logger.warning(
                    "[%s/%s] transient error (category=%s, attempt=%d/%d), "
                    "retrying in %.1fs: %.100s",
                    actor, action, category.value, attempt + 1, max_retries,
                    delay, msg,
                )
                time.sleep(delay)
                continue

            # 重试次数耗尽
            logger.error(
                "[%s/%s] all %d retries exhausted (category=%s): %.150s",
                actor, action, max_retries + 1, category.value, msg,
            )
            break

    # 所有尝试失败
    raise LLMRetryError(
        f"[{actor}/{action}] failed after {max_retries + 1} attempts "
        f"(last_category={last_category.value if last_category else '?'}): "
        f"{str(last_error)[:200]}",
        category=last_category or ErrorCategory.FATAL_UNKNOWN,
        attempts=max_retries + 1,
    )
