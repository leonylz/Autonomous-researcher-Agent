"""
LLM 调用降级 + 熔断器。

处理 API 限流、服务过载、网络超时等问题：

 正常：调用 primary model
 429 限流 → 切 fallback 模型
 529 过载 → 同 429 逻辑
 连续失败 ≥ threshold → 熔断器打开，所有请求走缓存
 彻底挂了 → 返回结构化错误（不是裸 Exception 堆栈）

面试价值：高可用 LLM 调用的标准模式（和微服务熔断器同源）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger("autoresearcher.degraded")


class CircuitBreaker:
    """熔断器。

    状态机：CLOSED → OPEN → HALF_OPEN → CLOSED
    """

    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._open_until = 0.0

    @property
    def is_open(self) -> bool:
        """熔断器是否打开。"""
        if self._open_until > time.time():
            return True
        # HALF_OPEN 状态：超时后允许一次试探
        return False

    def record_success(self):
        """记录一次成功 → 关闭熔断器。"""
        self._failure_count = 0
        self._open_until = 0.0

    def record_failure(self) -> float:
        """记录一次失败 → 返回需要等待的秒数。"""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self.failure_threshold:
            backoff = min(self.reset_timeout * (2 ** (self._failure_count - self.failure_threshold)), 1800.0)
            self._open_until = time.time() + backoff
            logger.warning(
                "Circuit breaker OPEN for %.0fs after %d consecutive failures",
                backoff, self._failure_count,
            )
            return backoff
        return 0.0

    @property
    def retry_after(self) -> float:
        """距离熔断器关闭还需多少秒。"""
        return max(0.0, self._open_until - time.time())


class DegradedLLM:
    """带降级策略的 LLM 调用器。

    用法:
        degraded = DegradedLLM(
            primary=ChatOpenAI(model="qwen-plus", ...),
            fallback=ChatOpenAI(model="qwen-turbo", ...),  # 更便宜的
            cache_ttl=300,
        )
        response, degraded = degraded.call(system_prompt, messages)
        if degraded:
            logger.warning("Using degraded LLM response")
    """

    def __init__(self, primary_llm, fallback_llm=None, *,
                 cache_ttl: float = 300.0,
                 circuit_threshold: int = 3,
                 circuit_reset: float = 60.0,
                 max_retries_per_model: int = 2,
                 name: str = "llm",
                 ):
        self.primary = primary_llm
        self.fallback = fallback_llm
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries_per_model
        self.name = name

        self._cache: dict[str, tuple[float, str]] = {}
        self._breaker = CircuitBreaker(
            failure_threshold=circuit_threshold,
            reset_timeout=circuit_reset,
        )
        self._stats = {
            "total_calls": 0,
            "primary_success": 0,
            "fallback_used": 0,
            "cache_hits": 0,
            "circuit_open": 0,
            "total_failures": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(self, system: str, messages: list) -> tuple[str, bool]:
        """
        调用 LLM，自动处理限流/过载/熔断。

        Returns
        -------
        (response_text, degraded)
            degraded=True 表示使用了降级结果（fallback 模型或缓存）。
        """
        self._stats["total_calls"] += 1

        cache_key = _make_cache_key(system, messages)

        # 熔断器检查
        if self._breaker.is_open:
            self._stats["circuit_open"] += 1
            cached = self._get_cache(cache_key)
            if cached:
                self._stats["cache_hits"] += 1
                logger.warning(
                    "[%s] Circuit open (retry in %.0fs) — returning cached result",
                    self.name, self._breaker.retry_after,
                )
                return cached, True
            # 没有缓存：返回结构化降级信息
            return self._degraded_error("circuit_open", self._breaker.retry_after), True

        # 尝试主模型
        models_to_try = [m for m in (self.primary, self.fallback) if m is not None]
        if not models_to_try:
            return self._degraded_error("no_models_available", 0), True

        last_error = ""
        for model_idx, llm in enumerate(models_to_try):
            is_fallback = model_idx > 0
            if is_fallback:
                self._stats["fallback_used"] += 1

            for attempt in range(self.max_retries):
                try:
                    response = str(
                        llm.invoke([
                            SystemMessage(content=system),
                            HumanMessage(
                                content=messages[-1].content
                                if hasattr(messages[-1], 'content')
                                else str(messages[-1])
                            ),
                        ]).content
                    )

                    # 成功
                    self._breaker.record_success()
                    if not is_fallback:
                        self._stats["primary_success"] += 1
                    self._set_cache(cache_key, response)
                    return response, is_fallback

                except Exception as exc:
                    last_error = str(exc)
                    msg_lower = last_error.lower()

                    # 判断错误类型
                    is_rate_limit = any(
                        kw in msg_lower
                        for kw in ("rate_limit", "429", "too many requests", "quota")
                    )
                    is_overload = any(
                        kw in msg_lower
                        for kw in ("overloaded", "529", "server error", "internal error")
                    )
                    is_transient = is_rate_limit or is_overload or any(
                        kw in msg_lower
                        for kw in ("timeout", "connection", "reset", "temporarily")
                    )

                    model_label = getattr(llm, 'model_name', f"model_{model_idx}")
                    logger.warning(
                        "[%s] %s attempt %d/%d: %s",
                        self.name, model_label, attempt + 1, self.max_retries,
                        last_error[:120],
                    )

                    if is_transient and attempt < self.max_retries - 1:
                        wait = 2 ** attempt  # 指数退避: 1s, 2s, 4s
                        logger.info("[%s] Retrying in %ds...", self.name, wait)
                        time.sleep(wait)
                        continue

                    if is_transient:
                        # 此模型的重试用完，尝试下一个模型
                        logger.warning(
                            "[%s] %s exhausted retries, trying next model...",
                            self.name, model_label,
                        )
                        break
                    else:
                        # 非瞬态错误（如 400 Bad Request）→ 不重试，直接换模型
                        break

        # 所有模型都失败 → 熔断
        self._stats["total_failures"] += 1
        backoff = self._breaker.record_failure()

        # 尝试缓存
        cached = self._get_cache(cache_key)
        if cached:
            self._stats["cache_hits"] += 1
            logger.warning("[%s] All models failed — returning stale cache", self.name)
            return cached, True

        logger.error("[%s] All models exhausted, no cache available", self.name)
        return self._degraded_error("all_models_failed", backoff), True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _degraded_error(self, reason: str, retry_after: float) -> str:
        return json.dumps({
            "error": reason,
            "retry_after": round(retry_after, 0),
            "action": "wait",
            "reason": "All LLM models temporarily unavailable. Will retry after backoff.",
        }, ensure_ascii=False)

    def _get_cache(self, key: str) -> Optional[str]:
        if key in self._cache:
            ts, val = self._cache[key]
            if time.time() - ts < self.cache_ttl:
                return val
            del self._cache[key]
        return None

    def _set_cache(self, key: str, val: str):
        self._cache[key] = (time.time(), val)
        # 防止缓存无限增长
        if len(self._cache) > 100:
            oldest = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest]

    @property
    def stats(self) -> dict:
        """返回调用统计。"""
        return {
            **self._stats,
            "circuit_state": "open" if self._breaker.is_open else "closed",
            "retry_after": round(self._breaker.retry_after, 1),
        }


# ═══════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════

def _make_cache_key(system: str, messages: list) -> str:
    """为 (system_prompt, messages) 生成缓存 key。"""
    raw = system + "|||" + json.dumps(
        [
            {"role": getattr(m, "type", "unknown"), "content": str(getattr(m, "content", m))}
            for m in messages[-3:]  # 只用最后 3 条消息做 key
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()
