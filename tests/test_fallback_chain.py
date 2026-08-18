"""降级链测试:主模型全失败 → fallback 模型接管 → 结构化降级 JSON(绝不抛异常)。"""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.nodes import ResearchGraph


def _mk_graph(fallback=None) -> ResearchGraph:
    g = object.__new__(ResearchGraph)
    g._fallback_llm = fallback
    g._fallback_failures = 0
    g.bad_case_collector = MagicMock()
    g.cost_tracker = MagicMock()
    g.event_log = MagicMock()
    return g


def _err(msg="rate limit exceeded"):
    exc = Exception(msg)
    return exc


class FallbackChainTests(unittest.TestCase):
    def test_primary_failure_falls_back_to_model(self):
        """主模型一直失败 → fallback 接管,返回其内容且标记 degraded。"""
        failing = MagicMock()
        failing.invoke.side_effect = _err  # 每次都抛
        fallback = MagicMock()
        fallback.invoke.return_value = SimpleNamespace(
            content="fallback answer",
            response_metadata={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        )
        fallback.model_name = "cheap-model"

        g = _mk_graph(fallback=fallback)
        with patch("time.sleep"):  # 指数退避不真睡
            text, degraded, usage = g._safe_llm_call(
                failing, "sys", [{"role": "user", "content": "hi"}],
                actor="leader", action="think", max_retries=2,
            )

        self.assertEqual(text, "fallback answer")
        self.assertTrue(degraded)
        self.assertEqual(usage["completion_tokens"], 5)
        # 成本按 fallback 模型记录
        g.cost_tracker.record_call.assert_called_once()
        args = g.cost_tracker.record_call.call_args
        self.assertEqual(args.kwargs["model"], "cheap-model")
        self.assertEqual(args.kwargs["action"], "think:fallback")

    def test_no_fallback_returns_structured_degraded_json(self):
        """未配置 fallback → 结构化降级 JSON,不抛异常。"""
        failing = MagicMock()
        failing.invoke.side_effect = _err
        failing.model_name = "main"

        g = _mk_graph(fallback=None)
        with patch("time.sleep"):
            text, degraded, usage = g._safe_llm_call(
                failing, "sys", [], actor="leader", action="think", max_retries=1,
            )

        self.assertTrue(degraded)
        self.assertIn("llm_unavailable", text)

    def test_fallback_also_fails_returns_degraded_json(self):
        """fallback 也失败 → 结构化降级 JSON,且失败计数递增。"""
        failing = MagicMock()
        failing.invoke.side_effect = _err
        fallback = MagicMock()
        fallback.invoke.side_effect = _err
        fallback.model_name = "cheap"

        g = _mk_graph(fallback=fallback)
        with patch("time.sleep"):
            text, degraded, _ = g._safe_llm_call(
                failing, "sys", [], actor="code", action="execute", max_retries=1,
            )

        self.assertTrue(degraded)
        self.assertIn("llm_unavailable", text)
        self.assertEqual(g._fallback_failures, 1)

    def test_transient_error_still_retries_primary_first(self):
        """瞬时错误先重试主模型,重试耗尽后才轮 fallback(不浪费便宜模型)。"""
        primary = MagicMock()
        calls = {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise _err("429 too many requests")
            return SimpleNamespace(content="primary ok", response_metadata={})

        primary.invoke.side_effect = flaky
        fallback = MagicMock()
        fallback.invoke.return_value = SimpleNamespace(content="fallback", response_metadata={})

        g = _mk_graph(fallback=fallback)
        with patch("time.sleep"):
            text, degraded, _ = g._safe_llm_call(
                primary, "sys", [], actor="leader", action="think", max_retries=3,
            )

        self.assertEqual(text, "primary ok")
        self.assertFalse(degraded)
        fallback.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
