"""retry_llm_call 上下文溢出路径的回归测试。

锁定的语义：on_context_overflow 返回「重试后的完整结果」（LLM 响应），
而不是缩减后的输入——修复前调用方误传 _trim_context(messages)，
导致上下文溢出时 last_msg 变成消息列表、被误判为最终答案。
"""
import pytest

from core.retry import retry_llm_call


class _ContextOverflowError(Exception):
    pass


def test_overflow_callback_result_is_returned():
    """on_context_overflow 的返回值 = retry_llm_call 的返回值。"""
    calls = {"invoke": 0, "trim": 0}

    def invoke():
        calls["invoke"] += 1
        raise _ContextOverflowError("maximum context length exceeded")

    def overflow():
        calls["trim"] += 1
        return {"response": "trimmed_and_retried"}  # 模拟缩减后重新调用 LLM 的结果

    result = retry_llm_call(
        invoke, max_retries=2, on_context_overflow=overflow,
        actor="test", action="overflow")
    assert result == {"response": "trimmed_and_retried"}
    assert calls["invoke"] == 1  # 原调用只失败一次
    assert calls["trim"] == 1


def test_overflow_only_once_then_exhausted():
    """overflow 回调只触发一次；再失败则正常耗尽重试。"""
    calls = {"invoke": 0, "trim": 0}

    def invoke():
        calls["invoke"] += 1
        raise _ContextOverflowError("context too long")

    def overflow():
        calls["trim"] += 1
        raise _ContextOverflowError("context too long")  # 缩减后仍溢出

    with pytest.raises(Exception) as excinfo:
        retry_llm_call(invoke, max_retries=2, on_context_overflow=overflow,
                       actor="test", action="overflow2")
    assert calls["trim"] == 1  # 不重复触发 overflow 回调
    assert "failed after" in str(excinfo.value)


def test_no_overflow_uses_normal_path():
    calls = {"invoke": 0, "trim": 0}

    def invoke():
        calls["invoke"] += 1
        return "ok"

    def overflow():  # pragma: no cover
        calls["trim"] += 1
        return "should not happen"

    result = retry_llm_call(invoke, max_retries=2, on_context_overflow=overflow,
                            actor="test", action="normal")
    assert result == "ok"
    assert calls["trim"] == 0
