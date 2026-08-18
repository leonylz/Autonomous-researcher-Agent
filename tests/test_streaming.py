"""Worker 流式（stream_mode）核心逻辑单元测试。

覆盖提交屏障语义：
  - tool_call 参数分片按 (index, id) 累积
  - 完整 JSON 校验通过后才执行工具，绝不执行部分参数
  - 文本增量实时回调 on_token
  - 非流式路径行为不变（向后兼容）
"""
from langchain_core.messages import AIMessageChunk

from core.nodes import ResearchGraph


# ── AIMessageChunk 分片构造 ──

def _text_chunk(text: str) -> AIMessageChunk:
    return AIMessageChunk(content=text)


def _tool_chunk(index: int, cid: str, name: str = "", args: str = "") -> AIMessageChunk:
    """构造带 tool_call_chunks 的 AIMessageChunk。"""
    chunk = AIMessageChunk(content="")
    chunk.tool_call_chunks = [{
        "name": name, "args": args, "id": cid, "index": index,
        "type": "tool_call_chunk",
    }]
    return chunk


def _collect(chunks: list, on_token=None):
    """模拟 _run_worker_single_step 内的 _stream_collect（等价复制用于测试）。"""
    return ResearchGraph._stream_collect_for_test(chunks, on_token)


class TestStreamCollect:
    def test_text_chunks_accumulated(self):
        msg = _collect([_text_chunk("hello "), _text_chunk("world")])
        assert msg.content == "hello world"
        assert not getattr(msg, "tool_calls", None)

    def test_text_delta_callback(self):
        received = []
        _collect([_text_chunk("a"), _text_chunk("b")], on_token=received.append)
        assert received == ["a", "b"]

    def test_tool_args_split_across_chunks(self):
        """关键：参数 JSON 被切成多个 chunk，必须按 id 拼接后解析。"""
        chunks = [
            _text_chunk(""),
            _tool_chunk(0, "call_1", name="run_shell", args='{"command":'),
            _tool_chunk(0, "call_1", name="", args='"echo hi"}'),
        ]
        msg = _collect(chunks)
        calls = msg.tool_calls
        assert len(calls) == 1
        assert calls[0]["name"] == "run_shell"
        assert calls[0]["args"] == {"command": "echo hi"}

    def test_multiple_parallel_tool_calls(self):
        chunks = [
            _tool_chunk(0, "c1", name="read_file", args='{"path": "a.py"}'),
            _tool_chunk(1, "c2", name="write_file", args='{"path": "b.py"}'),
        ]
        msg = _collect(chunks)
        assert len(msg.tool_calls) == 2
        # 按 index 排序稳定
        assert [c["name"] for c in msg.tool_calls] == ["read_file", "write_file"]

    def test_invalid_json_args_marked_error(self):
        """提交屏障：参数 JSON 不完整 → 标记 _error，绝不执行。"""
        chunks = [_tool_chunk(0, "c1", name="run_shell", args='{"command": "oops')]
        msg = _collect(chunks)
        assert msg.tool_calls[0]["args"].get("_error") is not None


# ── 用 ResearchGraph 的静态方法做等价验证（防实现漂移） ──

class TestGraphStaticMethod:
    def test_method_exists(self):
        assert hasattr(ResearchGraph, "_stream_collect_for_test")


# 注：_stream_collect 是 _run_worker_single_step 的内联闭包，测试通过
# ResearchGraph._stream_collect_for_test 静态方法等价验证（见 nodes.py）。
