"""MCP 适配层协议测试:JSON-RPC 形状、工具发现、工具执行、错误处理。"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.execution import LocalExecutionBackend  # noqa: E402
from core.mcp_server import MCPServer  # noqa: E402
from core.nodes import set_tool_context  # noqa: E402
from core.sandbox import Sandbox  # noqa: E402


class MCPServerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name) / "workspace"
        self.workspace.mkdir()
        set_tool_context(self.workspace, LocalExecutionBackend(self.workspace))
        self.server = MCPServer(workspace=self.workspace)

    def tearDown(self):
        self.tempdir.cleanup()

    def _call(self, method, params=None, request_id=1):
        msg = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            msg["params"] = params
        resp = json.loads(self.server.handle_line(json.dumps(msg)))
        return resp

    # ── 握手与发现 ──

    def test_initialize_handshake(self):
        resp = self._call("initialize", {"protocolVersion": "2025-03-26",
                                         "capabilities": {}, "clientInfo": {}})
        self.assertNotIn("error", resp)
        self.assertEqual(resp["result"]["protocolVersion"], "2025-03-26")
        self.assertIn("tools", resp["result"]["capabilities"])

    def test_tools_list_exposes_schemas(self):
        resp = self._call("tools/list")
        tools = {t["name"]: t for t in resp["result"]["tools"]}
        self.assertIn("read_file", tools)
        self.assertIn("launch_experiment", tools)
        self.assertIn("search_arxiv", tools)
        self.assertIn("inputSchema", tools["read_file"])
        self.assertIn("properties", tools["read_file"]["inputSchema"])

    def test_notification_gets_no_response(self):
        self.assertIsNone(
            self.server.handle_line(
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
            )
        )

    def test_unknown_method_returns_error(self):
        resp = self._call("bogus/method")
        self.assertEqual(resp["error"]["code"], -32601)

    def test_malformed_json_returns_parse_error(self):
        resp = json.loads(self.server.handle_line("{not json"))
        self.assertEqual(resp["error"]["code"], -32700)

    # ── resources(workspace 文件只读)──

    def test_resources_list_lists_text_files(self):
        (self.workspace / "README.md").write_text("hi", encoding="utf-8")
        (self.workspace / ".env").write_text("KEY=secret", encoding="utf-8")
        (self.workspace / "checkpoints").mkdir(exist_ok=True)
        (self.workspace / "checkpoints" / "best.pth").write_bytes(b"x")
        resp = self._call("resources/list")
        uris = [r["uri"] for r in resp["result"]["resources"]]
        self.assertTrue(any("README.md" in u for u in uris))
        self.assertFalse(any(".env" in u for u in uris))       # 敏感文件排除
        self.assertFalse(any("best.pth" in u for u in uris))   # 二进制/checkpoints 排除

    def test_resources_read_file(self):
        (self.workspace / "note.txt").write_text("hello resources", encoding="utf-8")
        resp = self._call("resources/read", {"uri": "workspace://note.txt"})
        self.assertEqual(resp["result"]["contents"][0]["text"], "hello resources")

    def test_resources_read_rejects_traversal(self):
        resp = self._call("resources/read", {"uri": "workspace://../outside"})
        self.assertEqual(resp["error"]["code"], -32602)

    # ── 工具执行 ──

    def test_tools_call_writes_file_through_sandbox(self):
        resp = self._call("tools/call", {"name": "write_file",
                                         "arguments": {"path": "note.txt",
                                                       "content": "hello"}})
        self.assertFalse(resp["result"]["isError"])
        self.assertTrue((self.workspace / "note.txt").exists())

    def test_tools_call_reads_file(self):
        (self.workspace / "note.txt").write_text("hello", encoding="utf-8")
        resp = self._call("tools/call", {"name": "read_file",
                                         "arguments": {"path": "note.txt"}})
        text = resp["result"]["content"][0]["text"]
        self.assertEqual(text, "hello")

    def test_tools_call_unknown_tool(self):
        resp = self._call("tools/call", {"name": "nope", "arguments": {}})
        self.assertEqual(resp["error"]["code"], -32602)

    def test_tools_call_bad_arguments_type(self):
        resp = self._call("tools/call", {"name": "read_file",
                                         "arguments": "not-a-dict"})
        self.assertEqual(resp["error"]["code"], -32602)

    def test_tools_call_path_traversal_still_blocked(self):
        """MCP 只是传输层:路径越界必须仍被工具层拒绝。"""
        resp = self._call("tools/call", {"name": "read_file",
                                         "arguments": {"path": "../outside"}})
        text = resp["result"]["content"][0]["text"]
        self.assertIn("out of workspace bounds", text)

    def test_tools_call_sandbox_read_only_blocks_write(self):
        """沙箱策略跨协议生效:read-only 模式下 MCP 写文件被拒。"""
        set_tool_context(self.workspace, LocalExecutionBackend(self.workspace),
                         sandbox=Sandbox("read-only"))
        resp = self._call("tools/call", {"name": "write_file",
                                         "arguments": {"path": "x.txt",
                                                       "content": "hi"}})
        text = resp["result"]["content"][0]["text"]
        self.assertIn("read-only", text)
        self.assertFalse((self.workspace / "x.txt").exists())


class MCPStdioTests(unittest.TestCase):
    def test_serve_stdio_end_to_end(self):
        """子进程方式跑 `python -m core.mcp_server --workspace`,走真实 stdio。"""
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.Popen(
                [sys.executable, "-m", "core.mcp_server",
                 "--workspace", tmp],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True,
                cwd=str(PROJECT_ROOT),
            )
            try:
                proc.stdin.write(json.dumps({
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2025-03-26",
                               "capabilities": {}, "clientInfo": {}},
                }) + "\n")
                proc.stdin.write(json.dumps({
                    "jsonrpc": "2.0", "id": 2, "method": "tools/list",
                }) + "\n")
                proc.stdin.flush()
                init = json.loads(proc.stdout.readline())
                listing = json.loads(proc.stdout.readline())
                self.assertEqual(init["result"]["serverInfo"]["name"],
                                 "deep-researcher-tools")
                names = {t["name"] for t in listing["result"]["tools"]}
                self.assertIn("write_file", names)
                self.assertIn("launch_experiment", names)
            finally:
                proc.stdin.close()
                proc.wait(timeout=30)


if __name__ == "__main__":
    unittest.main()
