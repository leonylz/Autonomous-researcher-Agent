"""
MCP 适配层 — 把工具集暴露为标准 MCP Server(stdio 传输,零依赖)。

设计立场(协议无关的工具层):
- 核心工具是纯函数(@tool.func),与调用协议解耦:
  * 默认路径:agent 内嵌 bind_tools + ToolMessage(零开销、launch↔monitor 状态共享)
  * 可选路径:本模块把它们包成 MCP Server(JSON-RPC 2.0 over stdio),
    任何 MCP 客户端(Claude Desktop / Cursor / 其他 agent)都能发现和调用
- 安全继承:工具执行仍走沙箱门控 + 路径边界 + 命令守卫 —— MCP 只是传输层,
  不绕过任何安全策略

协议面(2025 标准 MCP 的最小实现,newline-delimited JSON-RPC over stdio):
  initialize          → 握手(protocolVersion + capabilities.tools)
  notifications/initialized → 忽略
  tools/list          → 工具 schema 列表(来自 langchain @tool 的 args_schema)
  tools/call          → 执行工具(返回 isError + content 或 error)

用法:
    from core.mcp_server import MCPServer, serve_stdio
    server = MCPServer(workspace=Path("."), backend=LocalExecutionBackend(...))
    server.serve_stdio()   # 阻塞;由 MCP 客户端以子进程方式启动

面试叙事:
    "工具层协议无关 —— 内嵌 bind_tools 是默认路径, MCP 适配层是可选项,
     一行配置切换; 加工具 = 加一个 @tool, 两边自动可见。"
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("autoresearcher.mcp")

PROTOCOL_VERSION = "2025-03-26"
SERVER_NAME = "deep-researcher-tools"
SERVER_VERSION = "0.1.0"


def _tool_schemas(tool_functions: dict) -> list[dict]:
    """把 TOOL_FUNCTIONS 里所有工具(去重)转成 MCP tools/list 的 schema。"""
    seen: dict[str, object] = {}
    for _, tools in tool_functions.items():
        for name, fn in tools:
            if name in seen:
                continue
            try:
                schema = fn.args_schema.schema()
            except Exception:
                schema = {"type": "object", "properties": {}}
            seen[name] = {
                "name": name,
                "description": (fn.description or "").strip(),
                "inputSchema": schema,
            }
    return list(seen.values())


class MCPServer:
    """最小 MCP Server(stdio)。工具执行复用 core.nodes 的 @tool.func,
    天然继承沙箱门控/路径边界/命令守卫。"""

    def __init__(self, tool_functions: Optional[dict] = None,
                 workspace: Optional[Path] = None):
        # 惰性导入:避免 import core.nodes 拉起整个 LLM 栈(参考 dashboard 的做法)
        if tool_functions is None:
            from .nodes import TOOL_FUNCTIONS
            tool_functions = TOOL_FUNCTIONS
        self._tool_functions = tool_functions
        self._workspace = workspace
        self._tools = _tool_schemas(tool_functions)
        self._request_id: Optional[int] = None  # 最近一次请求 id(供工具执行日志)

    # ── 协议处理 ──

    def handle_line(self, line: str) -> Optional[str]:
        """处理一行 JSON-RPC 请求,返回响应行(通知类返回 None)。"""
        line = line.strip()
        if not line:
            return None
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return self._error(None, -32700, "Parse error: invalid JSON")

        method = msg.get("method")
        request_id = msg.get("id")
        self._request_id = request_id

        # 通知(无 id)→ 不响应
        if request_id is None:
            if method == "notifications/initialized":
                return None
            return None

        try:
            if method == "initialize":
                return self._response(request_id, {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}, "resources": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                })
            if method == "tools/list":
                return self._response(request_id, {"tools": self._tools})
            if method == "tools/call":
                return self._handle_tool_call(request_id, msg.get("params") or {})
            if method == "resources/list":
                return self._response(request_id, {"resources": self._resource_list()})
            if method == "resources/read":
                return self._handle_resource_read(request_id, msg.get("params") or {})
            if method == "ping":
                return self._response(request_id, {})
            return self._error(request_id, -32601, f"Method not found: {method}")
        except Exception as exc:  # 协议级兜底:任何异常都回结构化错误
            logger.exception("MCP request failed: %s", method)
            return self._error(request_id, -32603, f"Internal error: {exc}")

    # ── resources(workspace 文件只读,复用 _resolve_path 的边界约束)──

    def _resource_list(self) -> list[dict]:
        """列出 workspace 下的文本文件(排除敏感/缓存/二进制)。"""
        if self._workspace is None:
            return []
        skip = {".env", ".python_env.json", ".python_env.status",
                "dry_run_log.json", ".last_launch.json", ".crash_context.json",
                "PENDING_APPROVALS.md"}
        skip_dirs = {".git", "__pycache__", ".trainenv", "checkpoints",
                     ".snapshots", "eval"}
        out = []
        try:
            for p in sorted(self._workspace.rglob("*")):
                if p.is_dir() or p.is_symlink():
                    continue
                if any(s in p.parts for s in skip_dirs):
                    continue
                if p.name in skip or p.suffix.lower() in (
                        ".pth", ".pt", ".bin", ".db", ".png", ".jpg", ".pyc"):
                    continue
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                if size > 200_000:
                    continue
                rel = p.relative_to(self._workspace).as_posix()
                out.append({"uri": f"workspace://{rel}", "name": rel,
                            "mimeType": "text/plain"})
        except OSError:
            pass
        return out[:100]

    def _handle_resource_read(self, request_id, params: dict) -> str:
        uri = str(params.get("uri", ""))
        if not uri.startswith("workspace://") or self._workspace is None:
            return self._error(request_id, -32602, "invalid resource uri")
        rel = uri[len("workspace://"):]
        from .nodes import _resolve_path
        target = _resolve_path(rel)
        if target is None:
            return self._error(request_id, -32602, "path escapes workspace")
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return self._error(request_id, -32602, f"read failed: {exc}")
        return self._response(request_id, {
            "contents": [{"uri": uri, "mimeType": "text/plain", "text": text[:100_000]}],
        })

    def _handle_tool_call(self, request_id, params: dict) -> str:
        name = params.get("name", "")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return self._error(request_id, -32602, "arguments must be an object")

        fn = None
        for _, tools in self._tool_functions.items():
            for tname, tfn in tools:
                if tname == name:
                    fn = tfn
                    break
        if fn is None:
            return self._error(request_id, -32602, f"Unknown tool: {name}")

        try:
            # 与 agent 内嵌路径完全一致:@tool.func 直接调用,安全守卫在内
            output = fn.func(**arguments)
            if isinstance(output, dict):
                output = json.dumps(output, ensure_ascii=False)
            return self._response(request_id, {
                "content": [{"type": "text", "text": str(output)}],
                "isError": False,
            })
        except Exception as exc:
            logger.warning("MCP tool %s failed: %s", name, exc)
            return self._response(request_id, {
                "content": [{"type": "text", "text": f"Tool error: {exc}"}],
                "isError": True,
            })

    # ── JSON-RPC 封装 ──

    @staticmethod
    def _response(request_id, result: dict) -> str:
        return json.dumps({
            "jsonrpc": "2.0", "id": request_id,
            "result": result,
        }, ensure_ascii=False)

    @staticmethod
    def _error(request_id, code: int, message: str) -> str:
        return json.dumps({
            "jsonrpc": "2.0", "id": request_id,
            "error": {"code": code, "message": message},
        }, ensure_ascii=False)

    # ── stdio 服务 ──

    def serve_stdio(self, stdin=None, stdout=None) -> None:
        """阻塞式 stdio 服务:逐行读请求,逐行写响应。"""
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        for line in stdin:
            resp = self.handle_line(line)
            if resp is not None:
                stdout.write(resp + "\n")
                stdout.flush()


def main():
    """CLI 入口:python -m core.mcp_server [--workspace PATH]

    供 MCP 客户端以子进程方式启动(stdio 传输)。
    """
    import argparse
    from .execution import LocalExecutionBackend
    from .nodes import set_tool_context

    parser = argparse.ArgumentParser(description="Deep Researcher MCP server (stdio)")
    parser.add_argument("--workspace", type=str, default=".",
                        help="工具工作区(所有路径操作的根)")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    from .sandbox import resolve_sandbox
    set_tool_context(workspace, LocalExecutionBackend(workspace),
                     sandbox=resolve_sandbox({}))
    server = MCPServer(workspace=workspace)
    server.serve_stdio()


if __name__ == "__main__":
    main()
