# MCP Bridge — 把工具集暴露给任意 MCP 客户端

工具层是**协议无关**的:同一组 `@tool` 纯函数,既可以内嵌给 agent
(`bind_tools`,零开销、launch↔monitor 状态共享),也可以包成标准 MCP
Server(stdio + JSON-RPC 2.0,零第三方依赖)供任何 MCP 客户端调用。

## 快速开始

```bash
# 在项目工作区启动 MCP server(stdio 传输,供客户端以子进程方式拉起)
python -m core.mcp_server --workspace examples/eval_tasks/T1_mnist
```

## 协议冒烟(手动验证)

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{}}}' | python -m core.mcp_server
# → {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-03-26","capabilities":{"tools":{}},"serverInfo":{"name":"deep-researcher-tools","version":"0.1.0"}}}

echo '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' | python -m core.mcp_server
# → {"jsonrpc":"2.0","id":2,"result":{"tools":[{...list_files...},{...launch_experiment...},...]}}
```

## 接入 Claude Desktop / Cursor

在客户端的 MCP 配置里注册 stdio server:

```json
{
  "mcpServers": {
    "deep-researcher-tools": {
      "command": "python",
      "args": ["-m", "core.mcp_server", "--workspace", "/abs/path/to/project"],
      "cwd": "/abs/path/to/auto-deep-researcher-nodes"
    }
  }
}
```

之后客户端可以直接调用 `read_file` / `search_arxiv` / `git_status` 等工具。

## 安全边界(跨协议生效)

MCP 只是传输层,**不绕过任何安全策略**:

- 沙箱分级:`--workspace` 下未显式配置时默认 `workspace-write`;
  read-only 模式下 `write_file`/`run_shell`/`launch_experiment` 被拒;
- 路径边界:越界(`..`/绝对路径/符号链接逃逸)在工具层拒绝;
- 命令守卫:无 shell 执行 + 危险命令黑名单;
- 环境剥离:子进程环境不含 API key。

验证:`tests/test_mcp_bridge.py::test_tools_call_path_traversal_still_blocked`
与 `test_tools_call_sandbox_read_only_blocks_write` 覆盖了这两条。

## 设计取舍(面试可讲)

| 路径 | 优点 | 代价 |
|------|------|------|
| 内嵌 bind_tools(默认) | 零开销;launch 的 PID/log 与 monitor 状态共享 | 工具与 agent 同进程 |
| MCP server(可选) | 生态互操作(任何客户端可用);工具可远程/隔离 | 进程间通信;launch→monitor 状态共享需额外设计 |

**为什么默认内嵌**:训练启动的 PID 权威交接要求工具结果直接进入
monitor 的 `_active_experiments` 登记 —— 跨进程会切断这条链。
MCP 的价值在"互操作演示"与"只读工具暴露",而非替换主路径。
