# 块 3：工具系统 — Tool-Use 协议 + 安全防线

## Agent 概念

本块涉及 **4 个 Agent 设计模式**，聚焦于"Agent 的手怎么被约束"：

### 概念 1：Text Protocol Tool-Use（工具使用模式）

```
LLM 输出                   Python 处理                    反馈给 LLM
───────                   ──────────                    ──────────
<tool_call>    ─────────→  正则提取 JSON   ─────────→  <tool_result name="x">
{"name":"x",                 │                           ...结果...
 "args":{...}}               ├─ 查 handler 字典           </tool_result>
</tool_call>                 ├─ 路径安全检查
                             ├─ 执行 handler
                             └─ 返回 JSON
```

和 Anthropic/OpenAI 原生 tool-use 的本质区别：**工具定义和执行都在自己的 Python 代码里，不依赖 LLM SDK 的 tool 功能**。

### 概念 2：Tool Whitelist / Blacklist（安全模式）

每个 Agent 类型只有**部分工具**可用——这是最小权限原则：

| Agent | 可用工具 | 不可用 |
|-------|---------|--------|
| Leader | read_file, write_file, log_memory | ❌ run_shell, launch_experiment |
| Idea | search_papers, search_arxiv, get_paper, read, write | ❌ run_shell, launch_experiment |
| Code | run_shell, launch_experiment, read, write, list, search | — |
| Writing | read, write, list, search | ❌ run_shell, launch_experiment |

> 💡 **`list_files` vs `list_tree` 的区别**：表里的 "list" 对 Idea/Writing Agent 是 `list_files`（平铺列出单个目录），对 Code Agent 额外包含 `list_tree`（递归列出整个仓库的目录树）。为什么 Writing Agent 不需要 `list_tree`？它只需要找到已有报告/结果文件——`list_files` 就够了。Code Agent 需要 `list_tree` 来理解整个代码库的结构（哪些模块、哪些配置、哪些脚本），递归目录树是代码理解的第一步。

> ⚠️ **注意 `log_memory` 工具**：虽然 Leader 可以调用 `log_memory`，但它的实现（`tools.py:489-491`）只是一个**空壳**——它返回 `{"status": "logged"}` 但不做任何持久化。真正的记忆写入发生在 `_reflect()` 阶段（`loop.py:296-299`），由 `memory.log_decision()` / `memory.log_milestone()` 完成。这是有意设计的解耦：工具调用只是"提议"写记忆，Leader 复盘时才"批准"写入，防止 Worker 绕过 Leader 直接篡改项目记忆。

```
         ┌──────────────────────────────┐
         │     所有工具（12 个）          │
         │  ┌────────────────────────┐  │
         │  │  Code Agent 可用 (7)   │  │  ← 最危险的工具只给 Code Agent
         │  │  ┌──────────────────┐  │  │
         │  │  │ Idea Agent (5)   │  │  │
         │  │  │ ┌────────────┐   │  │  │
         │  │  │ │Leader (3)  │   │  │  │
         │  │  │ └────────────┘   │  │  │
         │  │  └──────────────────┘  │  │
         │  └────────────────────────┘  │
         └──────────────────────────────┘
```

### 概念 3：Path Traversal Prevention（安全模式—沙箱）

```python
# execution.py:438-450
def normalize_relative_path(path):
    if pure.is_absolute():        raise ValueError(...)  # ① 拒绝绝对路径
    if any(part == ".." for part in pure.parts):          # ② 拒绝 ..
        raise ValueError(...)
```

**两层验证**（normalize + resolve），防止 Agent 读取 `/etc/passwd` 或 `../../secrets.env`。

### 概念 4：Dangerous Command Blacklist（安全模式—命令黑名单）

```python
# tools.py:303-315
dangerous_bins = {"rm", "sudo", "su", "mkfs", "dd", "shutdown", "reboot", "poweroff", "halt"}
if Path(argv[0]).name in dangerous_bins:
    raise ValueError(f"Blocked executable: {argv[0]}")
```

**`Path(argv[0]).name` 为什么能抓到 `/usr/bin/rm`**：这是一个三步检查链：

```
命令字符串                     → shlex.split  → argv[0]      → Path(...).name
"/usr/bin/rm -rf /"           → ["/usr/bin/rm", "-rf", "/"]  → "rm"          → BLOCKED ✅
"echo 'rm -rf /'"             → ["echo", "rm -rf /"]          → "echo"        → 放行 ✅（echo 不在黑名单）
"python train.py"             → ["python", "train.py"]        → "python"      → 放行 ✅
"sudo python train.py"        → ["sudo", "python", "train.py"] → "sudo"       → BLOCKED ✅
```

关键点：`Path(argv[0]).name` 提取的是**纯可执行文件名**（basename），剥离了路径前缀。所以 `/usr/bin/rm`、`./rm`、`rm` 都会被统一识别为 `"rm"` 并拦截。`echo "rm -rf /"` 不会被误拦，因为 shlex 正确解析出 argv[0] 是 `echo`，不是 `rm`。

黑名单不是正则匹配命令字符串——是先 `shlex.split` 解析成 argv，再检查 argv[0] 的可执行文件名。这样 `echo "rm -rf /"` 不会被误拦（argv[0] 是 `echo`），而 `rm -rf /` 会被拦（argv[0] 是 `rm`）。

---

## 代码地图

代码分布在两个文件：

**`core/tools.py`**（492 行）— 工具定义 + 路由 + 执行：

| 函数（行号） | 职责 |
|-------------|------|
| `get_tools_for()` (35) | 按 Agent 类型返回工具列表（白名单） |
| `execute_tool()` (64) | 按名称路由到具体 handler |
| `_parse_command()` (290) | shlex 解析 + 危险命令黑名单检查 |
| `_exec_run_shell()` (319) | 执行短期命令（≤120s） |
| `_exec_launch_experiment()` (325) | 启动长期训练（nohup） |
| `_exec_write_file()` (339) | 写文件 + 保护文件检查 |
| `_exec_read_file()` (348) | 读文件 + 分片读取 + 截断上限 |
| 工具 schema 属性 (92-282) | 12 个 `@property` 返回工具定义 |

> **初学提示**：你看到的 `_tool_run_shell` 返回的那个 dict（name + description + input_schema），最终会被 `_render_tools_section()`（agents.py:309）渲染成一段 **Markdown 文本**，拼在 Worker 的 system prompt 后面。LLM 看到的不是 JSON Schema 对象，而是：
> ```
> - `run_shell` — Run a shell command and return output.
>     - `command` (string, required): Shell command to execute
>     - `timeout` (integer, optional): Timeout in seconds
> ```
> 这意味着**工具定义的真实"消费者"是 LLM 不是代码**。代码只管执行，LLM 管理解。所以 description 写得好不好，直接影响 LLM 会不会正确使用这个工具。

**`core/execution.py`**（1189 行）— 执行后端：

| 函数（行号） | 职责 |
|-------------|------|
| `normalize_relative_path()` (438) | 路径规范化 + 防穿透 |
| `_resolve_under_root()` (453) | 二次验证：resolve 后仍在 workspace 内 |

---

## 调用链追踪

### 第 1 步：Worker 说"我要调工具"→ 怎么被执行的？

从块 2 的 `dispatch_worker()` 内部开始追：

```
agents.py:252  tool_output = tool_registry.execute_tool(name, args)
                     │
tools.py:64     def execute_tool(self, name, args):
tools.py:66         handlers = {                       ← 字典路由
tools.py:67             "run_shell": self._exec_run_shell,
tools.py:68             "launch_experiment": self._exec_launch_experiment,
tools.py:69             ...
tools.py:78         }
tools.py:80         handler = handlers.get(name)       ← O(1) 查找
tools.py:81         if not handler:
tools.py:82             return json.dumps({"error": f"Unknown tool: {name}"})
tools.py:85         return handler(**args)             ← 真正执行
```

**整个工具系统就是一个大字典**——工具名 → handler 函数。没有反射、没有动态导入、没有工厂模式。简单到你可以看完就写一个。

### 第 2 步：run_shell 的安全检查链

当 Agent 说 `<tool_call>{"name":"run_shell","args":{"command":"rm -rf /"}}</tool_call>` 时：

```
tools.py:319  _exec_run_shell(command="rm -rf /", timeout=120)
tools.py:321      argv = self._parse_command(command)
                       │
tools.py:290          def _parse_command(command):
tools.py:292              if not command.strip(): raise        ← ① 空命令拒绝
tools.py:296              argv = shlex.split(command)          ← ② 用 shell 规则解析
                           # argv = ["rm", "-rf", "/"]
tools.py:303              dangerous_bins = {"rm","sudo",...}   ← ③ 危险命令黑名单
tools.py:314              if Path(argv[0]).name in dangerous_bins:
tools.py:315                  raise ValueError("Blocked!")     ← ④ 拒绝！
```

**4 层按顺序检查，任何一层失败都拒绝执行。** 注意 `shlex.split` 不是简单的 `str.split(" ")`——它能正确处理引号、转义：
```python
shlex.split('echo "hello world"')  # → ['echo', 'hello world'] ✅
"echo 'hello world'".split(" ")    # → ['echo', "'hello", "world'"] ❌
```

**`run_shell` 的输出截断**（`execution.py:679-680`）：命令执行完后，stdout 只保留最后 2000 字符，stderr 只保留最后 500 字符：

```python
result.stdout[-2000:]   # stdout 截断到末尾 2000 字符
result.stderr[-500:]    # stderr 截断到末尾 500 字符
```

**为什么截断？** 工具输出会作为 `<tool_result>` 注入下一轮 LLM 调用的 input。如果不截断，一个 `python train.py` 输出几万行日志 → 几万 tokens → 直接撑爆 context window → LLM 调用失败或成本爆炸。**为什么取末尾而不是开头？** 错误信息通常在输出的最后几行（traceback 的最后一行是真正的错误），开头往往是正常的启动日志。

**为什么 stdout 2000 而 stderr 只有 500？** 正常输出可能包含有意义的指标（loss、accuracy），值得多看。错误输出只需要最后几行就够了。

### 第 3 步：write_file 的文件保护

```
tools.py:339  _exec_write_file(path="MEMORY_LOG.md", content="...")
tools.py:341      normalized = self._normalize_path(path)     ← ① 防路径穿透
tools.py:342      if normalized.split("/")[-1] in self._protected_files:  ← ② 保护文件检查
tools.py:343          return json.dumps({"error": "Cannot overwrite protected file"})
tools.py:345      result = self.backend.write_file(normalized, content)
```

`_protected_files` 包括：`state.json`, `MEMORY_LOG.md`, `PROJECT_BRIEF.md`, `.lock`——Agent 永远不能覆盖这四个文件。

### 第 4 步：launch_experiment — 最复杂的工具

```
tools.py:325  _exec_launch_experiment(command="python train.py", log_file="logs/run1.log", gpu="0")
tools.py:327      env = {}
tools.py:329      if gpu: env["CUDA_VISIBLE_DEVICES"] = gpu    ← GPU 隔离
tools.py:331      argv = self._parse_command(command)          ← 和 run_shell 一样的安全检查
tools.py:332      result = self.backend.launch_command(        ← nohup + 返回 PID
                       argv=argv, log_file=log_file, env=env)
tools.py:337      return json.dumps(result)                    ← {"pid": 12345, "log_file": "..."}
```

关键区别：`run_shell` 用 `subprocess.run()`（**阻塞**——Python 卡住等命令跑完才继续），`launch_experiment` 用 `subprocess.Popen`（**非阻塞**——启动后立刻返回，只拿 PID）。

**`start_new_session=True` 的作用**（`execution.py:690-698`）：

```python
# execution.py:690-698
proc = subprocess.Popen(
    argv,
    stdout=handle,          # 输出重定向到日志文件
    stderr=subprocess.STDOUT,
    start_new_session=True, # ← 关键：让训练进程成为独立会话的 leader
    cwd=str(self.workspace),
)
```

`start_new_session=True` 底层调用了 `setsid()` 系统调用——训练进程脱离 Agent 进程的进程组，成为自己会话的 leader。这意味着：
- **Agent 崩了 → 训练继续跑**（两个独立的进程组）
- **Agent 重启 → 训练不受影响**（不需要 Agent 进程活着）
- 这就是"你可以关掉终端"这句承诺的底层实现——不是靠 `nohup` 命令，而是靠 `setsid()`

> **初学提示**：`run_shell` 像在终端跑 `ls`——输完命令等着结果。`launch_experiment` 像点"开始训练"按钮——按钮立刻弹回来（返回 PID），训练在后台跑 2 小时。Agent 不卡在"等训练"上，而是进入 Monitor 阶段每 15 分钟看一眼。搞混这两个执行模型就理解不了 Agent 为什么能同时"等训练"又不阻塞主循环。

---

## 每个概念：为什么选这个？有没有更好的？

### 概念 1：Text Protocol Tool-Use vs 原生 Tool-Use

**为什么选文本协议？**（块 2 已详细分析，这里从工具系统角度补充）

从工具系统的角度看，文本协议的核心优势是**完全掌控**：

| 维度 | 原生 tool-use | 文本协议（本项目） |
|------|-------------|-----------------|
| 工具定义格式 | 各 SDK 不同 | 统一 dict schema |
| 参数校验 | 依赖 SDK 自动校验 | 自己 handler 里处理，灵活 |
| 执行顺序 | SDK 决定 | Python for 循环，完全可控 |
| 错误处理 | SDK 内部处理 | 自己 catch，返回 JSON error |
| 审计 | 需要另加 | 每个 tool call 都在 tool_results_log 里 |

**业界主流做法**：

| 方案 | 适用场景 |
|------|---------|
| **OpenAI Function Calling** | 只用 OpenAI 生态，最成熟 |
| **Anthropic Tool Use** | 只用 Anthropic，类型最严格 |
| **MCP (Model Context Protocol)** | 工具跨平台、跨模型复用 |
| **文本协议（本项目）** | 多 provider + 需要完全控制 |
| **LangChain Tools** | 快速原型，不想自己写 |

**改进方向**：如果未来 MCP 成为行业标准，可以将工具定义从 dict 迁移到 MCP server，不改 Agent 循环逻辑。

### MCP 专题：2025-2026 面试必问

MCP（Model Context Protocol）是 Anthropic 2024 年底发布、2025 年迅速普及的工具调用开放标准。2026 年 OpenAI 也宣布支持。**这个项目没用 MCP，但你需要知道它是什么、和本项目的做法比有什么区别。**

**本质**：MCP 把工具调用从"代码内嵌"变成"C/S 架构"——

```
本项目做法（内嵌）：                   MCP 做法（C/S 架构）：
───────────────                      ───────────────
Python 代码里写死工具定义              工具定义在一个独立的 MCP Server 进程里
  │                                     │
  ├─ ToolRegistry.execute_tool()        ├─ MCP Client (Agent 侧)
  ├─ 工具定义和工具执行在同一进程         │    └─ 通过 JSON-RPC 发现和调用工具
  └─ 加工具 = 改 Python 代码             │
                                        ├─ MCP Server (工具侧)
                                        │    └─ 独立进程，暴露工具能力
                                        └─ 加工具 = 启动新的 MCP Server，不改 Agent 代码
```

**JSON-RPC 消息长什么样**——MCP 用 JSON-RPC 2.0 协议通信，下面是一个具体例子：

```json
// ① Agent 请求：列出所有可用工具
{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}

// ② Server 响应：返回工具列表
{"jsonrpc": "2.0", "id": 1, "result": {"tools": [
  {"name": "run_shell", "description": "Run a shell command",
   "inputSchema": {"type": "object", "properties": {
     "command": {"type": "string"},
     "timeout": {"type": "integer"}
   }}}
]}}

// ③ Agent 请求：调用工具
{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
  "name": "run_shell", "arguments": {"command": "nvidia-smi", "timeout": 30}
}}

// ④ Server 响应：返回工具执行结果
{"jsonrpc": "2.0", "id": 2, "result": {"content": [
  {"type": "text", "text": "GPU 0: Tesla V100, 78% util, 45°C"}
]}}
```

每一次交互都是 JSON-RPC 请求-响应对——和本项目 `execute_tool("run_shell", {"command": "nvidia-smi"})` → `json.dumps(result)` 本质上做一样的事，区别在于 MCP 把它标准化了，工具提供者和使用者可以在不同进程、不同语言里实现。

**MCP 的三个核心概念**：

| 概念 | 说明 | 类比 |
|------|------|------|
| **Tools** | MCP Server 暴露的可调用函数。LLM 可以请求调用 | 相当于本项目的 `_tool_run_shell` dict，但定义在 Server 端 |
| **Resources** | MCP Server 暴露的只读数据。LLM 可以读取但不执行 | 相当于本项目的 `read_file`，但数据源可以来自任何地方（数据库、API、文件系统） |
| **Prompts** | MCP Server 预定义的 prompt 模板。Agent 可以复用 | 本项目没有这个——prompt 是硬编码在 `agents/*.md` 里的 |

**本项目 vs MCP 的对比**：

| 维度 | 本项目（内嵌工具） | MCP |
|------|-----------------|-----|
| 工具定义位置 | `tools.py` 的 `@property` 方法 | 独立的 MCP Server（可以是任何语言） |
| 添加新工具 | 改 Python 代码 → 重启 Agent | 启动/注册新的 MCP Server → Agent 自动发现 |
| 跨语言 | ❌ 工具必须用 Python 写 | ✅ MCP Server 可以用任何语言 |
| 跨 Agent 复用 | ❌ 只有这一个 Agent 能用 | ✅ 多个 Agent 可以连接同一个 MCP Server |
| 调试 | 在 Agent 进程内调试 | 独立进程，可以单独测试 |
| 复杂度 | 简单（0 外部依赖） | 需要运行 MCP Server + JSON-RPC 通信 |
| 适用场景 | 单 Agent、工具 ≤ 20 个、团队 ≤ 3 人 | 多 Agent、工具多、团队协作、需要权限隔离 |

**为什么这个项目没用 MCP？**

项目作者的目标是"一个研究者在自己的 GPU 机器上跑 Agent"，工具就是读文件、写文件、跑 shell——12 个工具，不多。引入 MCP 的复杂度（运行 Server 进程、管理 JSON-RPC 连接、处理进程间通信）不值得。但如果你的场景是企业级（10+ Agent 共享工具池、工具由不同团队开发、需要权限管控），MCP 就是正确的选择。

**面试时的标准回答**：

> "我手写过内嵌式工具调用——工具定义是 Python dict，执行是字典路由 handler，LLM 通过 `<tool_call>` 文本协议触发。优点是零依赖、完全可控。MCP 我理解它的架构——工具提供者和使用者通过 JSON-RPC 解耦，适合企业级多 Agent 共享工具池的场景。两种我都清楚各自的 tradeoff。"

### 概念 2：Tool Whitelist（按 Agent 类型分配工具）

**为什么每个 Agent 只有部分工具？**

三个原因，按重要性排列：

1. **安全**：Leader 不能调 `run_shell`——即使 Leader 被 prompt injection 攻击，最坏情况也只是输出错误决策，不能执行命令。
2. **成本**：每个工具定义大约 200-400 tokens。全给所有 Agent 会显著增加每次调用的 input token 消耗。
3. **可靠性**：工具越少，LLM 选择正确工具的准确率越高。

**业界主流做法**：

| 方案 | 何时用 |
|------|--------|
| **按角色分配（本项目）** | Agent 职责明确分工 |
| **动态工具路由** | LLM 先选工具集，再从子集里选工具（两层） |
| **全部工具都给** | Agent 只有一种类型、工具 < 5 个 |
| **工具权限级别** | 每个工具有 risk level，Agent 有 clearance level |

**改进方向**：可以加入动态工具权限——比如 Code Agent 在 dry-run 阶段可以有 `run_shell` 但不能有 `launch_experiment`，通过后再给。

### 概念 3：路径防穿透

**为什么用两层验证（normalize + resolve）？**

```python
# 第一层：normalize_relative_path（execution.py:438-450）— 字符串级别
def normalize_relative_path(path: str) -> str:
    pure = PurePosixPath(str(path))
    if pure.is_absolute():                    # 拒绝 "/etc/passwd"
        raise ValueError("Path must be relative to workspace")
    if any(part == ".." for part in pure.parts):  # 拒绝 "../../secrets"
        raise ValueError(f"Path escapes workspace: {path}")
    return str(pure)

# 第二层：_resolve_under_root（execution.py:453-460）— 文件系统级别
def _resolve_under_root(root: Path, rel_path: str) -> Path:
    parts = [part for part in PurePosixPath(rel_path).parts if part not in ("", ".")]
    resolved = (root / Path(*parts)).resolve(strict=False)  # ① 解析所有符号链接和 ..
    try:
        resolved.relative_to(root)                          # ② 确认最终路径在 root 下
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace: {rel_path}") from exc
    return resolved
```

**为什么需要第二层？** 因为第一层是纯字符串检查，可能被绕过：
- 符号链接：`workspace/link → /etc` → 第一层通过（没有 ".."），第二层拦截
- 奇怪的 Unicode 路径名

**业界主流做法**：

| 方案 | 安全性 | 本项目 |
|------|--------|--------|
| 纯字符串匹配 | 低（符号链接绕过） | 第一层 |
| resolve + relative_to | 高 | 第二层 |
| chroot / container | 最高 | 未使用 |
| seccomp / AppArmor | 最高 | 未使用 |

> **seccomp**（Secure Computing Mode）：Linux 内核机制，限制进程能调用的系统调用。例如：只允许 `read`/`write`/`exit`，禁止 `open`/`unlink`/`execve`。配置一个 seccomp profile 后，Agent 进程即使被注入恶意代码也无法执行危险系统调用。
>
> **AppArmor**（Application Armor）：Linux LSM（Linux Security Module），为特定应用程序设定文件系统访问权限。例如：只允许读写 `/workspace/` 下的文件，禁止访问 `/etc/`、`/home/`。比 seccomp 更粗粒度（按路径控制而非按系统调用控制）。

**改进方向**：企业级部署可以加 Docker 容器隔离。本项目没做是因为目标用户是研究者，在自己的 GPU 机器上跑——Docker 会增加 GPU 驱动配置的复杂度。

### 概念 4：命令黑名单（dangerous_bins）

**为什么是黑名单而不是白名单？**

白名单（只允许特定命令）更安全，但在这个场景下不实用——Agent 需要执行任意的 Python 训练脚本。

**业界主流做法**：

| 方案 | 安全级别 | 适用场景 |
|------|---------|---------|
| **黑名单（本项目）** | 中 | Agent 需要灵活执行命令 |
| **白名单** | 高 | Agent 任务范围固定 |
| **沙箱（container/seccomp）** | 最高 | 多租户、不可信代码 |
| **审批流程（HITL）** | 最高（人工判断） | 高风险操作 |

**改进方向**：
1. 黑名单可以扩展——比如加上 `curl`、`wget`（防止数据外泄）
2. 可以把黑名单做成 `config.yaml` 里的可配置项，不同项目不同规则

---

## 关键代码解析

### 1. 工具定义的"双重用途"

每个工具定义是一个 dict，同时服务于两个目的：

```python
# tools.py:92-105
@property
def _tool_run_shell(self) -> dict:
    return {
        "name": "run_shell",                          # ① 工具名
        "description": "Run a shell command...",       # ② 给 LLM 看的说明
        "input_schema": {                              # ③ 给 LLM 看的参数格式
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "..."},
                "timeout": {"type": "integer", "default": 120},
            },
            "required": ["command"],
        },
    }
```

- `name` + `description` + `input_schema` → 由 `_render_tools_section()`（agents.py:309）渲染成 Markdown 注入 system prompt
- 同一个 schema 被 `get_tools_for()` 返回给 `dispatch_worker()` 使用

### 2. `shlex.split` — 正确解析命令的关键

```python
# tools.py:296
argv = shlex.split(command)
```

这不是普通的字符串分割——`shlex.split` 按照 POSIX shell 的引用规则解析：
- `echo "hello world"` → `['echo', 'hello world']`（引号内的空格不分割）
- `python -c "print('hi')"` → `['python', '-c', "print('hi')"]`
- 未闭合的引号会抛异常（行 297）

### 3. 保护文件检查的位置

```python
# tools.py:33
self._protected_files = {"state.json", "MEMORY_LOG.md", "PROJECT_BRIEF.md", ".lock"}

# tools.py:342
if normalized.split("/")[-1] in self._protected_files:
    return json.dumps({"error": f"Cannot overwrite protected file: {path}"})
```

**注意**：检查的是文件名（`split("/")[-1]`），不是完整路径。这意味着 Agent 在子目录里生成 `MEMORY_LOG.md` 也会被拦截。如果你要改这个逻辑，确保理解了它防止的是"Agent 覆盖人类的关键文件"这个场景。

### 4. read_file 的分片读取

```python
# tools.py:348-359
def _exec_read_file(self, path, start_line=None, end_line=None):
    if start_line is not None or end_line is not None:
        content = self.backend.read_file_range(normalized, start_line, end_line)
        return content[:20000]   # 分片读取上限 20K
    content = self.backend.read_file(normalized)
    return content[:10000]       # 全文读取上限 10K
```

**为什么分片读取上限更大？** 因为分片读取表明 Agent 知道自己在看哪一段（比如训练日志的最后 100 行），这通常意味着它在做有针对性的分析。全文读取更可能是"不管内容先全部读进来"——浪费 token。

**为什么是 10K/20K 字符？** ~10K 英文字符 ≈ 2,500 tokens，~20K ≈ 5,000 tokens。工具输出会成为下一轮 LLM 调用的 input token。如果没有上限，Agent 读了一个 50KB 的日志文件，那 ~12,500 tokens 直接塞进下一轮的 context——相当于多花了 ~$0.04 只为了传输日志。两个上限确保工具输出不会吃掉整个 context window 的 input budget。

---

## 设计决策分析

### 决策：工具定义用 `@property` 而不是常量 dict

**原因**：每个工具定义 200-400 tokens。如果用常量 dict，所有 12 个工具的定义在类初始化时全部加载到内存。用 `@property` 实现**按需加载**——只有 `get_tools_for()` 返回时才访问，且只访问分配给特定 Agent 的那几个（Code Agent 只用 7 个，Leader 只用 3 个）。

> ⚠️ **不是真正的"懒加载"**：Python 的 `@property` 是每次访问都运行 getter 函数（accessor descriptor），不是"首次访问后缓存结果"。但这 12 个 property 都返回静态 dict（不涉及 I/O 或计算），所以"每次重新计算"的开销可以忽略。真正的懒加载（cached property）需要 `functools.cached_property` 或 `__slots__`，但作者在这些地方没用到，实际上也不需要——因为 getter 太简单。

**另一个好处**（设计前瞻）：如果未来需要动态工具定义（比如根据 workspace 内容调整 tool description），`@property` 可以在每次访问时重新计算——不需要改 API，只要改 getter 的逻辑。

### 决策：execute_tool 的大字典路由而不是 if-elif

**原因**：12 个工具，`if-elif` 链是 O(n)，字典是 O(1)。虽然 12 个工具的性能差异可以忽略，但字典路由更易扩展——加新工具只需加一行 key-value。

### 决策：错误返回 JSON 而不是抛异常

```python
# 所有 exec_* 方法都返回 JSON
return json.dumps({"error": f"Unknown tool: {name}"})
```

**原因**：工具执行的错误不应该导致 Agent 崩溃。返回 `{"error": "..."}` 作为 `<tool_result>`，LLM 看到错误后可以自己决定怎么处理（换一个工具、换个参数、向人类求助）。这是**优雅降级**而不是**崩溃退出**。

---

## 本块要掌握的代码

| 代码 | 位置 | 掌握标准 |
|------|------|---------|
| `get_tools_for()` 白名单路由 | `tools.py:35-62` | 能说出四种 Agent 各自能调哪些工具，为什么 Leader 不能调 `run_shell` |
| `execute_tool()` 字典分发 | `tools.py:64-88` | 知道 12 个工具 handler 怎么路由（大字典 O(1)），未知工具怎么处理 |
| `_parse_command()` 安全检查链 | `tools.py:290-317` | 能说出 4 层检查（空命令→shlex 解析→黑名单→argv 提取），为什么用 shlex 而不是 str.split |
| `dangerous_bins` 黑名单 | `tools.py:303-315` | 知道哪些命令被禁，为什么检查的是 `argv[0]` 的 basename 而不是命令字符串 |
| `_exec_run_shell()` 阻塞执行 | `tools.py:319-323` | 知道和 `launch_experiment` 的本质区别（阻塞 vs 非阻塞） |
| `_exec_launch_experiment()` | `tools.py:325-337` | 知道为什么返回 PID 和 log_file，这两个字段下一步被谁消费 |
| `_exec_write_file()` 保护文件 | `tools.py:339-346` | 知道 4 个保护文件名，为什么检查 `split("/")[-1]` 而不是完整路径 |
| `normalize_relative_path()` 防穿透 | `execution.py:438-450` | 知道两层验证（字符串级 + `resolve` + `relative_to`），各能防什么攻击 |
| `_render_tools_section()` | `agents.py:309-356` | 知道 tool schema dict→Markdown 文本的转换，LLM 看到的是什么格式 |

---

## 检验

1. Code Agent 调 `run_shell` 和 Idea Agent 调 `run_shell` 有什么区别？（答案：Idea Agent 根本没这个工具）
2. 如果 Agent 想用 `sudo rm -rf /`，在哪一行被拦截？具体是什么机制拦截的？
3. `../../secrets.env` 这个路径在哪一层被拦截？（提示：两层检查，哪层拦截？）
4. 为什么 `_exec_read_file()` 对分片读取给 20K 上限，全文读取只给 10K？
5. `launch_experiment` 返回的 JSON 里最关键的两个字段是什么？它们下一步被谁消费？
6. 如果 Agent 在子目录 `workspace/subdir/MEMORY_LOG.md` 写文件，会被拦截吗？为什么？
7. `_parse_command` 用了 `shlex.split` 而不是 `str.split(" ")`，为什么？举个会出错的例子。
8. 12 个工具定义中，哪些 Agent 可以调用 `log_memory`？

---

> 下一步：**块 4 — `core/04-memory-storage.md`**，理解 Agent 怎么"记住"历史和知识，而不会撑爆上下文。
