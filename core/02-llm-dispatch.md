# 块 2：LLM 调度 — Orchestrator-Worker 架构 + 多后端适配

> ⚠️ **术语提示**：下文提到的 "Leader-Worker" 是项目叫法，业界通用叫法是 **Orchestrator-Worker**。"Text Protocol Tool-Use" 目前没有标准名称，面试时直接描述实现："provider-agnostic tool-use via structured text parsing"。详见 [总纲术语对照表](../LEARNING_INDEX.md)。

## Agent 概念

本块涉及 **4 个 Agent 设计模式**，它们共同构成了"LLM 怎么被调用"这一层：

### 概念 1：Leader-Worker 架构（多 Agent 协作模式）

```
Leader（指挥官）              Worker（工兵）
   │                            │
   │  一次性思考 + 一次性复盘     │  多轮工具调用循环
   │  不调工具，只做决策          │  反复调工具直到任务完成
   │  保持对话历史（一个周期内）    │  每次派工都是全新对话
   │  1 次 LLM 调用 / 阶段       │  最多 N 轮（max_turns）
```

**这就是"战略与战术分离"**——Leader 不需要知道 `read_file` 怎么用，Worker 不需要知道"现在该做什么方向"。

### 概念 2：Multi-Provider 适配（多后端模式）

```
                    AgentDispatcher
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
        anthropic       openai       CLI (claude/codex)
       (API SDK)      (API SDK)    (subprocess 文本输入输出)
```

支持 4 种 LLM 后端，但对外暴露统一接口。这是经典的 **Adapter 模式**。

### 概念 3：Text Protocol Tool-Use（工具使用模式）

和 Anthropic/OpenAI 原生 tool-use 不同，这个项目自己定义了一套文本协议：

```
LLM 输出:  <tool_call>{"name": "read_file", "args": {"path": "x.py"}}</tool_call>
Python 处理: 正则提取 → ToolRegistry 执行 → 返回
反馈给 LLM: <tool_result name="read_file">...文件内容...</tool_result>
```

### 概念 4：Prompt Caching（成本优化）

```python
"cache_control": {"type": "ephemeral"}   # 行 404
```

把不变的 system prompt 缓存起来，后续调用只传增量。Anthropic 的 prompt caching 可以让长 system prompt 的重复传输成本降低 90%。

---

## 代码地图

所有代码在 **`core/agents.py`**（707 行）。

| 函数（行号） | 职责 | 概念 |
|-------------|------|------|
| `dispatch_leader()` (146) | Leader 单次 LLM 调用 + 保持历史 | Leader-Worker |
| `dispatch_worker()` (174) | Worker 多轮 tool-use 循环 | Text Protocol + Multi-turn |
| `_call_llm()` (358) | 路由到具体 provider | Multi-Provider |
| `_call_anthropic()` (380) | Anthropic SDK 调用 + prompt caching | Prompt Caching |
| `_call_openai()` (415) | OpenAI SDK 调用 + model mapping | Multi-Provider |
| `_call_claude_cli()` (515) | claude CLI subprocess 调用 | CLI Adapter |
| `_call_codex_cli()` (536) | codex CLI subprocess 调用 | CLI Adapter |
| `_parse_tool_calls()` (278) | 正则提取 `<tool_call>` 块 | Text Protocol |
| `_render_tools_section()` (309) | 生成工具 schema 文本注入 system prompt | Text Protocol |
| `_format_leader_input()` (614) | 把 context dict 格式化成 Leader 的输入 | Prompt 工程 |
| `_parse_leader_response()` (644) | 从 Leader 输出中提取 JSON 决策 | 输出解析 |
| `_parse_worker_response()` (666) | 从 Worker 输出中提取结构化结果 | 输出解析 |

---

## 调用链追踪

### 第 1 步：Leader 怎么被调用的？— `dispatch_leader()`（行 146-172）

```
loop.py:239  dispatch_leader("think", context)
                │
agents.py:159  system_prompt = _load_prompt("leader.md")    ← 读提示词文件
agents.py:161  messages = [*history, {role:"user", content}] ← 拼接对话
agents.py:167  response = _call_llm(system, messages)        ← 真正的 LLM 调用
agents.py:170  _leader_history = messages + [response]       ← 保存历史（周期内复用）
agents.py:172  return _parse_leader_response(response)       ← 提取 JSON
```

**关键理解**：
- Leader 的对话历史在**一个周期内**保持——如果 Leader 先 think 再 reflect，reflect 可以看到 think 的内容
- 但**每个新周期**（行 133）`reset_leader_history()` 清空历史——防止上下文无限增长
- Leader **不调工具**——它只需要读 brief/log/结果，然后输出决策

### 第 2 步：Worker 怎么被调用的？— `dispatch_worker()`（行 174-271）

这是整个项目最复杂的函数。逐段拆解：

**阶段 A：准备（行 196-211）**
```python
config = self.WORKER_CONFIGS[agent_type]   # 取配置
base_prompt = _load_prompt("code_agent.md") # 读提示词
tool_defs = tool_registry.get_tools_for("code")  # 取工具列表（块 3）
system_prompt = base_prompt + "\n\n" + _render_tools_section(tool_defs)  # 拼系统提示词
max_turns = 40                             # 最多 40 轮工具调用
```

**`_render_tools_section()` 的输出长什么样**（agents.py:309-356）——tool schema dict 被渲染成这段 Markdown 文本，注入 system prompt：

```markdown
## Tool-Use Protocol

You have NO direct access to the filesystem, shell, or network.
To act on the environment you MUST emit `<tool_call>` blocks and
wait for the framework to return `<tool_result>` blocks in the
next user turn. Example:

    <tool_call>
    {"name": "read_file", "args": {"path": "config.yaml"}}
    </tool_call>

You may emit multiple `<tool_call>` blocks in one message; each
will be executed and its result returned. When you are finished,
produce a plain-text message with NO `<tool_call>` blocks — that
is how the framework knows you are done.

Emit `<tool_call>` blocks at the top level of the message. Do NOT
wrap them in triple-backtick code fences — fenced blocks are
treated as illustration, not as real calls.

### Available tools

- `read_file` — Read a file from the workspace
    - `path` (string, required): Relative path to the file
    - `offset` (integer, optional): Line number to start reading from
    - `limit` (integer, optional): Max lines to read
- `write_file` — Write or overwrite a file
    - `path` (string, required): Relative path
    - `content` (string, required): File content
- ...
```

注意几个关键设计：
- 第一句 "You have NO direct access" — 明确告诉 LLM 它不能直接操作环境，必须通过 `<tool_call>`
- "Do NOT wrap in triple-backtick" — 与 `_parse_tool_calls()` 的 `_FENCED_BLOCK_RE` 过滤呼应
- 工具参数标了 `required`/`optional` — 减少 LLM 漏填必填参数的概率

**阶段 B：工具调用循环（行 229-267）**
```python
messages = [{"role": "user", "content": task}]  # 初始消息

for turn in range(1, max_turns + 1):            # 最多 max_turns 轮
    response = _call_llm(system, messages)       # ① 调 LLM

    tool_calls = _parse_tool_calls(response)     # ② 正则提取 <tool_call>
    if not tool_calls:                           # ③ 没有工具调用 → 完成！
        break

    messages.append({"role": "assistant", "content": response})  # ④ 记录对话

    result_blocks = []
    for call in tool_calls:                      # ⑤ 执行每个工具调用
        tool_output = tool_registry.execute_tool(name, args)
        result_blocks.append(f'<tool_result name="{name}">\n{tool_output}\n</tool_result>')

    messages.append({"role": "user", "content": "\n\n".join(result_blocks)})  # ⑥ 反馈结果
```

**这是一个标准的 ReAct 循环！**

> **ReAct** = **Rea**soning + **A**cting（推理 + 行动），来自 Yao et al. 2022 论文。标准模式是：Thought（思考下一步）→ Action（执行工具）→ Observation（观察结果）→ Thought（基于结果再思考）→ ... 循环直到任务完成。本项目 Worker 的实现是：LLM 输出（Thought + Action 合并为 `<tool_call>`）→ Python 执行工具（Action）→ `<tool_result>`（Observation）→ 下一轮 LLM 调用。和标准 ReAct 的区别是：Thought 和 Action 没有分开——LLM 在一次回复里同时输出推理和工具调用。

Worker 内部是 "思考→调工具→看结果→再思考" 的 ReAct 模式，但外层 Leader 是 Plan-Execute。这个嵌套是理解整个项目的关键。

### 具体演练：Code Agent 的一次完整派工

Leader 派给 Code Agent 的任务："把 learning rate 从 0.01 改成 0.001，跑一个实验"。

**Turn 1**：LLM 先探查代码库
```
→ LLM 收到: "把 learning rate 从 0.01 改成 0.001，跑一个实验"
← LLM 输出: 我先看看项目结构和配置文件
             <tool_call>{"name": "list_tree", "args": {"path": "."}}</tool_call>
             <tool_call>{"name": "search_code", "args": {"pattern": "lr|learning_rate"}}</tool_call>
```
Python 执行 `list_tree` + `search_code`，返回结果。

**Turn 2**：LLM 看了结果，找到配置文件，修改它
```
→ LLM 收到: <tool_result name="list_tree">train.py, config.yaml, ...</tool_result>
             <tool_result name="search_code">config.yaml:15: lr: 0.01</tool_result>
← LLM 输出: 找到了，lr 在 config.yaml 第 15 行。我先读一下配置文件。
             <tool_call>{"name": "read_file", "args": {"path": "config.yaml"}}</tool_call>
```
Python 返回文件内容。

**Turn 3**：LLM 修改配置 + 跑干跑
```
→ LLM 收到: <tool_result name="read_file">batch_size: 128\nlr: 0.01\n...</tool_result>
← LLM 输出: 把 lr 改成 0.001，然后干跑验证。
             <tool_call>{"name": "write_file", "args": {"path": "config.yaml", "content": "batch_size: 128\nlr: 0.001\n..."}}</tool_call>
```
Python 写文件成功。

**Turn 4**：干跑
```
→ LLM 收到: <tool_result name="write_file">{"ok": true, "path": "config.yaml"}</tool_result>
← LLM 输出: <tool_call>{"name": "run_shell", "args": {"command": "python train.py --max_steps 2 --dry_run"}}</tool_call>
```
Python 执行干跑，返回 "Training completed 2 steps, no errors."

**Turn 5**：干跑通过，启动正式训练
```
→ LLM 收到: <tool_result name="run_shell">Training completed 2 steps, no errors.</tool_result>
← LLM 输出: 干跑通过。启动正式训练。
             <tool_call>{"name": "launch_experiment", "args": {"command": "python train.py", "log_file": "logs/exp_002.log", "gpu": "0"}}</tool_call>
```
Python 启动训练，返回 `{"pid": 12346, "log_file": "logs/exp_002.log"}`。

**Turn 6**：LLM 报告完成（没有 tool_call → 循环结束）
```
→ LLM 收到: <tool_result name="launch_experiment">{"pid": 12346, "log_file": "logs/exp_002.log"}</tool_result>
← LLM 输出: 实验已启动。PID=12346，日志在 logs/exp_002.log。预计 2 小时完成。
```
没有 `<tool_call>` → `_parse_tool_calls()` 返回空 → `break` → 循环结束。

**关键观察**：
- 6 轮交换，但每轮的 LLM 调用都是"基于上一轮的结果做下一步"——这就是 ReAct
- 第 5 轮最关键：LLM 用 `launch_experiment` 而不是 `run_shell`——因为 prompt 里写了 MUST
- 如果第 4 轮干跑失败了，LLM 会回到 Turn 3 改代码再试——这就是 max_turns=40 存在的意义

**面试时可以这样讲**："外循环是 Plan-Execute——Leader 做战略决策不调工具；内循环是 ReAct——Worker 每轮基于工具结果决定下一步。这种嵌套设计在成本和智能之间取得了平衡：战略层便宜（2 次单轮调用）、执行层灵活（最多 40 轮自适应）。"

**阶段 C：结果解析（行 269-271）**
```python
result = _parse_worker_response(last_response, agent_type, tool_results_log)
# 优先从 launch_experiment 工具结果取 PID（行 683-698）——权威数据源
# 兜底：从文本中正则匹配 PID（行 700-704）
```

### 第 3 步：4 种 LLM 后端怎么切换？— `_call_llm()`（行 358-378）

```python
def _call_llm(self, system, messages):
    if self.provider == "claude_cli":   return _call_claude_cli(...)
    if self.provider == "codex_cli":    return _call_codex_cli(...)
    if self.provider == "openai":       return _call_openai(...)
    return _call_anthropic(...)         # 默认
```

一个 if-else 链，不是工厂模式也不是注册表。**但这是故意的——只有 4 个后端，不需要过度设计。**

### 第 4 步：文本协议怎么解析？— `_parse_tool_calls()`（行 278-306）

```python
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

def _parse_tool_calls(text):
    stripped = _FENCED_BLOCK_RE.sub("", text)  # ① 先移除代码块（防止误执行）
    for match in _TOOL_CALL_RE.finditer(stripped):
        body = match.group(1)
        parsed = json.loads(body)               # ② JSON 解析
        if isinstance(parsed, dict) and parsed.get("name"):
            calls.append(parsed)                # ③ 必须有 name 字段
    return calls
```

**两个安全措施**：
1. 代码块里的 `<tool_call>` 被忽略——LLM 举例说明协议时不会误触发
2. 必须有 `name` 字段——格式不对的静默跳过，不崩溃

---

## 每个概念：为什么选这个？有没有更好的？

### 概念 1：Leader-Worker 架构

**为什么选这个？**

| 需求 | Leader-Worker 怎么满足 |
|------|----------------------|
| Leader 需要看到全局（brief + log + 结果） | Leader 一次调用拿到全部 context |
| Worker 需要反复调工具（写代码→跑→看报错→改） | Worker 多轮 ReAct 循环 |
| 成本控制 | Leader 只在 think/reflect 时调 LLM（2 次/周期），训练期间 0 次 |
| 任务隔离 | 每个 Worker dispatch 是全新对话，不会相互污染 |

**有没有更好的选法？**

| 替代方案 | 优点 | 缺点 | 适合什么场景 |
|---------|------|------|------------|
| **纯 ReAct（不用 Leader）** | 简单，一个循环搞定 | 每次都要传完整 context，成本高；没有战略/战术分离 | 简单任务、对话 Agent |
| **Multi-Agent Debate** | 多个 Agent 讨论，决策更稳健 | 成本 N 倍（每个 Agent 都要调 LLM） | 高风险决策（医疗、金融） |
| **Hierarchical（嵌套 Agent）** | Leader 可以调用子 Leader | 复杂度爆炸，调试困难 | 超大规模任务（企业级 workflow） |
| **Router + Worker（不规划直接路由）** | 更快，少一次 LLM 调用 | 没有全局规划，容易走偏 | 客服路由、简单分类任务 |

**结论**：对于"长时间运行的科研实验"这个场景，Leader-Worker 是最优解——用最少的 LLM 调用次数换取最大的战略清晰度。面试时你不需要只说"用了 Leader-Worker"，而是能说"为什么这个场景不用 Debate 或纯 ReAct"。

### 概念 2：Multi-Provider 适配

**为什么选 if-else 而不是工厂模式/注册表？**

```python
# 现状（行 372-378）
if self.provider == "claude_cli": return ...
if self.provider == "codex_cli":  return ...
if self.provider == "openai":     return ...
return self._call_anthropic(...)
```

**原因**：只有 4 个后端，每个后端的调用方式差异巨大（SDK vs subprocess、不同的参数格式），抽象成统一接口反而增加理解成本。

**业界主流做法**：

| 方案 | 适用场景 | 本项目为什么不用 |
|------|---------|----------------|
| **Adapter 模式（统一接口）** | 后端 > 5 个，或频繁增删 | 只有 4 个，过度设计 |
| **if-else 分支（本项目）** | 后端 ≤ 5 个，且差异大 | ✅ 刚好合适 |
| **LiteLLM / LangChain 等中间层** | 需要几十种模型统一调用 | 引入依赖、增加抽象层、出问题时难调试 |
| **Provider Registry + 自动发现** | 插件式架构 | 本项目不需要热插拔后端 |

**改进方向**：如果扩展到 10+ 后端，可以改成注册表模式：
```python
# 业界改进方向
class LLMProvider(Protocol):
    def call(self, system: str, messages: list) -> str: ...

PROVIDERS: dict[str, LLMProvider] = {
    "anthropic": AnthropicProvider(),
    "openai": OpenAIProvider(),
    ...
}
# 新增后端只需注册，不用改 _call_llm
```

### 概念 3：Text Protocol Tool-Use

**为什么不用 Anthropic/OpenAI 原生 tool-use？**

这是本项目最反直觉的设计决策之一。

**现状**：自己定义 `<tool_call>{json}</tool_call>` 文本协议，正则提取。

**原因**：

| 维度 | 原生 tool-use | 文本协议（本项目） |
|------|-------------|-----------------|
| **跨平台** | 每个 provider 的 tool 格式不同 | 一套协议通吃 4 个后端 |
| **CLI 支持** | `claude -p` 不支持原生 tool-use | 文本协议在 CLI 下也能用 |
| **可控性** | LLM 可能绕过 tool schema | 正则提取，格式不对就跳过 |
| **调试** | SDK 内部处理，黑盒 | 日志里直接看到 `<tool_call>` 原文 |
| **锁定风险** | 换 provider 要重写 tool 调用逻辑 | 换 provider 只改变 `_call_*` |

**业界主流做法**：

| 方案 | 何时用 |
|------|--------|
| **原生 tool-use（Anthropic/OpenAI SDK）** | 只用一种 LLM provider 时——最可靠、有类型校验 |
| **文本协议（本项目）** | 需要跨多个 provider 时——最大兼容性 |
| **MCP (Model Context Protocol)** | 工具定义与 LLM 解耦——Anthropic 提出的开放标准 |
| **Function Calling (OpenAI 风格)** | OpenAI 生态——事实标准，很多非 OpenAI 模型也兼容 |

**改进方向**：如果只锁定一个 provider（比如只用 Anthropic），可以切换到原生 tool-use，获得：
- 更低的解析错误率（LLM 被训练过原生格式）
- 自动的 tool 参数校验
- 流式输出（streaming tool calls）

但代价是失去多 provider 兼容性。**这是 tradeoff，不是"文本协议不好"。**

### 概念 4：Prompt Caching（提示词缓存）

**为什么选 Anthropic 的 ephemeral caching？**

```python
# 行 404
"cache_control": {"type": "ephemeral"}
```

**通俗解释**：每次调 LLM 时，system prompt（Leader 的提示词 ~2000 tokens + Worker 的工具 schema ~1000 tokens）都完全相同。不缓存的话，这 3000 tokens 每次都要原样发送+计费。加了 `cache_control` 标记后，Anthropic 服务器记住这 3000 tokens，后续调用只需要发送"新内容"（user message），缓存的部分**不重复计费**。

```
不加缓存，每轮的计费：
  input: system(3000 tokens) + user(500 tokens) = 3500 tokens → $0.01

加缓存后，每轮的计费：
  第一轮: system(3000 tokens, 写入缓存) + user(500 tokens) = 3500 tokens → $0.01
  后续轮: system(3000 tokens, 缓存命中，按 10% 计费) + user(500 tokens) → $0.001
                                                                    ↑ 省了 90%
```

对于跑几百个周期的 24/7 Agent，这个差异累计起来是几十美元和几美元的区别。**一行代码，省 90% 的 system prompt 费用。**

> **初学提示**：两种缓存策略的区别：
> - **Ephemeral caching（Anthropic，显式标记）**：客户端在 content block 上加 `cache_control: {"type": "ephemeral"}`，服务器缓存这个 block 5 分钟。每次命中缓存重置 TTL。你知道**哪块被缓存了**，计费时 cached input tokens 价格约 10%。
> - **Automatic caching（OpenAI，自动检测）**：服务器自动检测多个请求的相同前缀，不需要客户端标记。但你不知道**哪次调用命中了缓存**——只能从 usage 字段的 `cached_input_tokens` 推断。
>
> Anthropic 的显式标记更可控——你知道"这一块一定会被缓存"，不会出现"以为缓存了其实没有"的意外。

**有没有更好的选法？**

| 方案 | 成本 | 复杂度 | 适用场景 |
|------|------|--------|---------|
| **Ephemeral caching（本项目）** | 缓存命中时 ~90% off | 一行代码 | Anthropic only |
| **不缓存，每次全传** | 原价 | 零 | 短 prompt，低频调用 |
| **手动缓存管理** | 更精细控制 | 需要跟踪 cache TTL | 高频、长周期、精确控制成本 |
| **摘要代替全量** | 固定低成本 | 需要摘要逻辑 | 超长 system prompt |

**改进方向**：OpenAI 在 2025 年也推出了类似功能（automatic caching），可以考虑添加：
```python
# 如果切换到 OpenAI 的 automatic caching
# 不需要手动标记，OpenAI 自动检测重复前缀
```

但 Anthropic 的显式标记更可控——你知道什么被缓存了什么没有，不会出现"以为缓存了其实没有"的意外。

---

## 关键代码解析

### 1. CLI 降级方案：`_call_claude_cli()`（行 515-534）

```python
argv = ["claude", "-p", "--output-format", "text", "--tools", ""]
```

`--tools ""` 是最关键的一个 flag——**禁用 claude CLI 的内置工具**。如果不加这个，CLI 会跳过 `<tool_call>` 协议，自己直接调文件系统、shell。那样的话 Worker 的工具调用完全绕过 ToolRegistry，安全防线形同虚设。

`codex_cli` **没有这个选项**（行 539-545 注释说明了），这也是为什么 `codex_cli` 不适合当 Worker——它会"越狱"执行。

**codex_cli 的补救措施**：虽然关不掉内置工具，但 `_call_codex_cli()`（`agents.py:555-604`）用 **tempfile 方案**来降低风险——通过 `codex exec -o <tempfile>` 参数把最终回复写入文件，框架只读取那个文件作为结果，忽略 codex 的 agentic trace（中间的工具调用输出不出现在 stdout 里）。`--skip-git-repo-check` 跳过 git 仓库检查（这个项目不需要）。最后在 `finally` 块里删除临时文件。

```python
# agents.py:560-569 (简化)
with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as out:
    out_path = out.name
result = subprocess.run(["codex", "exec", "--skip-git-repo-check",
                          "-o", out_path, prompt], ...)
# 读取 -o 指定的输出文件（只有最终回复，不含 agentic trace）
with open(out_path, "r") as f:
    return f.read().strip()
```

**CLI 输入是怎么组装的**：`_flatten_for_cli()`（`agents.py:449-464`）把结构化的 `(system, messages)` 转成 CLI 能吃的纯文本：

```python
# agents.py:449-464
@staticmethod
def _flatten_for_cli(system: str, messages: list) -> str:
    parts = [f"===== SYSTEM =====\n{system.strip()}\n"]
    for msg in messages:
        role = str(msg.get("role", "user")).upper()
        content = str(msg.get("content", "")).strip()
        parts.append(f"===== {role} =====\n{content}\n")
    parts.append("===== ASSISTANT =====\n")
    return "\n".join(parts)
```

用 `===== ROLE =====` 标记分隔不同角色的消息，因为 CLI 子进程不接收结构化的 messages 数组——它只接受一段纯文本。最后的 `===== ASSISTANT =====` 提示 CLI 从这里开始输出助手回复。

**argv 太大怎么办**：`_run_cli()`（`agents.py:498-502`）有 E2BIG 兜底——当系统 prompt 太长导致 argv 超过 OS 限制（Linux 通常 2MB），触发 `errno == 7` (E2BIG) 后会通过 **stdin 管道**重试，绕过 argv 大小限制：

```python
# agents.py:498-502
except OSError as e:
    if not use_stdin and getattr(e, "errno", None) == 7:  # E2BIG
        logger.info(f"{tool_label} argv exceeded OS limit; retrying via stdin.")
        return self._run_cli(argv, prompt, tool_label, install_hint, use_stdin=True)
    raise
```

### 2. 输入格式化：`_format_leader_input()`（行 614-642）

Leader 收到的输入不是原始 dict，而是被格式化成一个结构化的 Markdown 文本：

```markdown
## Task: THINK
## Human Directive (HIGHEST PRIORITY)
...（如果有）
## Project Brief
...
## Memory Log
...
## Active Violations
...
## Phase Gate
...
## Cycle: 5
```

**为什么不是 JSON？** 因为 LLM 更擅长理解 Markdown。这是提示词工程的一个重要实践——给 LLM 的输入用 Markdown 结构化，比纯 JSON 或纯文本效果好。

### 3. Worker 配置表：`WORKER_CONFIGS`（行 53-72）

```python
WORKER_CONFIGS = {
    "idea":    {"prompt_file": "idea_agent.md",    "max_turns": 12, "tools": [5 tools]},
    "code":    {"prompt_file": "code_agent.md",    "max_turns": 40, "tools": [7 tools]},
    "writing": {"prompt_file": "writing_agent.md", "max_turns": 30, "tools": [4 tools]},
}
```

**为什么 idea_agent 只有 12 轮 max_turns，而 code_agent 有 40 轮？**

- Idea Agent：搜索论文 → 读摘要 → 形成假设。每步相对独立，12 轮足够。
- Code Agent：写代码 → 干跑 → 看报错 → 改代码 → 再干跑 → 启动训练。需要更多轮次来调试。
- Writing Agent：读结果 → 写报告 → 检查格式。30 轮用于生成长报告。

这个配置反映了对每种 Agent 工作量的实际估算。

---

## 设计决策分析

### 决策：Leader 单轮 vs 多轮思考

**原因**：Leader 的职责是"基于所有可用信息做一个决策"，不调工具，不需要多轮。如果需要多轮思考（比如先分析日志再决定），那应该让 Worker 去做。

**问题**：单轮 LLM 调用可能不够深入。复杂的战略决策可能需要 Leader 先"想"再"想"。

**业界做法**：Google DeepMind 的 DRIVE 论文中，Leader 可以用 Chain-of-Thought 做多步推理，但仍然是**一次 LLM 调用**（长输出）。本项目的 Leader prompt 里有 Decision Framework，在一定程度上引导了 CoT 推理。

### 决策：Worker 每次派工都是全新对话

**原因**：防止不同任务之间的污染。如果 Worker 保留了上次派工的对话历史，它可能"记住"了无关的信息，影响当前任务。

**问题**：Worker 无法利用之前的经验。如果两次派工是同一个 task 的延续，Worker 要从头开始。

**改进方向**：可以在派工时选择性注入一些历史摘要（来自 memory），让 Worker 有"上次做到哪了"的上下文，但不带完整的对话历史。

### 决策：CLI 模式用 subprocess 而不是 SDK

**原因**：订阅制（Claude Pro/Max $20/月 或 ChatGPT Plus $20/月）比 per-token API 便宜得多。通过 CLI subprocess 调用可以利用已有订阅。

**问题**：
- CLI 是文本输入/输出，没有原生 tool-use → 必须依赖文本协议
- `codex_cli` 无法禁用内置工具 → Worker 不可靠
- subprocess 启动有开销（~几百ms）
- 没有 prompt caching

**结论**：如果你有订阅，用 `claude_cli` 是最便宜的方案。如果你追求可靠性，用 `anthropic` SDK + API key。

---

## 本块要掌握的代码

| 代码 | 位置 | 掌握标准 |
|------|------|---------|
| `dispatch_leader()` | `agents.py:146-172` | 能说出 Leader 的三步流程（load prompt→call LLM→parse JSON），知道 leader_history 何时 reset |
| `dispatch_worker()` 工具循环 | `agents.py:174-271` | 能画出 6 步循环（LLM→parse→无tool则break→append→execute→feed result），知道为什么这是个 ReAct 循环 |
| `_call_llm()` 路由 | `agents.py:358-378` | 知道 4 种 provider 的 if-else 分支，为什么用 if-else 而不是工厂模式 |
| `_parse_tool_calls()` | `agents.py:278-306` | 知道代码块过滤（`_FENCED_BLOCK_RE`）的原因，JSON 解析失败时的静默处理 |
| `_render_tools_section()` | `agents.py:309-356` | 知道 tool schema dict 是怎么变成 Markdown 文本注入 system prompt 的 |
| `_call_anthropic()` prompt caching | `agents.py:380-413` | 知道 `cache_control: ephemeral` 的位置和作用，能说出省了多少成本 |
| `_call_claude_cli()` | `agents.py:515-534` | 知道 `--tools ""` 这个 flag 为什么关键（禁用 CLI 内置工具） |
| `_format_leader_input()` | `agents.py:614-642` | 知道 Leader 输入是 Markdown 不是 JSON，为什么 |
| `WORKER_CONFIGS` 配置表 | `agents.py:53-72` | 能说出三种 Worker 的 max_turns 和工具数量差异及原因 |
| `_parse_worker_response()` | `agents.py:666-706` | 知道 PID 优先从 tool result 取（权威），兜底从文本正则匹配 |

---

## 检验

1. `dispatch_leader()` 和 `dispatch_worker()` 在对话历史管理上有什么根本区别？
2. Worker 的工具调用循环最多跑多少轮？超过后会怎样？
3. 为什么 `codex_cli` 不适合当 Worker？（提示：看行 218-225 的 warning）
4. `<tool_call>` 如果在代码块（``` ```）里会被执行吗？在代码的哪一行阻止的？
5. `_call_anthropic()` 里的 `cache_control` 是干什么的？省了多少成本？
6. Leader 的输入是 JSON 还是 Markdown？为什么这么选？
7. Idea Agent 和 Code Agent 的 max_turns 分别是多少？为什么不一样？
8. 如果 Leader 的输出不是合法 JSON，`_parse_leader_response()` 怎么处理？

---

> 下一步：**块 3 — `core/03-tool-system.md`**，理解 Worker 的"手"：工具怎么定义、怎么执行、怎么防护。
