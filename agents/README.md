# 块 6：提示词工程 — Agent 的"源代码"

## Agent 概念

> 💡 提示词是 LLM Agent 的"源代码"。和传统软件开发不同——传统软件的 bug 在代码里，Agent 的 bug 可能在 prompt 的一句话里。

本块涉及 **2 个 Agent 设计模式**：

### 概念 1：Chain-of-Thought (CoT) — 通过提示词引导推理

```
没有 CoT：                          有 CoT（本项目）：
────────                            ────────────────
Leader: "接下来做什么？"             Leader Decision Framework:
→ LLM 直接输出决策                   1. 当前最佳结果是什么？
→ 可能是拍脑袋的结果                  2. 哪些假设没验证过？
                                    3. 基于趋势最有希望的方向？
                                    4. 最小可行实验是什么？
                                    → LLM 沿着框架推理 → 更好的决策
```

**CoT 不是写 `"请一步步思考"` 就完了。** 好的 CoT 提示词是给 LLM 一个**结构化的推理框架**——不是让它自由联想，而是引导它按特定的 checklist 走。

### 概念 2：Prompt Engineering as Constraint Design（约束设计）

提示词不仅仅是"告诉 LLM 做什么"，更是**约束 LLM 不能做什么**：

```markdown
# Leader prompt 的约束（leader.md:48-52）
- Never modify PROJECT_BRIEF.md          ← 硬约束：禁止行为
- Keep task descriptions self-contained  ← 软约束：格式要求
- Maximum 3 sub-agent dispatches         ← 数量约束：防止失控
- Always include success criteria        ← 质量约束：强制可验证
- Prefer small, fast experiments         ← 策略约束：成本控制
```

这些约束不是"建议"——它们在 LLM 的输出里起到结构性限制的作用。

---

## 代码地图

4 个提示词文件，总计约 160 行：

| 文件 | Agent | 核心设计 |
|------|-------|---------|
| `leader.md` (53 行) | Leader | Decision Framework（CoT 推理框架）+ JSON 输出格式 |
| `code_agent.md` (67 行) | Code Agent | 5 步强制工作流 + dry-run 硬性要求 |
| `idea_agent.md` (42 行) | Idea Agent | Snowballing 文献检索策略 + 2 步搜索策略 |
| `writing_agent.md` (43 行) | Writing Agent | 结构化报告模板 |

提示词被加载的位置：

```
agents.py:159  system_prompt = _load_prompt("leader.md")    ← Leader
agents.py:207  base_prompt = _load_prompt("code_agent.md")  ← Worker
agents.py:209  system_prompt = base_prompt + tools_section  ← 拼接工具协议
```

**`_load_prompt()` 的实现**（`agents.py:606-612`）——极其简单，但有一个重要的隐含行为：

```python
# agents.py:606-612
def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parent.parent / "agents" / name
    return path.read_text().strip()
```

**关键**：这个函数**不做任何解析**——直接把整个 `.md` 文件（包括 `---` YAML frontmatter）作为原始文本返回。这意味着 prompt 文件开头的 YAML 元数据：

```yaml
---
name: Code Agent
description: Writes code and launches experiments
model: inherit
---
```

**会原样发给 LLM 作为 system prompt 的一部分**。`model: inherit` 字段是给 Python 调度层（`_call_llm`）读取的，不是 `_load_prompt` 的事。LLM 看到这些元数据不会造成问题——它只是把它们当成 prompt 开头的一段说明文字。

---

## 逐个拆解

### 1. Leader 提示词（leader.md）— 战略指挥官

```markdown
## Decision Framework

When thinking about the next experiment:
1. What is the current best result?                       ← 基准锚定
2. What hypotheses haven't been tested?                    ← 避免重复
3. What is the most promising direction based on recent trends?  ← 趋势感知
4. What is the minimum viable experiment to test this hypothesis? ← 成本控制（关键！）
```

**最关键的是第 4 条**："minimum viable experiment"（最小可行实验）。这个约束防止 Leader 设计一个需要跑 3 天的实验来验证一个小假设。在科研里这叫"先跑个小规模试点"。

**面试时可以说的点**："我在 prompt 里引入了 MVP 实验设计原则——每个实验只验证一个假设，用最小规模跑。这相当于软件工程里的'每次只改一个变量'原则。"

```markdown
## Output Format
Always respond with a JSON block:
{
  "action": "experiment|wait|report",   ← 有限状态机
  "agent": "code|idea|writing",         ← 路由决策
  "task": "...",                         ← 自包含任务描述
  "hypothesis": "...",                   ← 强制声明假设（事后可验证）
  "success_criteria": "...",            ← 强制定义成功标准
  "milestone": "...",
  "decision": "..."
}
```

**为什么要求输出 JSON？** 因为 `_parse_leader_response()` 用正则提取 JSON。如果 Leader 输出纯文本，解析可能出错。**结构化输出 = 可编程消费 = 减少解析错误。**

**为什么要求 `hypothesis` 和 `success_criteria`？** 这强制执行科学方法：每个实验都有假设和验证标准。如果没有这两个字段，Agent 可能变成"随机尝试不同参数"——这不是科研，是调参。

### 2. Code Agent 提示词（code_agent.md）— 战术执行者

```markdown
## Mandatory Workflow

Step 0: Explore the codebase first     ← 防止瞎改（v2 新增）
Step 1: Understand                     ← 理解任务
Step 2: Implement                      ← 修改代码
Step 3: Dry-Run (MANDATORY)           ← 核心安全机制！
Step 4: Launch                         ← 正式启动
Step 5: Report                         ← 回报 PID
```

**Step 0 是 v2 新增的**——在改代码之前先用 `list_tree` + `search_code` 探查代码库。这解决了 v1 的一个痛点：Agent 会猜测文件名和参数名，然后写一个不存在的文件或调一个不存在的 flag。

**Step 3 是安全的关键**：`--max_steps 2 --dry_run` 让训练只跑 2 步就停下。如果代码有语法错误、import 失败、OOM，在这 2 步里就会暴露。这省去了"启动训练→等 5 分钟→崩溃→看日志→改代码→再启动"的循环。

> ⚠️ **注意**：`--max_steps` 和 `--dry_run` 不是本项目框架提供的内置参数——它们是 Code Agent prompt 里的**示例命令**。你的训练脚本需要自己支持这些参数：
> - PyTorch Lightning：用 `--fast_dev_run` 代替
> - HuggingFace Trainer：可能需要另写一个验证脚本
> - 自定义训练脚本：需要自己在 argparse 里加 `--dry_run` 和 `--max_steps` 参数
> 
> 如果框架不匹配，你需要修改 `code_agent.md` 里的 dry-run 命令来适配你的训练框架。

**面试时可以说的点**："我在 Code Agent 的 prompt 里强制了 5 步工作流，其中 Step 0（探查代码库）和 Step 3（强制 dry-run）是最关键的安全机制。Dry-run 相当于 CI 里的 smoke test——不通过就不允许正式训练。"

### 3. Idea Agent 提示词（idea_agent.md）— 学术侦察兵

```markdown
2. Cast a wide net: search_arxiv for the newest work AND
   search_papers for established, well-cited work
3. Pick the 2-3 most relevant papers and call get_paper
   on each, then snowball: walk their references and
   citations to find the closely-related cluster
```

**两个搜索的区别**：
- `search_arxiv`：找最新的（几天内的预印本）
- `search_papers`（Semantic Scholar——Allen AI 研究所的免费学术搜索引擎，覆盖 2 亿+ 论文，提供引用次数和引用关系图）：找最经典的（有引用数的已发表论文，经过同行评审）

**Snowballing（雪球滚动）** 是文献检索的经典方法——找一篇好论文，然后顺着它的引用链往前（references）和往后（citations）探索。一步好的 snowballing 比十次关键词搜索更能覆盖一个子领域。

### 4. Writing Agent 提示词（writing_agent.md）— 报告写手

```markdown
## Output Format
| Experiment | Config | Metric | Notes |
|------------|--------|--------|-------|
```

Writing Agent 最特别的设计是**固定表格格式**。不是让 LLM 自由发挥，而是规定好表格的列——这样 Leader 在 reflect 时看到的是标准化的结果，更容易做对比。

---

## 提示词设计方法论：从这 4 个 prompt 里学到的

### 原则 1：给框架，不给答案

❌ 不好的 prompt：`"你是一个 AI 研究员，设计下一个实验。"`
✅ 好的 prompt（leader.md）：`"按这 4 步思考：1. 当前最佳？2. 未验证假设？3. 最有希望的方向？4. 最小可行实验？"`

**框架 > 自由发挥**。LLM 在有框架时比自由发挥时更可靠。

### 原则 2：MUST > should

```markdown
# code_agent.md
**You MUST do a dry-run before launching real training.**
Do NOT skip to real training.
```

大写 MUST + 粗体 + 第二句重复。这不是啰嗦——LLM 对强调的敏感度和人类不同。多重强调可以显著降低"跳过 dry-run"的概率。

### 原则 3：约束数量，不给无限自由

```markdown
# leader.md
Maximum 3 sub-agent dispatches per cycle
```

如果不限制，Leader 可能在一个周期内派 10 个 Idea Agent 去搜文献。限制数量 = 控制成本。

### 原则 4：要求结构化输出

```markdown
# leader.md: Always respond with a JSON block
# writing_agent.md: 固定表格格式
```

结构化输出不是为了让代码好解析——虽然这也是好处——而是**约束 LLM 不遗漏关键信息**。如果 Leader 输出纯文本，它可能只说了"跑一个 ResNet 实验"，没说成功标准是什么。JSON 的 `success_criteria` 字段强制它填这个。

### 原则 5：自包含的任务描述

```markdown
# leader.md
Keep task descriptions self-contained (workers are stateless)
```

Worker 每次派工是新对话（块 2 分析过），所以任务描述必须包含 Worker 需要的所有上下文。不能写"继续上次的实验"——上次的实验已经不在 Worker 的记忆里了。

> 💡 **以上 5 条原则是我从本项目 4 个 prompt 里归纳的，不是来自论文或标准。** 不同项目、不同模型、不同任务类型可能归纳出不同的原则。MUST vs should 的大小写效果、框架 vs 自由的可靠性差异——这些都是实践经验，没有基准测试数据支持。在面试中引用时，建议表述为"我在实践中观察到..."而不是"业界公认的原则是..."。

---

## 每个概念：为什么选这个？有没有更好的？

### 概念 1：写死的工作流 vs 让 LLM 自己决定步骤

**为什么选固定工作流？**

```markdown
# code_agent.md — 5 步强制工作流
Step 0 → Step 1 → Step 2 → Step 3 (MUST dry-run) → Step 4 → Step 5
```

**原因**：Code Agent 的可靠性比灵活性重要。跳过 dry-run 的代价是几小时的 GPU 时间（跑一个会崩的训练）。固定工作流消除了这个风险。

**业界主流做法**：

| 方案 | 适用场景 |
|------|---------|
| **固定工作流（本项目）** | 高风险、高成本操作（训练） |
| **LLM 自由决定步骤** | 低风险、创意性任务（写报告） |
| **混合：关键步骤固定 + 非关键自由** | 最佳实践——本项目其实也是这样（dry-run 固定，实现细节自由） |

### 概念 2：Markdown 提示词 vs 结构化配置

**为什么提示词用 Markdown 文件而不是 YAML/JSON 配置？**

四个原因：
1. **LLM 训练数据里 Markdown 最多**——LLM 对 Markdown 的理解最自然
2. **人可以直观编辑**——不需要了解配置格式
3. **层次清晰**——`##` 标题天然分层
4. **可以用粗体/列表/代码块**——这些格式 LLM 都理解

**改进方向**：大规模部署时可以用 Jinja2 模板生成 prompt。Jinja2 是 Python 的文本模板引擎——在 Markdown prompt 里插入变量和条件逻辑（如 `{% if task_type == 'cv' %} 使用 CNN 架构 {% endif %}`），根据项目类型在运行时动态拼装不同的系统提示词，替代当前四个硬编码 `.md` 文件。本项目规模小，不需要。

### 概念 3：YAML frontmatter 的用途

```yaml
---
name: leader
description: Central decision-maker that plans experiments and reflects on results
model: inherit
---
```

`model: inherit` 表示不强制指定模型，跟随 dispatcher 的配置。如果某类任务需要更强的模型（比如 Leader 决策比 Writer 更需要 Opus 级别的推理），可以改成 `model: claude-opus-4-6`。

---

## 魔改入口：你最可能改的提示词

### 改 Leader prompt（改战略风格）

- 加更多决策维度："What does the literature say about this approach?"
- 改实验策略：把 "prefer small experiments" 改成 "prefer thorough ablation studies"

### 改 Code Agent prompt（适配你的训练框架）

- 把 `python train.py --max_steps 2 --dry_run` 改成你的框架的 dry-run 命令
- 加框架特定的约束："Never use `torch.compile` with `--dry_run`"
- 把 metric 提取的示例改成你的指标名称

### 改 Idea Agent prompt（改文献检索策略）

- 加更多数据源（比如 ArXiv 的特定 category）
- 改 snowballing 的深度（1 hop vs 2 hops）

---

## 本块要掌握的代码

| 代码 | 位置 | 掌握标准 |
|------|------|---------|
| Leader Decision Framework (4 步) | `leader.md:18-22` | 能说出 4 个决策步骤，为什么第 4 步（最小可行实验）最关键 |
| Leader 输出 JSON 格式 (7 字段) | `leader.md:34-43` | 知道 `action`/`agent`/`task`/`hypothesis`/`success_criteria` 各自被哪段 Python 代码消费 |
| Leader 5 条约束 | `leader.md:48-52` | 能解释每条约束防止什么问题（如"max 3 dispatches"是成本控制） |
| Code Agent 6 步工作流 | `code_agent.md:20-67` | 知道 Step 0（探查代码库）和 Step 3（强制 dry-run）为什么是最关键的两步 |
| Code Agent dry-run 命令 | `code_agent.md:42-46` | 知道 `--max_steps 2 --dry_run` 的原理：只跑 2 步验证无语法错误 |
| Idea Agent 双源搜索 | `idea_agent.md:22-23` | 知道 `search_arxiv`（最新预印本）和 `search_papers`（已发表+引用数）的区别 |
| Idea Agent snowballing 策略 | `idea_agent.md:25-26` | 知道怎么顺着 reference/citation 图从一篇好论文扩展到相关领域 |
| Writing Agent 固定表格 | `writing_agent.md:28-37` | 知道为什么规定固定表格列而不是让 LLM 自由发挥 |

---

## 检验

1. Leader 的 Decision Framework 有几步？为什么第 4 步（MVP）最关键？
2. Code Agent 的 Step 0 是干什么的？为什么 v2 才加上？
3. `search_arxiv` 和 `search_papers` 分别用于什么场景？在哪一行提示词里体现的？
4. Writing Agent 的报告为什么要规定固定表格格式？
5. Leader 的输出 JSON 里有 `hypothesis` 和 `success_criteria` 两个字段——为什么两个都要，而不是只用一个？
6. 提示词里的 `MUST` 大写和 `Do NOT` 大写——是风格问题还是功能性要求？
7. 如果把 Code Agent 的 dry-run 步骤从 prompt 里删掉，Agent 的行为会怎么变？
8. 为什么 Leader 和每个 Worker 都强调了"不要修改 PROJECT_BRIEF.md"？在哪些文件里出现了这个约束？

---

> 🎉 **全部 6 块完成！** 回到 [`LEARNING_INDEX.md`](../LEARNING_INDEX.md) 看看你覆盖了多少 Agent 概念，或者看 `MODIFICATION_BLUEPRINT.md` 开始魔改。
