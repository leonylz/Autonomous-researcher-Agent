# 块 4：记忆与存储 — 上下文管理 + 持久化存储

> ⚠️ **术语提示**：本项目叫"双层恒定记忆"（Two-Tier Constant-Size Memory），这不是业界标准术语。业界说法是 **fixed-size context window management** 或 **tiered memory with FIFO compaction**。PROJECT_BRIEF.md / MEMORY_LOG.md 只是本项目的文件名，不是概念。详见 [总纲术语对照表](../LEARNING_INDEX.md)。

## Agent 概念

本块涉及 **3 个 Agent 设计模式**，解决"Agent 跑久了记忆怎么不爆炸"：

### 概念 1：Constant-Size Memory（固定大小记忆）

```
普通 Agent（上下文无限增长）：
  周期 1:   500 tokens
  周期 10:  5000 tokens    ← 开始变慢
  周期 100: 50000 tokens   ← 超出 context window，开始丢信息
  周期 1000: 崩了

本项目（恒定大小记忆）：
  周期 1:   ~1500 tokens
  周期 10:  ~1500 tokens   ← 完全一样
  周期 100: ~1500 tokens   ← 永远不变
  周期 1000: ~1500 tokens  ← 跑一年也不变
```

**核心思想**：记忆不是"积累"，而是"维护一个固定大小的窗口"。

### 概念 2：Hierarchical Memory（分层记忆）

```
热度    层次         存储            内容                   生命周期
───    ────         ────            ────                   ────
热     工作记忆     LLM context    当前周期的 think/reflect  一个周期
温     短期记忆     MEMORY_LOG.md  里程碑 + 最近决策        自动压缩，老内容丢弃
冷     长期归档     experiments.jsonl  每个实验的完整记录    永久追加，永不删除
                    DEAD_ENDS.md       失败方法清单         永久追加，自动轮转
                    INSIGHTS.md        持久洞察            永久追加，自动轮转
```

类比人脑：
- 工作记忆 = 你正在想的事（几秒）
- 短期记忆 = 今天发生的重要事（几分钟到几小时）
- 长期归档 = 实验记录本（永久）

### 概念 3：Append-Only Storage（只追加存储）

```python
# ledger.py:61 — 只有 "a" (append)，没有 "w" (write/overwrite)
with open(self.path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(entry) + "\n")
```

**永远不修改已有数据，只在末尾追加。** 这个设计原则贯穿三个存储文件。

---

## 代码地图

三个文件，总计 ~460 行：

**`core/memory.py`**（149 行）— 双层恒定记忆：

| 函数（行号） | 职责 |
|-------------|------|
| `get_brief()` (48) | 读 Tier 1：冻结的项目目标 |
| `get_log()` (55) | 读 Tier 2：滚动的记忆日志 |
| `log_milestone()` (67) | 追加关键结果，超预算则淘汰最旧的 |
| `log_decision()` (79) | 追加决策，超 N 条则淘汰最旧的 |
| `_parse_log()` (96) | 把 MEMORY_LOG.md 解析成 sections |
| `_section_size()` (147) | 计算某个 section 的总字符数（`sum(len(e) for e in entries)`） |
| `_build_content()` (138) | 把 sections dict 重新组装成 MEMORY_LOG.md 文本 |
| `_write_log()` (113) | 写回 + 最终安全兜底截断 |

**`core/ledger.py`**（204 行）— 实验账本（v2 新增）：

| 函数（行号） | 职责 |
|-------------|------|
| `record()` (34) | 追加一条实验记录 |
| `recent()` / `summary()` (85/89) | 取最近 N 条 / 渲染为上下文文本 |
| `detect_stagnation()` (137) | 纯 Python 停滞检测（块 1 的 _enrich_context 用到） |
| `check_phase_gate()` (189) | 纯 Python 门槛检测 |

**`core/journal.py`**（108 行）— 研究日志：

| 类/函数 | 职责 |
|---------|------|
| `_AppendOnlyDoc` (25) | 通用的只追加文档（DEAD_ENDS + INSIGHTS 共用） |
| `append()` (37) | 追加条目，超大小自动轮转到 .bak |
| `tail()` (73) | 取尾部 N 字符注入 context |
| `ResearchJournal` (89) | 封装两个 _AppendOnlyDoc |

---

## 调用链追踪

### 第 1 步：Tier 1 — PROJECT_BRIEF.md（永不修改）

```python
# memory.py:48-53
def get_brief(self):
    content = self.brief_path.read_text()
    return content[:self.brief_max]   # 截断到 3000 字符
```

**这是人类写的，Agent 只读不写。** 连 `log_memory` 工具都不能修改它。这就是 Tier 1 "冻结"的含义。

### 第 2 步：Tier 2 — MEMORY_LOG.md（自动压缩）

从 `_reflect()` 阶段开始追踪：

```
loop.py:278  _reflect(execute_result)
    │
    ▼ （Leader 的 reflect 决策说"记录这个结果"）
memory.py:67  log_milestone("[cycle 5] ResNet-18 test acc: 72.3% (+2.1%)")
    │
memory.py:69  sections = self._parse_log()        ← ① 解析当前 log
memory.py:71  sections["milestones"].append(...)   ← ② 追加新条目
memory.py:74  while section_size > milestone_max:  ← ③ 超预算？
memory.py:75      sections["milestones"].pop(0)    ← ④ 淘汰最旧的
memory.py:77  self._write_log(sections)            ← ⑤ 写回文件
```

**关键观察**：`pop(0)` 删除的是列表第一个元素——最早的条目。这是 **FIFO（先进先出）淘汰策略**。

### 场景演练：MEMORY_LOG.md 的"自动压缩"到底长什么样？

假设 `milestone_max=1200`（里程碑部分最多 1200 字符），当前 MEMORY_LOG.md 的里程碑区有 8 条：

```
## Key Results
[06-01 10:30] ResNet-18 baseline: acc=68.2%                         ← 180 字符
[06-01 14:15] ResNet-18 + data augmentation: acc=71.5%               ← 210 字符
[06-01 18:00] ResNet-18 + lr=0.001: acc=70.1% (worse)               ← 195 字符
[06-02 09:20] ResNet-18 + weight_decay=1e-4: acc=72.3%              ← 220 字符
[06-02 13:45] ResNet-18 + label_smoothing=0.1: acc=73.1%            ← 230 字符
[06-02 17:30] ViT-B/16 baseline: acc=75.8%                           ← 185 字符
[06-03 08:00] ViT-B/16 + DropPath=0.1: acc=76.9%                    ← 215 字符
[06-03 12:30] ViT-B/16 + lr_warmup: acc=77.4%                        ← 200 字符
                                                                       ─────────
                                                                       1835 字符 ← 超了！
```

现在 Agent 想追加第 9 条：`[06-03 16:00] ViT-B/16 + mixup: acc=78.2%`（190 字符）。

**压缩过程**：

```
追加后 size = 2025 字符 > 1200 milestone_max

while 2025 > 1200:
    pop(0)  → 删除 "[06-01 10:30] ResNet-18 baseline: acc=68.2%"
    size = 1845，仍然 > 1200

while 1845 > 1200:
    pop(0)  → 删除 "[06-01 14:15] ResNet-18 + data augmentation: acc=71.5%"
    size = 1635，仍然 > 1200

while 1635 > 1200:
    pop(0)  → 删除 "[06-01 18:00] ResNet-18 + lr=0.001: acc=70.1%"
    size = 1440，仍然 > 1200

while 1440 > 1200:
    pop(0)  → 删除 "[06-02 09:20] ResNet-18 + weight_decay=1e-4: acc=72.3%"
    size = 1220，仍然 > 1200

while 1220 > 1200:
    pop(0)  → 删除 "[06-02 13:45] ResNet-18 + label_smoothing=0.1: acc=73.1%"
    size = 990，OK！
```

**压缩后的 MEMORY_LOG.md**：

```
## Key Results
[06-02 17:30] ViT-B/16 baseline: acc=75.8%                           ← 保留
[06-03 08:00] ViT-B/16 + DropPath=0.1: acc=76.9%                      ← 保留
[06-03 12:30] ViT-B/16 + lr_warmup: acc=77.4%                         ← 保留
[06-03 16:00] ViT-B/16 + mixup: acc=78.2%                             ← 新加的
                                                                         ─────────
                                                                         990 字符 ✅
```

**关键洞察**：
- 最早的 5 条（ResNet-18 相关）被淘汰了——但它们没有"丢失"，完整记录在 `experiments.jsonl` 里
- MEMORY_LOG.md 只保留**最近的关键里程碑**，给 Leader 最新的上下文
- Leader 看到的始终是"最近的趋势"（ViT 系列在稳步提升），不会被 3 天前的 ResNet 实验干扰
- 如果 Leader 需要查 ResNet-18 的历史细节，`_enrich_context` 会从 JSONL 注入 `recent_experiments` 摘要

**这就是三层记忆协同工作的方式**：MEMORY_LOG 给"趋势感知"，JSONL 给"完整历史"，context 给"当前决策"。

### 第 3 步：_write_log 的最终安全兜底（行 127-134）

```python
# 写之前再检查一遍：总大小不超过 log_max (2000 chars)
if len(content) > self.log_max:
    while len(content) > self.log_max and len(sections["milestones"]) > 1:
        sections["milestones"].pop(0)        # 先砍里程碑
    while len(content) > self.log_max and len(sections["decisions"]) > 1:
        sections["decisions"].pop(0)         # 再砍决策
```

这是**防御性编程的经典实践**：即使前面的 compact 逻辑有 bug，这里还有一道最终防线。永远不会写出超出预算的文件。

**为什么需要两个 while 循环而不仅靠 per-section compact？** 考虑这个场景：

```
milestones 部分：900 字符（在 1200 字符上限内 ✅）
decisions 部分：1300 字符（在 15 条上限内 ✅）
总字符数：2200 字符（超过 log_max=2000 ❌）
```

每个部分单独看都合规，但组合后溢出。per-section compact（`log_milestone`/`log_decision` 各自的 while 循环）不会触发——它们只检查各自的预算。`_write_log` 的双 while 循环能捕捉这种跨部分的总量溢出：先砍 milestones（低优先级），不够再砍 decisions。

### 第 4 步：experiments.jsonl — 完整实验记录

```
loop.py:266  self.ledger.record(cycle=5, hypothesis="...", metrics={"acc": 0.72}, ...)
    │
ledger.py:34  def record(self, ...)
ledger.py:49      entry = {"ts": ..., "cycle": ..., "metrics": ..., ...}
ledger.py:61      with open(self.path, "a") as handle:    ← "a" = append only
ledger.py:62          handle.write(json.dumps(entry) + "\n")
```

**JSONL 格式** = 每行一个独立 JSON 对象。好处：
- 人可以直接 `tail` 看
- 程序可以逐行解析，不需要一次加载全部
- 追加不涉及重写整个文件
- 程序崩溃不会损坏已有数据（不像 SQLite 可能锁表）

### 第 5 步：_enrich_context 怎么消费这些存储？

回到块 1 的 `_enrich_context()`（loop.py:383-447），现在你能理解每个信号的来源：

```python
context["recent_experiments"] = self.ledger.summary(5)     ← 来自 JSONL
context["progress_signal"]    = self.ledger.detect_stagnation(...)  ← 纯 Python 计算
context["dead_ends"]          = self.journal.dead_ends_tail(1500)   ← 来自 journal
context["insights"]           = self.journal.insights_tail(1500)    ← 来自 journal
```

**所有这些都是纯 Python 代码，0 次 LLM 调用。** 这就是"把检测和思考分离"的实践。

---

## 每个概念：为什么选这个？有没有更好的？

### 概念 1：Constant-Size Memory

**为什么选固定大小？**

| 需求 | Constant-Size Memory 怎么满足 |
|------|---------------------------|
| 24/7 运行，上下文不能爆炸 | 硬上限：brief ≤ 3000 + log ≤ 2000 = 永远 ≤ 5000 字符 |
| 信息不能完全丢失 | 分层：冷数据归档到 JSONL/journal |
| 成本可控 | 每次 LLM 调用的 input token 固定 |

**业界主流做法**：

| 方案 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **FIFO 截断（本项目）** | 保持总大小不变，淘汰最旧的 | 简单、零成本 | 重要旧信息可能被淘汰 |
| **LLM 摘要压缩** | 让 LLM 定期把旧记忆总结成一段 | 保留语义，信息密度高 | 多花一次 LLM 调用 |
| **重要性加权淘汰** | 每条记忆打分，淘汰最低分 | 保留重要信息 | 需要打分机制（谁来打？） |
| **RAG + 向量检索** | 记忆存外部向量库，按语义检索 | 无限容量，精准检索 | 需要 embedding 模型 + 向量库 |
| **MemGPT/Letta** | OS 风格的虚拟上下文管理 | 自动分页、swap | 复杂度高，依赖特定框架 |

> **MemGPT** 是 UC Berkeley 2023 年的研究项目——把 LLM 的上下文窗口当作操作系统的虚拟内存管理：超出上下文容量时自动将旧内容"换出"（swap out）到外部存储，需要时再"换入"（swap in），给 LLM 一个"无限上下文"的错觉。**Letta** 是 MemGPT 的商业化框架版本，将其包装成可部署的服务。两种方案都能突破本项目的固定 5000 字符限制，但引入额外的框架依赖和换页延迟（每次 swap 都是一次 LLM 调用或 embedding 检索）。

**改进方向**：中等改进是加入 **LLM 摘要压缩**——不是淘汰最旧的，而是让 LLM 把老记忆压缩成一段摘要。成本增加 ~$0.01/次压缩，但信息保留更好。本项目没做是因为 Leader 的 reflect 阶段已经承担了部分"提炼洞察"的功能。

### 概念 2：Hierarchical Memory（三层分离）

**为什么分三层？**

```
工作记忆（context）→ 给 LLM 看的，只在当前周期有效
短期记忆（MEMORY_LOG）→ 跨周期的摘要，自动压缩
长期归档（JSONL + journal）→ 完整记录，永久保存
```

每一层的读写成本完全不对称：
- 工作记忆：**最贵**（占用 LLM context window，按 token 计费）
- 短期记忆：**便宜**（纯文件 I/O，只在 need 时读）
- 长期归档：**几乎免费**（追加一行 JSON，只在 _enrich_context 时读摘要）

**业界主流做法**：

| 方案 | 适用场景 |
|------|---------|
| **三层分离（本项目）** | 长周期科研 Agent，需要完整实验记录 |
| **只有 context（最简单）** | Demo / 短期 Agent |
| **Context + SQLite** | 需要复杂查询（按时间/指标过滤） |
| **Context + Vector DB** | 需要语义检索（"找和 ResNet 相关的所有实验"） |
| **LangChain Memory 模块** | 快速原型，不想自己写 |

**改进方向**：把 JSONL 迁移到 SQLite。好处是可以做复杂查询（"找出所有 accuracy > 70% 且用了 ViT 的实验"），坏处是不能再 `tail experiments.jsonl` 看最近记录了。

### 概念 3：Append-Only Storage

**为什么只追加不修改？**

1. **崩溃安全**：追加一行写入是原子操作（在文件系统层面），程序在任何时候崩溃都不会损坏已有数据。SQLite 的 UPDATE 可能在崩溃时损坏数据库。
2. **审计友好**：每一条记录都保留了，不会被后续操作覆盖。可以完整追溯"Agent 在第 3 轮时做了一个错误的决定"。
3. **简单**：`open("a")` 比任何数据库都简单。没有 schema migration、没有锁、没有连接管理。

**业界主流做法**：

| 方案 | 何时用 |
|------|--------|
| **JSONL 追加（本项目）** | 记录量 < 10 万条，查询简单（取最近 N 条、按指标排序） |
| **SQLite** | 记录量 > 10 万条，需要复杂查询和索引 |
| **PostgreSQL** | 多 Agent 共享数据，需要并发写 |
| **Kafka / Event Sourcing** | 微服务架构，多个消费者订阅实验事件 |

> **Apache Kafka** 是分布式消息队列（不是设计模式，是一个具体的基础设施软件）。实验记录作为事件发布到 Kafka topic，多个下游服务（监控、报表、通知）可以独立订阅消费，互不阻塞。**Event Sourcing** 是一种设计模式——所有状态变更存储为不可变事件序列（"第 3 轮启动了 ResNet-18"、"第 3 轮 accuracy 达到 72%"），可以完整追溯"谁在什么时候做了什么"。本项目的 JSONL 追加本质上是轻量级的 Event Sourcing——只是单机文件而非分布式队列。适合单机 &lt;10 万条记录的场景。

---

## 关键代码解析

### 1. MEMORY_LOG.md 的解析逻辑（行 96-111）

```python
def _parse_log(self):
    content = self.get_log()       # ← 读取 MEMORY_LOG.md 全文
    sections = {"milestones": [], "decisions": []}
    current_section = None
    for line in content.split("\n"):
        if line_stripped == "## Key Results":
            current_section = "milestones"
        elif line_stripped == "## Recent Decisions":
            current_section = "decisions"
        elif line_stripped.startswith("[") and current_section:
            sections[current_section].append(line_stripped)
    return sections
```

**这是一个简单的行解析器**——它依赖 Markdown 的 `## ` 标题来切换 section，依赖 `[...]` 前缀来识别条目。这比用正则或 Markdown parser 简单得多，而且正好够用。

### 2. 日志轮转（journal.py:53-71）

```python
def _rotate(self, stamp):
    content = self.path.read_text(encoding="utf-8")
    backup = self.path.with_name(f"{self.path.stem}.{stamp}.bak")
    backup.write_text(content, encoding="utf-8")
    tail = content[-(self.max_chars // 2):]   # 保留后半
    self.path.write_text(f"# {self.title}\n\n_(rotated; full history in {backup.name})_\n{tail}")
```

当 DEAD_ENDS.md 超过大小上限时：
1. 整个文件归档为 `DEAD_ENDS.2026-08-10_1430.bak`
2. 新文件保留后半部分（最近的内容）+ 一个指向备份的链接

**历史从不丢失，只是移动位置。**

### 3. 停滞检测的算法（ledger.py:137-186）

```python
def detect_stagnation(entries, metric_key, direction, threshold_cycles=3, min_delta=0.0):
    points = _metric_values(entries, metric_key)  # 提取所有 (index, value)
    best_val = points[0][1]
    cycles_since_improvement = 0
    for _, val in points[1:]:
        improved = (val > best_val + min_delta) if higher else (val < best_val - min_delta)
        if improved:
            best_val = val
            cycles_since_improvement = 0     # 重置计数器
        else:
            cycles_since_improvement += 1    # 累加

    verdict["stagnating"] = cycles_since_improvement >= threshold_cycles
```

**这是经典的 Moving Best 追踪**——维护历史最佳值，记录距离上次刷新过了多少轮。不是算滑动平均或趋势线（那些更复杂），而是最简单有效的指标。

**为什么需要 `min_delta` 参数？** 考虑这个真实场景：

```
轮次 1: accuracy = 72.1%
轮次 2: accuracy = 72.3%  ← 比上轮高 0.2%，但可能是随机种子波动
轮次 3: accuracy = 72.0%
轮次 4: accuracy = 72.4%  ← 比最佳高 0.3%，还是波动范围
```

如果不设 `min_delta`（=0.0），每轮都被算作"有进步"——`cycles_since_improvement` 始终为 0，停滞检测永不触发。设 `min_delta=0.5` 后，±0.3% 的波动被正确忽略——只有真正超过 72.6% 才算进步。这个参数把**统计噪声**和**真正提升**区分开。

---

## 设计决策分析

### 决策：FIFO 淘汰 vs LLM 摘要

**原因**：FIFO 淘汰是零成本的。LLM 摘要是要花钱的。对于这个项目的规模（~1500 tokens 的记忆预算），FIFO 够用了。

**改进方向**：当记忆预算增大到 ~5000 tokens 时，FIFO 淘汰可能丢失重要信息。此时 LLM 摘要的价值就体现出来了——花 $0.01 做一次压缩，换取更好的信息保留。

### 决策：JSONL vs SQLite

**原因**：JSONL 在"追加"和"tail 查看"这两种操作上是完美的。而实验记录恰好只需要这两种操作。

**什么时候该换 SQLite**：当你需要查询"第 3-10 轮之间有哪些实验的 accuracy 超过了 70%？"时，JSONL 需要加载全部数据再过滤，SQLite 可以用索引快速查询。

### 决策：记忆大小为什么是 3000 + 2000 = 5000 字符？

```
3000 字符（brief）+ 2000 字符（log）= 5000 字符 ≈ 1500 tokens
```

Leader 的 context 还包括 cycle count、recent experiments、dead ends、insights、violations 等。这些加起来大约 2000-3000 tokens。所以 Leader 的总 input 大约 4000-5000 tokens——在 Claude 的 200K context window 里只占 2-3%。**Headroom 巨大，永远不会触及上限。**

如果换一个 context window 只有 8K 的模型，这个 5000 预算就需要更谨慎了。

---

## 本块要掌握的代码

| 代码 | 位置 | 掌握标准 |
|------|------|---------|
| `MemoryManager` 双层初始化 | `memory.py:27-46` | 知道 brief_max/log_max/milestone_max/max_recent 四个参数的默认值和含义 |
| `get_brief()` Tier 1 读 | `memory.py:48-53` | 知道为什么只读不写，为什么截断到 brief_max |
| `log_milestone()` + FIFO 淘汰 | `memory.py:67-77` | 能说出 while 循环逐条 pop(0) 的淘汰过程，为什么淘汰最早的而不是最没用的 |
| `log_decision()` + 数量淘汰 | `memory.py:79-89` | 知道和 milestone 的区别——按条数（max_recent）淘汰而不是按字符数 |
| `_parse_log()` 行解析器 | `memory.py:96-111` | 知道怎么用 `##` 标题切换 section，为什么不用 Markdown parser |
| `_write_log()` 安全兜底 | `memory.py:113-136` | 知道为什么有两层 while（先砍 milestone 再砍 decision），永远不会超预算 |
| `ExperimentLedger.record()` | `ledger.py:34-66` | 知道 `open("a")` 的 append-only 语义，崩溃不会损坏已有数据 |
| `detect_stagnation()` | `ledger.py:137-186` | 能说出 Moving Best 追踪算法：维护历史最佳 → 计数无进步轮次 → 超阈值告警 |
| `_AppendOnlyDoc.rotate()` | `journal.py:53-71` | 知道轮转策略：全量归档 .bak → 保留后半 → 历史从不丢失 |
| `ResearchJournal` 双日志 | `journal.py:89-107` | 知道 DEAD_ENDS（禁止重试）和 INSIGHTS（持久洞察）的区别 |

---

## 检验

1. PROJECT_BRIEF.md 最大多少字符？Agent 能修改它吗？
2. MEMORY_LOG.md 的两个 section 分别是什么？各自用什么策略淘汰旧内容？
3. `_write_log()` 里为什么有两层 while 循环做截断？（提示：一层不够吗？）
4. 如果程序在 `ledger.record()` 执行到一半时崩溃，已写入的数据会损坏吗？（提示："a" mode 和 JSONL 的特性）
5. DEAD_ENDS.md 文件超过大小上限后，旧内容去哪了？
6. `detect_stagnation()` 的 min_delta 参数是干什么的？如果 min_delta=0.01 和 min_delta=0，检测结果会有什么不同？
7. 三层记忆分别存储在什么文件里？（工作记忆/短期/长期归档）
8. 为什么 Leader 的 context 永远不会超限？（提示：记忆预算是固定的）

---

> 下一步：**块 5 — `core/05-monitor-safety.md`**，理解训练期间怎么零成本监控，以及安全怎么分层防御。
