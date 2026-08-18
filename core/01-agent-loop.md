# 块 1：Agent 主循环 — Plan-Execute 模式

## Agent 概念

> 💡 这个项目实际上覆盖了 **6 大类 20+ 种 Agent 设计模式**。详见 [`LEARNING_INDEX.md`](../LEARNING_INDEX.md) 的全景表。本块只聚焦其中最核心的一个。

这个项目实现的是 **Plan-Execute 模式**（也叫 Two-Loop Architecture），是**决策模式**大类下的一种：

```
外层循环（Plan）：  Leader 分析 → 制定计划 → 指派任务 → 等待结果 → 复盘
内层循环（Execute）：Worker 接收任务 → 调工具 → 看结果 → 继续调工具 → 完成

> ⚠️ **术语提示**：本项目叫 "Leader-Worker"，业界通用叫法是 **Orchestrator-Worker** 或 **Plan-Execute**。下文保持项目原始叫法方便对照源码，但你面试时换用业界叫法。详见 [LEARNING_INDEX.md 术语对照表](../LEARNING_INDEX.md)。
```

同一大类下还有 ReAct、CoT、Self-Critique、ToT 等模式。这个项目同时用了其中三种：**Plan-Execute（本块）+ Self-Critique/Reflexion（REFLECT 阶段）+ Chain-of-Thought（提示词中）**。

和纯 **ReAct 模式**（Reasoning + Acting 交替）的区别：

| | ReAct | Plan-Execute（本项目） |
|---|---|---|
| 决策方式 | 想一步做一步，交替进行 | 先完整规划，再派工兵执行 |
| 适合场景 | 信息检索、对话、debug | 长周期任务（训练几小时） |
| LLM 调用 | 每一步都调 LLM | 决策时调，执行时多轮，监控时零调用 |
| 成本 | 高（频繁调用） | 低（训练期间 $0.00） |

**这个项目的创新点**：在 Plan 和 Execute 之间插入了 **Monitor 阶段（OS-level process supervision，本项目叫"零成本监控"）**。训练占 90%+ 时间，但完全不调 LLM——只靠 `kill -0` + `nvidia-smi` + `tail`。

### 场景演练：如果同一个任务用 ReAct 做会怎样？

假设任务："训练一个 ResNet-18 在 CIFAR-10 上达到 85% accuracy"。训练需要 2 小时。

**ReAct 模式（想一步做一步）：**

```
00:00  LLM: "我先写训练代码"        → $0.05
00:01  LLM: "代码写好了，启动训练"    → $0.05
00:02  [训练启动，需要 2 小时]
00:07  LLM: "训练还在跑吗？看一下 GPU" → $0.05   ← 每 5 分钟问一次
00:12  LLM: "训练还在跑吗？..."      → $0.05
00:17  LLM: "训练还在跑吗？..."      → $0.05
      ...（2 小时内反复问 24 次）
02:00  LLM: "训练结束了，看结果"     → $0.05
02:01  LLM: "accuracy 72%，不够，改学习率重试" → $0.05

总计：2 + 24 + 2 = 28 次 LLM 调用，约 $1.40
其中 24 次（$1.20）只是确认"训练还在跑"
```

**Plan-Execute 模式（本项目）：**

```
00:00  Leader._think():    "跑 ResNet-18 baseline"     → $0.05  ← 战略决策
00:01  Worker:              写代码 → 干跑 → 启动训练      → $0.40  ← 战术执行（多轮工具调用）
00:02  Monitor.wait():      sleep(900) × 8 次             → $0.00  ← 零成本！
      在 2 小时内：          kill -0, nvidia-smi, tail     → $0.00
02:00  Leader._reflect():   "acc=72%，下次调学习率"       → $0.05  ← 复盘

总计：2 次 Leader 调用 + 1 次 Worker 多轮 = 约 $0.50
训练期间：$0.00
```

**差距**：ReAct 把 86% 的 LLM 费用花在"确认训练还活着"上。Plan-Execute 把这 86% 变成 $0。

这就是为什么**模式选择直接决定成本结构**。面试时把这个例子讲出来，比说"Plan-Execute 比 ReAct 好"有力 10 倍。

---

## 代码地图

所有代码在 **`core/loop.py`**（650 行）。关键函数：

| 函数（行号） | 职责 | 调 LLM？ |
|-------------|------|----------|
| `run()` (112) | 主循环入口，永不停止的 while | 间接（调其他） |
| `_think()` (228) | Leader 分析上下文 → 输出 JSON 决策 | ✅ 1 次 |
| `_execute()` (248) | 派 Worker 执行 Leader 的决策 | ✅ 多轮 |
| `_monitor_experiment()` (263) | 等训练完成，只做 OS 调用 | ❌ 0 次 |
| `_reflect()` (278) | Leader 分析结果 → 更新记忆 → 决定下一步 | ✅ 1 次 |
| `_enrich_context()` (383) | 注入 ledger/journal/safety 信号 | ❌ 纯 Python |
| `_consume_directive()` (559) | 检查人类是否有新指令 | ❌ 文件操作 |
| `_throttle_if_needed()` (517) | 反烧钱限速 | ❌ 纯计算 |
| `_apply_no_progress_fallback()` (325) | 检测重复无进展 → 强制冷却 | ❌ 纯逻辑 |

---

## 调用链追踪

**从入口开始，一步步追。** 打开 `core/loop.py`，跟着走。

### 第 1 步：`run()` — 主循环骨架（行 112-226）

```python
def run(self):
    while self._running:                              # ① 永不停止
        if self.max_cycles > 0 and self.cycle_count >= self.max_cycles:
            break                                     # ② 达到上限则退出

        self._throttle_if_needed()                    # ③ 反烧钱限速检查
        self.cycle_count += 1

        try:
            self.dispatcher.reset_leader_history()    # ④ 清空 Leader 对话历史
            directive = self._consume_directive()     # ⑤ 读人类指令
            think_result = self._think(directive)     # ⑥ Leader 思考
            execute_result = self._execute(think_result)  # ⑦ Worker 执行
            monitor_result = self._monitor_experiment(execute_result)  # ⑧ 等训练
            reflect_result = self._reflect(execute_result)  # ⑨ Leader 复盘
        except Exception as e:
            self._cooldown_after_error()              # ⑩ 出错不崩，冷却重试
```

**你不需要现在理解每个方法的内部实现。** 先建立心智模型：一个周期 = ⑥→⑦→⑧→⑨。这就是 Agent 的全部。

**关键观察**：
- ④ 每个周期 reset Leader 历史——防止上下文膨胀
- ⑤ 人类指令在每轮开始检查——如果正在 MONITOR 阶段（sleep 15 分钟），指令最多等 15 分钟才生效
- ⑩ 任何异常都被 catch——Agent **永不崩溃**，最多冷却一会儿再试

### 第 2 步：`_think()` — Leader 拿到什么信息？（行 228-246）

```python
def _think(self, directive=None):
    context = {
        "brief":       self.memory.get_brief(),       # PROJECT_BRIEF.md（目标+约束）
        "memory_log":  self.memory.get_log(),          # MEMORY_LOG.md（历史+决策）
        "cycle":       self.cycle_count,               # 当前第几轮
        "directive":   directive,                      # 人类指令（可选）
    }
    self._enrich_context(context)   # ← 这里注入更多信号！
    result = self.dispatcher.dispatch_leader("think", context)
    # dispatch_leader 在 agents.py 里，块 2 会详细追
```

**先不用追 `dispatch_leader` 的内部实现**——那是块 2 的内容。现在只需要知道：Leader 拿到所有上下文后，做一次 LLM 调用，返回一个 JSON 决策。

> ### 📌 停下来：PROJECT_BRIEF.md 和 MEMORY_LOG.md 到底是什么？
> 
> 这两个文件是整个 Agent 的"大脑"，后面每一块都会提到它们。在这里一次性说清楚：
> 
> **PROJECT_BRIEF.md（Tier 1 — 冻结的使命书）**
> - **谁写的**：你在启动 Agent 之前手写
> - **里面有什么**：研究目标（"在 CIFAR-10 上训练 ResNet-18 达到 85% accuracy"）、约束条件（GPU 型号、最大 epoch、batch size）、决策树（"如果 accuracy < 70% 就换优化器，如果 > 85% 就生成报告"）
> - **格式**：自由 Markdown，≤ 3000 字符
> - **谁能改**：**没人能改**。Agent 只能读，不能写——这是硬约束，tool 层面和 prompt 层面双重保护
> - **为什么冻结**：这是 Agent 的"宪法"。如果 Agent 能自己改目标，它会把"accuracy 85%" 改成 "accuracy 50%" 然后宣布胜利
> 
> **MEMORY_LOG.md（Tier 2 — 滚动的记忆日志）**
> - **谁写的**：Agent 自己在 REFLECT 阶段写（`loop.py:296-299`），每轮实验结束后更新
> - **里面有什么**：两个 section——
>   - `## Key Results`：每轮实验的关键指标，如 `[08-10 14:30] ResNet-18 lr=0.01: acc=72.1% (epoch 50)`
>   - `## Recent Decisions`：Leader 做了什么决策以及为什么，如 `[08-10 14:00] 决定将 lr 从 0.1 降到 0.01，因为 loss 在 epoch 20 后震荡`
> - **格式**：Markdown，带时间戳的列表条目，≤ 2000 字符
> - **怎么保持不超限**：FIFO 淘汰——满了就把最旧的条目 `pop(0)` 丢掉。被丢掉的条目没有丢失，完整记录在 `experiments.jsonl` 里
> - **为什么叫"滚动记忆"**：它永远不超过 2000 字符，不管 Agent 跑了 5 轮还是 500 轮。Leader 每次只看到最近的关键信息，上下文不会被历史淹没
> 
> 两个文件加起来永远 ≤ 5000 字符（≈ 1500 tokens），这是"恒定大小记忆"的核心。**详细机制见块 4**。

### 第 3 步：`_enrich_context()` — 注入了什么"情报"？（行 383-447）

这是理解 Agent 智能的关键。Leader 看到的不仅仅是 brief + log，还有：

```
context 注入的内容（全部来自纯 Python 代码，0 次 LLM 调用）：
├── recent_experiments    ← ledger.summary(): 最近 N 个实验的摘要（ledger.py:85-109）
├── progress_signal       ← ledger.detect_stagnation(): "你已连续 3 轮无进展"（ledger.py:137-186）
├── dead_ends             ← journal.dead_ends_tail(): "以下方法已验证无效"（journal.py）
├── insights              ← journal.insights_tail(): "以下洞察值得深入"（journal.py）
├── active_violations     ← safety.scan_violations(): "训练状态卡了 6 小时"（safety.py:19-48）
└── phase_gate            ← ledger.check_phase_gate(): "当前指标是否过门槛"（ledger.py）

> ⚠️ 注意：虽然 _enrich_context() 不注入 cost/budget 信息，但如果你要在自己项目里加成本追踪，
> 可以在这里注入一个 `budget_info` 字段，让 Leader 看到累计花费做成本感知决策。
```

**核心理解**：这些信号**不是 LLM 发现的**——是纯 Python 代码算出来的。把"检测"和"思考"分离：
- Python 做检测（便宜、可靠、可测试）
- LLM 做决策（贵、但有智能）

面试时可以直说："我用了 separation of concerns——检测逻辑是纯 Python 纯函数，LLM 只负责基于检测结果做决策。"

**信号来源速查表**（方便你追踪依赖）：

| 信号 | 来源模块 | 来源函数 | 文件:行号 |
|------|---------|---------|-----------|
| `recent_experiments` | ledger | `summary()` | `ledger.py:85-109` |
| `progress_signal` | ledger | `detect_stagnation()` | `ledger.py:137-186` |
| `phase_gate` | ledger | `check_phase_gate()` | `ledger.py` |
| `dead_ends` | journal | `dead_ends_tail()` | `journal.py` |
| `insights` | journal | `insights_tail()` | `journal.py` |
| `active_violations` | safety | `scan_violations()` | `safety.py:19-48` |

### 第 4 步：`_execute()` — 怎么派工兵？（行 248-261）

```python
def _execute(self, think_result):
    plan = think_result  # {"action": "experiment", "agent": "code", "task": "...", ...}
    
    if plan.get("action") == "experiment":
        agent_type = plan["agent"]        # "idea" | "code" | "writing"
        task = plan["task"]
        result = self.dispatcher.dispatch_worker(agent_type, task, self.tool_registry)
        # dispatch_worker 在 agents.py 里，块 2/3 会详细追
        return result
    elif plan.get("action") == "wait":
        time.sleep(plan.get("wait_seconds", 3600))
    elif plan.get("action") == "report":
        # 生成最终报告
```

Leader 的决策是一个 JSON，有三种 action：
- `experiment`：派 Worker 做实验（最常见）
- `wait`：等着（比如训练还在跑，或者被限速了）
- `report`：目标达成，生成最终报告

### 第 5 步：`_monitor_experiment()` — 零成本的核心（行 263-276）

```python
def _monitor_experiment(self, execute_result):
    if not execute_result.get("experiment_launched"):
        return execute_result  # 没启动训练，跳过
    
    pid = execute_result["pid"]
    log_file = execute_result["log_file"]
    return self.monitor.wait_for_completion(pid, log_file)
```

**`wait_for_completion()` 内部**（`monitor.py:82-107`）——这就是零成本监控的真身：

```python
# monitor.py:82-107
while self._is_process_alive(pid):       # ① 检查进程是否还活着
    time.sleep(self.poll_interval)        # ② 等 15 分钟（默认 900s）
    
    gpu_info = self._safe_gpu_status()    # ③ nvidia-smi 看 GPU
    log_tail = self._safe_tail_file(log_file, lines=5)  # ④ tail 日志最后 5 行
    elapsed = time.time() - start_time
    
    logger.info(                          # ⑤ 打一行日志（不调 LLM）
        f"PID={pid} alive | elapsed={elapsed/3600:.1f}h | "
        f"GPU={gpu_info.get('utilization', 'N/A')}"
    )

# 训练结束后：
final = self._safe_final_status(pid)      # ⑥ 获取最终状态：Slurm 集群用 sacct 查退出码，本地环境返回 unknown
```

**关键**：这个 while 循环里**没有任何 LLM 调用**。`_is_process_alive(pid)` 最终调用 `os.kill(pid, 0)`（块 5 详述），`_safe_gpu_status()` 调用 `nvidia-smi`，`_safe_tail_file()` 调用 `Path.read_text().splitlines()[-N:]`。三个操作都是纯 OS 调用，不花钱。

> 💡 **Slurm 是什么？** Slurm 是高性能计算（HPC）集群上常用的作业调度系统——就像在本地跑 `python train.py`，在 Slurm 集群上跑 `sbatch train.sh` 提交作业。`sacct` 是 Slurm 的记账命令，可以查询已完成作业的退出状态码。本项目的 `_safe_final_status()` 在 Slurm 环境下通过 `sacct` 获取训练的最终退出码（区分正常结束 vs OOM/超时），在普通单机环境下因为没有 sacct，只能返回 `unknown`。

### 第 6 步：`_reflect()` — Leader 怎么做复盘？（行 278-301）

和 `_think()` 类似的模式：收集结果 → 构建 context → 调 Leader → 更新记忆。

区别在于：
- `_think` 输入是"当前状态"，输出是"下一步做什么"
- `_reflect` 输入是"实验结果"，输出是"这个实验说明了什么 + 要不要更新记忆"

---

## 关键代码解析

### 1. 错误退避：`_cooldown_after_error()`（行 553-557）

```python
def _cooldown_after_error(self):
    """Back off after an error to prevent burn loops."""
    backoff = min(self.cooldown * 2, 1800)  # Max 30 min
    logger.warning(f"Error backoff: waiting {backoff}s")
    time.sleep(backoff)
```

**注意**：这个项目用的是**固定加倍退避**，不是经典的指数退避（exponential backoff）。每次出错都等 `cooldown * 2` 秒（默认 cooldown=300s → 等 600s=10min），最多 1800s（30min）。没有 `_error_streak` 计数器，不追踪连续错误次数——简单直接。

**为什么不用固定间隔？** 如果错误是因为外部原因（API 挂了、网络断了），固定间隔会反复撞墙。加倍等待给外部系统恢复的机会。

> 💡 **面试扩展**：如果你想展示更完整的错误处理，可以在自己项目里实现真正的指数退避带 streak 计数：
> ```python
> self._error_streak += 1
> wait = min(self.cooldown * (2 ** (self._error_streak - 1)), 3600)
> # streak=1 → cooldown×1, streak=2 → cooldown×2, streak=3 → cooldown×4, ...
> ```

### 2. 无进展 fallback：`_apply_no_progress_fallback()`（行 325-355）

```python
def _apply_no_progress_fallback(self, think_result, directive):
    if think_result.get("action") != "experiment":
        return think_result  # 不是实验类决策，不管
    
    signature = self._plan_signature(think_result)  # 对决策做"指纹"
    # 如果连续 N 次相同的指纹且都无进展 → 强制改为 wait
    if (self._no_progress_streak >= self.no_progress_fallback_threshold
        and signature == self._last_no_progress_signature):
        # 记录 dead end + 强制冷却
        return {"action": "wait", "reason": "Fallback: repeated no-progress cycles"}
```

**为什么需要这个？** LLM 可能在"尝试 X→失败→反思→再尝试 X"的循环里打转。这个纯规则检测在 Python 层面截断循环，不需要 LLM 自己意识到。

**`_plan_signature()` 怎么算"指纹"**（`loop.py:315-323`）：

```python
def _plan_signature(self, plan: dict) -> str:
    """Build a stable signature for repeated-plan detection."""
    normalized = {
        "action": plan.get("action", ""),
        "agent": plan.get("agent", ""),
        "task": " ".join(plan.get("task", "").split())[:300],   # ① 空白压缩 + 截断
        "hypothesis": " ".join(plan.get("hypothesis", "").split())[:200],
    }
    return json.dumps(normalized, sort_keys=True, ensure_ascii=True)  # ② 稳定 JSON
```

**关键设计**：`" ".join(...split())` 把所有连续空白压缩成单个空格（所以 `"训练   ResNet"` 和 `"训练 ResNet"` 产生相同指纹）。`sort_keys=True` 保证 JSON 字段顺序不影响比较。这两个措施确保只有**语义相同的计划**才会被识别为重复，不会被 LLM 输出格式的微小波动干扰。

### 3. 反烧钱限速：`_throttle_if_needed()`（行 517-538）

> ⚠️ "反烧钱"是项目自己的叫法，业界通用叫法是 **rate limiting** 或 **cost-aware throttling**。

```python
def _throttle_if_needed(self):
    wait = seconds_until_allowed(self._cycle_timestamps, time.time(), self.max_cycles_per_hour)
    if wait > 0:
        time.sleep(wait)
```

如果 Agent 陷入快速失败循环（30 秒一个周期），一小时能跑 120 次 → 烧 $6。限速器在 Python 层面阻止这个。

---

## 设计决策分析

### 决策：`while True` + sleep vs 事件驱动

**为什么选这个**：零外部依赖。不依赖消息队列、事件总线、K8s——能在任何有 Python 的 GPU 机器上跑。

**有什么问题**：
- 训练崩了要等到下一个 poll 才知道（最长等 15 分钟）
- 人类指令有延迟

**业界做法**：Local 模式下用 SIGCHLD 信号处理器——子进程退出时 OS 立即通知。但核心的 while + sleep 模式保持不动，因为简单就是最大的优点。

### 决策：文件轮询 vs API

**为什么选这个**：人类通过 `echo "指令" > HUMAN_DIRECTIVE.md` 就能干预 Agent。不需要 Web 服务器、不需要认证。Unix 哲学——一切皆文件。

**有什么问题**：vim 编辑时可能读到半写文件；人类不知道 Agent 收到没有。

**实际的防护措施**（`loop.py:559-572`）：`_consume_directive()` 使用 **读后归档（read-then-archive）** 模式——读取内容后立即 `directive_path.rename(archive_dir / ...)`。`os.rename()` 在同一个文件系统上是**原子操作**（POSIX 保证），不会出现"读了一半文件被删"的情况。即使 vim 正在编辑，rename 也只会移动 inode，不影响已打开的文件句柄。

### 决策：把所有信号塞进 context 而不是让 LLM 自己查

**为什么选这个**：LLM 调用是唯一花钱的地方。如果让 LLM 在思考时说"我先看看最近的实验结果"（又一个 tool call），那就多花一轮钱。把信号预计算好塞进 context，LLM 一次调用就能看到全部——省钱。

---

## 本块要掌握的代码

学习完本块，以下代码你应该能不靠文档讲清楚：

| 代码 | 位置 | 掌握标准 |
|------|------|---------|
| `run()` 主循环骨架 | `loop.py:112-226` | 能画出 ⑥→⑦→⑧→⑨ 四阶段流程图，说出每个阶段调不调 LLM |
| `_think()` 上下文组装 | `loop.py:228-246` | 知道 context dict 有哪些字段，`_enrich_context` 在哪被调用 |
| `_enrich_context()` 信号注入 | `loop.py:383-447` | 能列出注入的 7 种信号，说出每种来自哪个模块（ledger/journal/safety） |
| `_execute()` 决策分发 | `loop.py:248-261` | 知道 Leader JSON 的三种 action 分别走什么分支 |
| `_monitor_experiment()` | `loop.py:263-276` | 知道这个函数内部调了什么（`monitor.wait_for_completion`），LLM 调用次数 = 0 |
| `_reflect()` 复盘 | `loop.py:278-301` | 能说出和 `_think()` 的输入/输出区别 |
| `_cooldown_after_error()` | `loop.py:553-557` | 能说出等待时间公式（`min(cooldown*2, 1800)`），知道最多等 30 分钟 |
| `_apply_no_progress_fallback()` | `loop.py:325-355` | 能解释"指纹签名"怎么检测重复计划，触发后怎么处理 |

---

## 检验

完成这些再进入块 2：

1. `run()` 的 while 循环有几种退出路径？分别在哪些行？
2. 如果某一轮抛异常（213 行），Agent 会崩吗？怎么恢复？
3. `_consume_directive()` 为什么是"读后归档"而不是"反复读"？
4. `_enrich_context()` 注入的 5 种信号分别来自哪个模块？在哪个文件？
5. `_cooldown_after_error()` 等待多长时间？每次错误的等待时间一样吗？（提示：看 loop.py:555）
6. 如果 Leader 连续 3 次输出同一个实验计划且都没进展，系统会怎么做？
7. 训练期间的 MONITOR 阶段做了什么 LLM 调用？（答案：0 次）
