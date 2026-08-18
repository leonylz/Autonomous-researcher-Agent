# 块 5：监控与安全 — 进程级监控 + 运行时安全

> ⚠️ **术语提示**：本项目叫"零成本监控"（Zero-Cost Monitoring），本质是 **OS-level process supervision**。"反烧钱限速"的本质是 **sliding-window rate limiting**。"6层安全防线"是项目自己的分层计数，业界说法是 **defense in depth**。详见 [总纲术语对照表](../LEARNING_INDEX.md)。

## Agent 概念

本块涉及 **3 个 Agent 设计模式**，解决"怎么在烧钱和不崩之间取得平衡"：

### 概念 1：Zero-Cost Monitoring（零成本监控）

```
传统做法（贵）：                         本项目做法（免费）：
─────────────                          ────────────────
每 5 分钟调一次 LLM:                   每 15 分钟做一次 OS 调用:
  "训练还在跑吗？进度如何？"              kill -0 $PID        ← 进程还活着？
  → LLM 回答: "是的，还在跑"              nvidia-smi          ← GPU 还在用？
  → $0.03 × 12次/小时 × 24小时           tail -50 log.txt    ← 最新指标？
  → 一天 $8.64 花在"确认还活着"上         → $0.00 × 4次/小时 × 24小时
                                         → 一天 $0.00 花在监控上
```

**核心思想**："训练是否在跑"是一个布尔问题——不需要智能，只需要 `kill -0`。

### 概念 2：Rate Limiting — Sliding Window（反烧钱限速）

> ⚠️ 本项目的实现是 **Sliding Window** 算法（滑动窗口计数），不是 Token Bucket（令牌桶）。标题用 "Rate Limiting" 作为业界通用分类。

```
普通限速器（固定间隔）：              本项目限速器（滑动窗口）：
───────────────────                 ─────────────────────
每 5 分钟最多 1 次                   过去 1 小时内最多 N 次
if last_call < 5min ago: sleep      if count_in_window >= N: sleep until oldest rolls off
```

**为什么滑动窗口更好？** 固定间隔（等 5 分钟）在正常时浪费了"可以连调两次再等"的灵活性。滑动窗口允许突发（burst），但限制长期平均速率。

### 概念 3：Pure-Function Safety（纯函数安全检测）

```python
# safety.py — 所有函数都是纯函数：输入确定 → 输出确定，无副作用
def scan_violations(state, fail_count, now) -> list[str]:  # 纯函数
def seconds_until_allowed(timestamps, now, max_per_hour) -> float:  # 纯函数
```

**为什么强调"纯函数"？** 因为纯函数可以**单元测试**。你可以构造"连续 3 次失败的 state"输入，验证它确实返回了 violation。不需要真的让 Agent 跑 3 轮来测试。

---

## 代码地图

两个文件，总计 ~274 行：

**`core/monitor.py`**（196 行）— 零成本实验监控：

| 函数（行号） | 职责 | LLM 调用？ |
|-------------|------|-----------|
| `wait_for_completion()` (74) | while 循环等训练完成 | ❌ 0 次 |
| `_is_process_alive()` (137) | `kill -0 $PID` 检查进程 | ❌ 0 次 |
| `_safe_gpu_status()` (141) | `nvidia-smi` 检查 GPU | ❌ 0 次 |
| `_safe_tail_file()` (154) | `tail -50` 读最新日志 | ❌ 0 次 |
| `_extract_metrics()` (160) | 正则从日志中提取指标 | ❌ 0 次 |

> 💡 **两层架构：wrapper（安全包装）vs implementation（实际执行）**：
> ```python
> # monitor.py:141-145 — 安全包装层（永不抛异常）
> def _safe_gpu_status(self) -> dict:
>     try:
>         return self.backend.get_gpu_status()  # → 委托给 execution.py
>     except Exception:
>         return {"utilization": "N/A"}          # 任何异常都静默处理
> 
> # execution.py:715-742 — 实际实现层（OS 调用）
> def get_gpu_status(self) -> dict:
>     result = subprocess.run(["nvidia-smi", "--query-gpu=...", ...], ...)
>     # 解析 CSV 输出...
> ```
> `_safe_` 前缀 = "这个函数永不抛异常"。监控循环（`while True`）不能被 `nvidia-smi` 偶尔报错打断——GPU 驱动故障、命令超时、输出格式变化都不应该让整个 Agent 退出。wrapper 捕获所有异常，返回 fallback 值，让监控循环继续。

**`core/safety.py`**（79 行）— 纯函数安全检查：

| 函数（行号） | 职责 | 是纯函数？ |
|-------------|------|-----------|
| `scan_violations()` (19) | 检测异常状态（重复失败 + 卡死） | ✅ |
| `seconds_until_allowed()` (52) | 滑动窗口限速计算 | ✅ |
| `prune_timestamps()` (76) | 清理过期时间戳 | ✅ |

---

## 调用链追踪

### 第 1 步：训练期间发生了什么？

从块 1 的 `_monitor_experiment()` 进入：

```
loop.py:263  _monitor_experiment(execute_result)
    │
loop.py:269  self.monitor.wait_for_completion(pid, log_file)
    │
monitor.py:74  def wait_for_completion(self, pid, log_file):
monitor.py:82      while self._is_process_alive(pid):      ← ① 进程还活着？
monitor.py:83          time.sleep(self.poll_interval)       ← ② 睡 15 分钟
monitor.py:86          gpu_info = self._safe_gpu_status()   ← ③ nvidia-smi
monitor.py:87          log_tail = self._safe_tail_file(...) ← ④ tail 日志
monitor.py:90          logger.info(...)                     ← ⑤ 记录状态
    │
    │  训练结束（进程退出）
    ▼
monitor.py:100  log_tail = self._safe_tail_file(log_file, lines=50)  ← ⑥ 取更多日志
monitor.py:102  final = self._safe_final_status(pid)       ← ⑦ 检查退出状态
monitor.py:116  metrics = self._extract_metrics(log_tail)  ← ⑧ 正则提取指标
monitor.py:127  return result                              ← ⑨ 返回给 loop.py
```

**整个函数没有一行调 LLM。** 从训练开始到训练结束，LLM 完全不知道发生了什么——直到训练结束，`reflect()` 阶段才拿到结果。

### 拆开看：三个零成本操作的具体实现

"零成本"不是魔法，是三个纯 OS 层面的操作。下面是它们在 `execution.py` 里的实际代码。

#### ① `kill -0`：进程还活着吗？（execution.py:701-706）

```python
def is_process_alive(self, pid: int) -> bool:
    try:
        os.kill(pid, 0)        # 信号 0 = 不发送任何信号，只检查进程是否存在
        return True             # 没抛异常 → 进程存在
    except OSError:
        return False            # 抛 OSError → 进程不存在（已退出/崩溃）
```

**不是 shell 命令，是 Python 的 `os.kill()` 系统调用。** 信号编号 `0` 是 POSIX（Portable Operating System Interface，Linux/macOS 等 Unix-like 系统遵循的可移植接口标准）规范里的特殊值——"null signal"，内核只做权限检查和进程存在性检查，不向目标进程发送任何东西。这是你能对操作系统发出的最轻量级的查询。

```
os.kill(12345, 0)  → 内核: "PID 12345 存在吗？" → 存在 → 返回
os.kill(12345, 0)  → 内核: "PID 12345 存在吗？" → 不存在 → raise OSError
```

- 耗时：~0.001 秒（一次系统调用）
- 费用：$0（不涉及网络）
- 依赖：POSIX 兼容系统（Linux/macOS，Windows 不支持）

> **Windows 注意（2026-08 实测踩坑）**：Windows 上 `os.kill(pid, 0)` 不可靠——
> 已退出但进程对象尚未被内核回收的 pid 会**误报存活**（monitor 轮询死循环），
> 对象已完全回收的 pid 会抛 `SystemError`（CPython bug，`except OSError` 兜不住，
> 启动直接崩溃）。现在统一走 `core/execution.py` 的 `pid_alive()`：Windows 下用
> `GetExitCodeProcess`（退出码 ≠ 259 = 已死），POSIX 保持 `os.kill(pid, 0)`。
> 回归测试：`tests/test_execution.py::LocalBackendAliveCheckTests`。

#### ② `nvidia-smi`：GPU 还在跑吗？（execution.py:715-742）

```python
def get_gpu_status(self) -> dict:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True, timeout=10, check=False,
    )
    # 输出示例: "85, 10240, 24576\n"
    if result.returncode == 0:
        lines = result.stdout.strip().splitlines()
        gpus = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            # parts = ["85", "10240", "24576"]
            gpus.append({
                "utilization": f"{parts[0]}%",      # "85%"
                "memory": f"{parts[1]}MB/{parts[2]}MB",  # "10240MB/24576MB"
            })
        return {"gpus": gpus, "utilization": gpus[0]["utilization"] if gpus else "N/A"}
```

**本质是启动一个子进程跑 `nvidia-smi`，解析它的 CSV 输出。** `--query-gpu` 指定要查什么字段，`--format=csv,noheader,nounits` 让输出变成纯数字不带单位，方便 `split(",")` 解析。

这里的关键设计：`nvidia-smi` 的输出只是写进日志——**没有喂给 LLM**。Monitor 阶段只记录"GPU 利用率 85%"到 logger，不向 LLM 报告。LLM 只在训练结束后的 reflect 阶段才看到最终结果。

#### ③ "tail"：最新日志是什么？（execution.py:708-713）

```python
def tail_file(self, path: str, lines: int = 50) -> list[str]:
    rel_path = normalize_relative_path(path)
    file_path = _resolve_under_root(self.workspace, rel_path)
    if not file_path.exists():
        return []
    return file_path.read_text().splitlines()[-lines:]
    #             ① 读整个文件      ② 按行分割      ③ 取最后 N 行
```

**不是 shell 的 `tail` 命令，是 Python 文件读取 + 列表切片。** `read_text().splitlines()[-lines:]` ——读进内存，按换行切，取最后 N 行。

**为什么可以直接读整个文件？** 训练日志通常不大（几 MB），一次读到内存完全可以。如果日志特别大（几百 MB），这里会慢，但训练日志一般不会那么大。

- 耗时：< 0.01 秒（几 MB 的文本文件）
- 费用：$0
- 和 shell `tail` 的区别：shell `tail` 不需要读整个文件（它 seek 到文件末尾），但 Python 标准库没有 seek-to-last-N-lines 的方法，这个小文件场景下直接读更简单

#### 汇总：整个 Monitor 阶段的执行消耗

```
while 进程还活着:                    # 每 15 分钟一次
    os.kill(pid, 0)                # ~0.001s, $0
    subprocess.run("nvidia-smi")   # ~0.5s, $0
    file.read_text().splitlines()  # ~0.01s, $0
    time.sleep(900)                # 15 分钟

训练 2 小时 = 8 次检查 = 8 × (0.001 + 0.5 + 0.01)s ≈ 4 秒的 CPU 时间
总费用: $0.00（零网络调用，零 LLM 调用）
```

**这就是"零成本"的物理基础**：`os.kill` 是内核调用，`nvidia-smi` 是本地命令，`splitlines()` 是内存操作。三个操作都不经过网络，都不需要 API key，都不会出现在任何账单上。

### 第 2 步：`_extract_metrics()` — 正则提取指标（行 160-185）

```python
def _extract_metrics(self, log_lines):
    for line in reversed(log_lines):           # 从最新行往前找
        for pattern, key in [
            (r"loss[:\s]+([0-9.]+)", "loss"),
            (r"acc(?:uracy)?[:\s]+([0-9.]+)", "accuracy"),
            (r"FGD[:\s]+([0-9.]+)", "FGD"),
            (r"FID[:\s]+([0-9.]+)", "FID"),
        ]:
            if key not in metrics:             # 每个指标只取第一个（最新的）
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    metrics[key] = match.group(1)
```

**这是 CV 工程师最容易魔改的地方**——把正则 pattern 换成你训练框架的日志格式：
```python
# 如果你的日志是 "mAP: 0.856 @ epoch 50"
(r"mAP[:\s]+([0-9.]+)", "mAP"),
# 如果你的日志是 "IoU: 0.723"
(r"IoU[:\s]+([0-9.]+)", "IoU"),
```

### 第 3 步：`scan_violations()` — 异常检测（safety.py:19-49）

```python
def scan_violations(state, fail_count, now):
    violations = []

    # 检测 1：重复无进展
    if fail_count >= fail_threshold:  # 默认 3 次
        violations.append("连续 N 次同一计划无进展，换方向！")

    # 检测 2：状态卡死
    if state["status"] == "running" and age_hours > stale_state_hours:  # 默认 6 小时
        violations.append("训练已跑 6 小时无更新——可能卡死了")

    return violations
```

**两个检测都是纯规则**，不需要 LLM 判断。返回的 violation 字符串被注入 `_enrich_context()`，然后 Leader 自己决定怎么应对。

### 第 4 步：`seconds_until_allowed()` — 滑动窗口限速（safety.py:52-73）

```python
def seconds_until_allowed(timestamps, now, max_per_hour, window=3600):
    recent = [t for t in timestamps if (now - t) < window]   # 过去 1 小时内的周期
    if len(recent) < max_per_hour:                            # 还没超
        return 0.0
    recent_sorted = sorted(recent)
    target = recent_sorted[len(recent) - max_per_hour]        # 第 N 旧的
    return max(0.0, window - (now - target))                  # 等它滚出窗口
```

**举例（用真实时钟时间）**：`max_per_hour=6`，当前时间是 `14:00:00`。

过去 1 小时内完成的 6 个周期（都在窗口内）：

```
周期 1: 13:05 启动  周期 2: 13:12 启动  周期 3: 13:20 启动
周期 4: 13:28 启动  周期 5: 13:40 启动  周期 6: 13:55 启动
```

现在 14:00，Agent 想跑第 7 个周期。`seconds_until_allowed()` 的计算：

```
recent = [13:55, 13:40, 13:28, 13:20, 13:12, 13:05]  ← 6 个，全在过去 1 小时内
#                                                 ↑
# 条件：len(recent)=6，不小于 max_per_hour=6 → 不返回 0，继续计算

recent_sorted = [13:05, 13:12, 13:20, 13:28, 13:40, 13:55]
#                 ↑
# target = recent_sorted[6-6] = recent_sorted[0] = 13:05

return 3600 - (14:00 - 13:05) = 3600 - 3300 = 300 秒 = 5 分钟
```

**结果**：需要等 5 分钟。到 14:05 时，13:05 的周期滚出窗口（超过 1 小时前），窗口内只剩 5 个周期，可以跑第 7 个。

**和固定间隔的对比**：

| 场景 | 固定间隔（每 10 分钟） | 滑动窗口（本项目） |
|------|---------------------|-------------------|
| 正常节奏（每 8-12 分钟一个周期） | ❌ 必须等满 10 分钟 | ✅ 不用等 |
| 快速失败（30 秒一个周期，6 个全失败） | ❌ 6 个 x 10 分钟 = 还要 1 小时才能跑完 | ✅ 1 小时内跑 6 个后被限速，等最旧的滚出 |
| 训练中（2 小时无新周期） | ✅ 训练结束后可以立即跑下一个 | ✅ 训练结束后窗口已空，可以立即跑 |

---

## 每个概念：为什么选这个？有没有更好的？

### 概念 1：Zero-Cost Monitoring

**为什么不用 LLM 监控？**

| 维度 | LLM 监控 | OS 调用监控（本项目） |
|------|---------|-------------------|
| 成本 | $0.03-0.05/次 | $0.00/次 |
| 延迟 | API 往返 ~1-3 秒 | OS 调用 ~0.001 秒 |
| 可靠性 | API 可能限流/超时/返回废话 | `kill -0` 只有 True/False |
| 信息量 | 可以做趋势分析 | 只有原始指标 |

**那什么时候需要 LLM 监控？** 当"判断是否需要干预"不是一个布尔问题时——比如"训练 loss 在震荡，但趋势是下降的，要不要 early stop？"。这种需要判断力的任务，才值得花 LLM 的钱。

**业界主流做法**：

| 方案 | 成本 | 适用场景 |
|------|------|---------|
| **纯 OS 调用（本项目）** | $0 | 训练/批处理等长周期任务 |
| **日志正则 + 阈值告警** | $0 | 已知指标的异常检测 |
| **LLM 定期检查** | $0.03-0.05/次 | 需要理解复杂日志的任务 |
| **metrics server (Prometheus + Grafana)** | 基础设施成本 | 生产环境、多任务并行 |

> **Prometheus** 是开源时序指标采集系统——定时从 Agent 拉取指标数据（如每轮耗时、GPU 利用率、LLM 成本）并存入时间序列数据库。**Grafana** 是仪表盘可视化工具——连接 Prometheus 将数据渲染成图表和告警面板（如"GPU 利用率连续 30 分钟低于 10% → 发 Slack 告警"）。这个组合是生产环境监控的标准方案，适合同时跑多个 Agent 的团队场景，但需要额外部署和维护这两个服务。本项目的纯 Python 轮询方案不需要任何外部服务。
| **ML-based anomaly detection** | 训练成本 | 超大规模（数千个任务同时跑） |

**改进方向**：混合策略——OS 调用做常规心跳检测 + LLM 只在检测到异常时才介入分析。这样既保持低成本，又有智能判断能力。

### 概念 2：Token Bucket / Sliding Window 限速

**为什么选滑动窗口而不是固定间隔？**

| 方案 | 允许突发？ | 实现复杂度 | 本项目 |
|------|----------|----------|--------|
| **固定间隔**（每 N 分钟最多 1 次） | ❌ | 最简单 | ❌ |
| **滑动窗口**（过去 M 分钟内最多 N 次） | ✅ | 简单 | ✅ **本项目** |
| **Token Bucket**（有 N 个 token，每个周期消耗 1 个，每秒恢复 rate 个） | ✅ | 中等 | ❌ |
| **Leaky Bucket**（请求先进队列，按固定速率流出） | ❌（削峰） | 中等 | ❌ |

**滑动窗口 vs Token Bucket 的区别**：
- 滑动窗口：看过去 M 分钟有多少次
- Token Bucket：有一个 token 池，每个请求消耗一个 token，token 按固定速率恢复

Token Bucket 允许更大的突发（可以攒 N 个 token 然后一口气用完），但对这个项目来说过度设计了。

**业界主流做法**：

| 方案 | 何时用 |
|------|--------|
| **滑动窗口** | API rate limiting 场景（GitHub API、Cloudflare） |
| **Token Bucket** | 需要允许突发流量的场景（CDN、消息队列） |
| **固定间隔** | 极简场景 |

### 概念 3：Pure-Function Safety（可测试的安全检测）

**为什么强调纯函数？**

```python
# 纯函数 —— 可以写单元测试：
def test_scan_violations():
    state = {"status": "running", "updated_at": 100}
    now = 100 + 7 * 3600  # 7 hours later
    violations = scan_violations(state, fail_count=3, now=now)
    assert len(violations) == 2  # 一条 fail，一条 stale
    assert "stuck" in violations[1]  # 第二条是卡死警告

# 非纯函数 —— 没法测试（依赖于全局状态）：
def check_training_health():          # 不好的写法
    pid = read_pid_from_file()        # 读文件 → 有副作用
    is_alive = os.kill(pid, 0)        # 系统调用 → 有副作用
    if not is_alive: ...
```

**业界做法**：这是函数式编程的核心思想——把副作用推到系统边界（IO 操作），核心逻辑保持纯函数。写测试不需要 mock 文件系统。

---

## 关键代码解析

### 1. 为什么 `wait_for_completion` 内部是 while 而不是事件驱动？

```python
# monitor.py:82-94
while self._is_process_alive(pid):
    time.sleep(self.poll_interval)  # 默认 900 秒 = 15 分钟
```

**原因**：简单。不需要信号处理器、不需要 inotify、不需要任何 OS 特定的机制。一个 while + sleep 在任何平台都能跑。

**代价**：训练崩了要等到下一个 poll 才知道。对于几小时的训练来说，15 分钟的延迟可以接受。

### 2. `_extract_metrics` 用 `reversed()` 从后往前找

```python
for line in reversed(log_lines):  # 从最后一行往前
    ...
    if key not in metrics:        # 每个指标只取最新的
        match = re.search(pattern, line)
```

**为什么从后往前？** 训练日志通常最后几行是最新的指标。从后往前找可以更快命中，而且取到的是最新值。

### 3. `seconds_until_allowed` 的边界处理

```python
# safety.py:64
if not max_per_hour or max_per_hour <= 0:
    return 0.0  # 限速关闭
```

**如果 `max_per_hour=0`（配置文件中关闭限速），函数直接返回 0.0（无需等待）。** 这意味着限速是可选的——用于成本敏感的场景，但在"钱不是问题"的研发环境可以关闭。

---

## 设计决策分析

### 决策：poll_interval 默认 900 秒（15 分钟）

**为什么是 15 分钟？**

- 太短（如 1 分钟）：对几小时的训练来说没意义，只是多了很多无用的日志行
- 太长（如 1 小时）：训练崩了要等 1 小时才知道
- 15 分钟：4 次/小时，训练 6 小时期间检查 24 次——足够及时，又不会太频繁

**改进方向**：可以根据训练时长动态调整——开始 5 分钟，1 小时后 15 分钟，3 小时后 30 分钟。训练越久，突然崩溃的概率越低。

### 决策：`scan_violations` 只返回建议，不管决策

```python
# safety.py 只返回 violations 列表
violations = ["连续 3 次无进展", "训练卡了 6 小时"]
# loop.py 把 violations 注入 Leader context
# Leader 自己决定：换方向？kill 训练？继续等？
```

**这是关注点分离**：Python 做"检测"（便宜的、确定的），LLM 做"决策"（贵的、需要判断力的）。如果把决策也写死在 Python 里（"连续 3 次失败 → 自动换方向"），那就失去了 LLM 的灵活性——也许第 3 次失败只是因为 batch size 太大导致 OOM，减小就行，不需要换方向。

---

## 本块要掌握的代码

| 代码 | 位置 | 掌握标准 |
|------|------|---------|
| `wait_for_completion()` | `monitor.py:74-127` | 能说出 while 循环里 4 个操作（is_alive→sleep→gpu_status→tail），每个的 LLM 调用次数 = 0 |
| `_is_process_alive()` | `monitor.py:137-139` | 知道底层是 `kill -0 $PID`，不发送信号只检查存在，~0.001 秒 |
| `_safe_gpu_status()` / `_safe_tail_file()` | `monitor.py:141-158` | 知道三个操作都有 `_safe` 前缀——任何异常都不崩，静默返回空 |
| `_extract_metrics()` | `monitor.py:160-185` | 知道为什么用 `reversed()` 从后往前找，每个指标只取第一个匹配，怎么加自己的指标 pattern |
| `scan_violations()` | `safety.py:19-49` | 能写出两种检测（重复无进展 + 状态卡死）的触发条件，为什么是纯函数 |
| `seconds_until_allowed()` | `safety.py:52-73` | 能写出滑动窗口算法：取窗口内时间戳→排序→找第 N 旧的→算等待时间 |
| `prune_timestamps()` | `safety.py:76-78` | 知道这行一行代码的作用（清理 1 小时前的时间戳，防止列表无限增长） |

---

## 检验

1. `wait_for_completion()` 里有几次 LLM 调用？（答案：0 次）
2. 如果训练在两次 poll 之间崩溃了，Agent 最长多久后才知道？
3. `_extract_metrics` 为什么用 `reversed()` 遍历日志行？
4. 如果你想加一个新的指标提取（比如你的 CV 训练日志里有 "mAP: 0.856"），改哪几行代码？
5. `seconds_until_allowed` 返回的是什么？假设 max_per_hour=6，当前窗口内有 5 个周期，返回什么？
6. `scan_violations` 能检测哪两种异常？具体在哪几行？
7. 如果 `max_per_hour=0`，限速器还工作吗？在哪一行处理的？
8. 为什么安全检测是纯函数？这有什么实际好处？

---

> 下一步：**块 6 — `agents/README.md`**，理解 Agent 的"源代码"——提示词怎么设计。
