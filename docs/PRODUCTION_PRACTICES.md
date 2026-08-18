# 大厂 Agent 生产实践参考（2026）

> 本文档记录 Anthropic Claude Code、LangGraph、OpenAI Agents SDK 三大框架
> 的最新生产部署最佳实践，作为本项目优化的设计依据。
>
> 采集时间：2026-08-12

---

## 一、Anthropic Claude Code CLI — 重试 & 流式 & 上下文

**来源**：Claude Code CLI 源码（2026年3月版）、Claude Agent SDK 官方文档

### 1.1 分级重试架构（QueryEngine.ts）

```
指数退避 + 随机抖动（避免惊群效应）

baseDelay = min(BASE_DELAY * 2^attempt, maxDelay)
jitter   = random() * 0.25 * baseDelay
result   = baseDelay + jitter
```

| 错误码 | 行为 |
|--------|------|
| **529 Overloaded** | 前台重试；后台立即放弃（避免放大容量雪崩）。连续 3 次 529 → triggered `FallbackTriggeredError` → 切换到 fallback 模型 |
| **429 Rate Limit** | 指数退避，最长 60s |
| **OAuth 401** | 强制 token 刷新 → 重试，绝不盲重试 |
| **400 Context Overflow** | 解析返回的 token 数 → 计算新 `maxTokensOverride` → 重试 |
| **ECONNRESET / EPIPE** | 检测到 stale keep-alive → `disableKeepAlive()` → 重试 |
| **Persistent mode** | `UNATTENDED_RETRY` 无限重试，30 分钟退避上限，每 30s 发心跳防 timeout |

### 1.2 流式事件架构

SSE 事件序列：
```
message_start → content_block_start → content_block_delta* →
content_block_stop → message_delta → message_stop
```

事件类型体系（生产标准）：
- `text` — 增量文本
- `tool_use` / `tool_progress` / `tool_result` — 工具调用全生命周期
- `thinking` — 扩展思考内容
- `usage` — Token 计数
- `message_completed` / `run_started` / `run_completed` / `run_failed`
- `context_compacted` / `subagent_started` / `subagent_completed`

每个事件需携带：`event_id`, `event_type`, `run_id`, `session_id`, `sequence`, `timestamp`, `payload`, `schema_version`

### 1.3 上下文压缩流水线

按优先级依次执行：
1. **工具结果预算** — 截断过长的 Tool Result
2. **历史裁剪** — 删除最旧的轮次
3. **微压缩** — 合并相邻的短消息
4. **上下文坍缩** — 将早期消息替换为摘要
5. **自动压缩** — Fork 子 Agent 做完整上下文摘要

Prompt Caching 可减少 ~90% Token 成本、延迟减半。

---

## 二、LangGraph — 持久化 & Checkpoint & Store

**来源**：LangChain 官方文档、LangGraph Deployment Guide、Diagrid 生产分析

### 2.1 两层持久化

| 系统 | 范围 | 用途 | 后端 |
|------|------|------|------|
| **Checkpointer** | 单线程 | 短时状态快照、对话连续性、HITL、容错 | `PostgresSaver`, `AerospikeSaver`, `DynamoDBSaver` |
| **Store** | 跨线程 | 长期记忆、用户画像、实体抽取、跨会话知识 | `PostgresStore`, `AerospikeStore`, `DynamoDBStore` |

### 2.2 生产铁律

1. **绝不使用 `InMemorySaver` / `InMemoryStore` 上生产** — 进程重启全部丢失、不能水平扩展、内存无限增长
2. **PostgreSQL 是默认标准** — JSONB 原生支持、并发安全、SQL 可查询/分析
3. **始终提供 `thread_id`** — 不传则静默丢失持久化
4. **始终调用 `setup()`** — 建表/索引必须在启动时完成
5. **使用连接池** — `psycopg_pool.ConnectionPool(min_size=1, max_size=10)`

### 2.3 Checkpoints ≠ Durable Execution（关键缺口）

LangGraph 开源版不提供：
- **自动故障检测** — 需外部 health check / 心跳
- **自动恢复** — 需手动调 `invoke(None, config)` 恢复
- **重复执行防护** — 两进程同 thread_id 可能同时执行
- **死信队列** — 失败工作流无通知机制

需要自建：故障检测 + 自动恢复 + 分布式锁 + DLQ + Worker Pool（Celery/BullMQ）

### 2.4 运维清单

- ✅ `PostgresSaver` / `AsyncPostgresSaver` + 连接池
- ✅ `autocommit=True`（psycopg 需要）
- ✅ Checkpoint 清理策略（保留最近 N 个 / 按时间剪枝）
- ✅ 健康检查端点验证数据库连接
- ✅ Store 存跨线程长期记忆（非 Checkpointer）
- ✅ 命名空间隔离并行子图

---

## 三、OpenAI Agents SDK — 流式 & 上下文 & 会话

**来源**：OpenAI Agents SDK 官方文档（2026）、社区生产部署指南

### 3.1 三层流式事件

| 事件类型 | 携带内容 | 用途 |
|----------|---------|------|
| `raw_model_stream_event` | Token 级 delta（文本、函数调用参数、推理内容） | 实时展示生成文本 |
| `run_item_stream_event` | 完成的工具调用、工具结果、消息 | Agent 编排、日志、工作流触发 |
| `agent_updated_stream_event` | 多 Agent 切换通知 | UI 上下文更新 |

**关键**：流式 run 在异步迭代器完全耗尽前不会 complete。总是在循环退出后检查 `result.is_complete`。

### 3.2 四种上下文管理模式

| 模式 | 特点 | 适用场景 |
|------|------|---------|
| **Sessions（推荐）** | 持久化，Survive 重启，支持可恢复审批 | 生产默认 |
| **Local History** | 完全掌控对话列表，自行截断/格式化 | 需要极致控制 |
| **Conversation ID** | 服务端管理，多系统共享线程 | 多服务架构 |
| **previousResponseId** | 轻量级，服务端简单续接 | 简单场景 |

### 3.3 核心生产模式

1. **分离运行时上下文与模型可见历史** — 用户ID、DB句柄、Logger 放 `RunContext`，不放 Prompt
2. **缓冲工具调用流** — 非 OpenAI 供应商（DeepSeek/LiteLLM/Bedrock）delta 不可靠时，开启 `buffer_streamed_tool_calls=True`
3. **优雅取消** — `result.cancel(mode="after_turn")` 让当前轮次干净完成
4. **中断 & HITL** — 流遇到需审批的工具时，Stream 结束，审批在 `result.interruptions`，转 `RunState` 后恢复
5. **Token 使用追踪** — 每次 response cycle 独立可追踪

---

## 四、跨框架共识（本项目的对齐依据）

### 4.1 工具调用
- ✅ 结构化 schema（Pydantic/JSON Schema），不用正则解析自由文本
- ✅ 工具错误以**工具返回值**形式返回，不抛异常
- ✅ HITL 审批门控：文件写入、API 调用、DB 修改等破坏性操作
- ✅ 凭证隔离：Secrets Manager，不注入 System Prompt
- ✅ 并行工具调用：fan-out/fan-in 减少延迟

### 4.2 重试
- ✅ 错误分类 → 差异化退避策略
- ✅ 指数退避 + 随机抖动
- ✅ 瞬时重试 / 致命立即中断
- ✅ 上下文溢出 → 缩减输入重试

### 4.3 流式
- ✅ 增量展示 + 积累拼接
- ✅ 显式 `max_tokens` 防失控
- ✅ 请求超时（短任务 30s，长生成 120s）
- ✅ 挂起请求 Kill + 优雅错误

### 4.4 上下文管理
- ✅ 压缩流水线（不是简单滑动窗口）
- ✅ Prompt Caching 降低 90% 成本
- ✅ 分离运行时上下文和模型可见历史
- ✅ 恒定预算，6 个月不膨胀

### 4.5 可观测性
- ✅ 标准化事件协议（event_id, run_id, session_id, sequence, timestamp, payload）
- ✅ 每个 response cycle 独立追踪
- ✅ Token 使用 → Cost Attribution
- ✅ Agent 决策链路可审计

---

## 五、本项目对齐后的优化方案

基于以上三大框架的实践，本项目的 10 项优化调整如下：

| # | 优化项 | 对齐来源 | 调整 |
|---|--------|---------|------|
| 1 | 滑动窗口 | Anthropic 多级压缩流水线 | 改为多级压缩：工具结果截断 → 早期摘要 → 完整压缩 |
| 2 | Store 持久化 | LangGraph 生产铁律 | SQLite 单机版（对标 PostgresSaver） |
| 3 | LLM 重试 | Anthropic 分级退避 | 错误分类 + 指数退避 + 抖动 |
| 4 | 流式输出 | Anthropic 事件架构 | 三层事件：text / tool_call / usage |
| 5 | Monitor 异步 | OpenAI Sessions | async/await 非阻塞轮询 |
| 6 | Dashboard 推送 | Anthropic SSE 事件协议 | 标准化事件 schema |
| 7 | 删除旧协议 | 跨框架共识 #1 | deprecated + 工具返回值形式返回错误 |
| 8 | FAISS 检索 | LangGraph Store | IVF 索引 + fallback |
| 9 | 结构化路由 | OpenAI 结构化输出 | `with_structured_output()` |
| 10 | Eval 框架 | Anthropic 录制/回放 | recorder/replayer + golden dataset |
