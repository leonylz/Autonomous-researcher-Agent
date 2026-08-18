# Architecture

> 本分支(LangGraph 重写版)的架构说明。上游原始框架的文档见原仓库。

## 0. 演进路线(为什么是现在这样)

```
v1 (上游)  自定义文本 <tool_call> 协议 + while 循环 + state.json
v2 (本分支) LangGraph StateGraph:监督路由 + SQLite checkpointer + 原生工具循环
```

迁移动机(commit `1431f43` 起):
- 文本协议靠正则解析,模型输出畸形即整轮失败 → 换成 LangChain 原生
  `bind_tools` + `ToolMessage`,参数校验与重试由框架保证;
- while 循环 + 手写 state.json → 换成 StateGraph + SQLite checkpointer,
  崩溃后从最近 checkpoint 续跑(含损坏自愈);
- 迁移旧测试时暴露 3 个安全回归(shell 注入/路径越界/符号链接泄漏),
  逐一修复并固化为测试 —— 详见 `docs/INTERVIEW_STORY.md`。

## 1. 系统概览

```
┌──────────────────────────────────────────────────────────────┐
│                    Deep Researcher Agent (LangGraph)          │
│                                                              │
│  config.yaml → ResearchGraph (core/nodes.py)                 │
│                                                              │
│  START → supervisor(规则路由,零 LLM)                          │
│        → think → execute → monitor → reflect → supervisor    │
│              (LLM)   (worker)  (零 LLM)  (LLM)               │
│                                                              │
│  ┌──────────── 支撑层 ────────────────────────────────────┐  │
│  │ 工具层: @tool 纯函数 + TOOL_FUNCTIONS 按角色注册         │  │
│  │ 安全层: 规则沙箱 + 策略沙箱 + 环境剥离 + 干跑门           │  │
│  │ 记忆层: MemoryManager + Ledger + Journal + Store(4层)   │  │
│  │ 知识层: RAG 论文库(分节/幂等/新鲜度闸)                  │  │
│  │ 执行层: Local / SSH / Slurm 后端 + 环境钉死             │  │
│  │ 观测层: audit / event_journal / cost / dashboard / eval │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## 2. 核心循环(`core/nodes.py` — ResearchGraph)

StateGraph 节点,全部事件经 `EventJournal` 上报 dashboard:

| 节点 | 角色 | LLM? | 关键机制 |
|------|------|------|----------|
| supervisor | 纯规则路由(`_deterministic_next`) | 否 | 不用 LLM 路由:省成本 + 防"模型反复选 think"死循环 |
| think | Leader 决策 | 是 | PromptBuilder 优先级折叠、recall-before-reason、Plan-then-Replan 注入、`with_structured_output`(LeaderDecision schema)、预算封顶、决策树纪律 |
| execute | worker 工具循环 | 是 | `bind_tools` + `ToolMessage` 原生循环、review 门控(先审查再干跑)、强制干跑门、checkpoint 模板硬校验 |
| monitor | 训练监控 | **否** | 进程存活 + GPU + 日志 tail,零 LLM;`track_experiment` 登记保证耗时/状态统计真实 |
| reflect | 结果分析 | 是 | 结构化输出(milestone/decision 字段)、成本记录、失败画像反馈;日志截断 1500 字符省 token |

**反卡死与熔断**:连续无进展 → `wait`(终止);LLM 失败 → retry 分级(瞬时退避/致命中断/上下文溢出自动缩减)→ fallback 模型 → 结构化降级 JSON,绝不抛异常。

## 3. 工具层:协议无关的纯函数工具集

- 工具是 `@tool` 装饰的**纯函数**,与调用协议解耦:
  - 默认路径:agent 内嵌 `bind_tools`(零开销,launch↔monitor 状态共享);
  - 可选路径:`core/mcp_server.py` 把它们包成标准 MCP Server
    (stdio + JSON-RPC 2.0,标准库手写),任何 MCP 客户端可发现/调用,
    **安全策略跨协议生效**(沙箱/路径/命令守卫不被绕过);
- `TOOL_FUNCTIONS` 按角色注册最小工具集(code 9 / idea 5 / writing 4 / review 6),
  加工具 = 加一个 `@tool`,两边自动可见;
- 每个工具带使用规则 docstring(何时用、何时禁止),这是防"工具退化"的第一道防线。

## 4. 安全层(四道防线)

| 层 | 机制 | 说明 |
|----|------|------|
| 规则沙箱 | 无 shell 执行(shlex argv)、路径越界拦截(`_resolve_path`)、符号链接防泄漏、危险命令黑名单(含 zsh/ksh/dash/fish -c、cmd /c、大小写变体)、敏感文件(.env/私钥)读写拦截 | 机制层:什么命令/路径是危险的 |
| 策略沙箱 | `sandbox.mode`:read-only / workspace-write / full;环境剥离(API key 不进子进程) | 策略层:工具在什么权限下可用 |
| 幂等与门控 | launch 幂等(活跃训练拒绝重复启动)、实例锁(O_EXCL 原子)、dry-run 门(系统执行+指纹) | 防重复扣费/双实例并发 |
| 人工审批 | ApprovalGate(opt-in,launch 前可配人工确认) | HITL 兜底 |

**系统干跑门**(`launch_experiment(dry_run=true)`):干跑由系统用**绑定解释器**执行,
系统写权威 `dry_run_log.json`(interpreter + 脚本指纹 + 依赖指纹);
真实训练校验三者一致,任何漂移强制重新干跑 —— **干跑/训练环境不一致在机制上不可能**。

## 5. 环境钉死(`workspace/.python_env.json`)

- 解析顺序:config 指定 → 钉住记录 → 项目 venv(`.trainenv`)→ 自动创建 → 借用(opt-in);
- 解析结果原子写钉住文件,**跨 cycle、跨进程永远同一解释器**;
- 项目隔离:默认不碰 base/系统环境,创建走 CPU 版 torch(~200MB),
  只有真正 launch 才可能触发下载(构造阶段 `auto_create=False`);
- `check_python_deps` 用 `find_spec`(毫秒级,不 import torch)。

## 6. 记忆与知识

- **两 tier 记忆**(MemoryManager):Tier1 brief 冻结 / Tier2 滚动压缩,总 5K 固定;
- **实验账本**(ledger,append-only):每个 cycle 一行,衍生 stagnation/phase-gate 信号;
  也是 RAG 新鲜度闸的事实源(`_tried_arxiv_ids` —— 已实验论文不再注入);
- **Store 四层记忆**(SqliteStore):episodic / semantic / procedural / cross-project,
  recall-before-reason 检索注入;
- **RAG 论文库**(`core/rag.py` + `scripts/ingest_papers.py`):
  arXiv 元数据 + ar5iv 全文(方法/实验段按标题分类)、按段幂等(content hash)、
  注入时 methods 段优先 + 更长额度、注入攻击检测丢弃、新鲜度过滤。

## 7. 观测与成本

- audit(actor/action 审计)、event_journal(append-only + SSE)、
  cost_tracker(逐调用成本 + 每日预算 + 预算封顶)、dashboard(FastAPI,
  与引擎解耦,只读 workspace + HUMAN_DIRECTIVE 指令通道)、
  eval 录制(AgentRecorder,零额外成本);
- **预算封顶**:`cost.daily_budget` 超限 → 循环终止;80% → 上下文警告。

## 8. 评测(eval-first)

- `examples/eval_tasks/`:T1-T5 科研任务(目标指标 + 预算 + golden 规范);
- `core/scripted_llm.py`:确定性 LLM 替身,**无 API key 驱动完整循环**,
  零成本回归全循环健康(事件/账本/锁);
- `scripts/run_eval.py`:dry(校验)/ scripted(零成本回归)/ real(真实评测)/ report(聚合);
- 全量测试 356 个(280 个旧测试迁移 + 76 个新增,全部离线)。

## 9. 执行后端(`core/execution.py`)

- Local / SSH / Slurm,统一 `run_command`/`launch_command` 接口;
- `_scrub_env` 统一子进程环境剥离(API key 防泄露硬约束);
- `pid_alive` 跨平台统一收口(Windows 上裸 os.kill 有误报/崩溃两类坑);
- monitor 轮询可终止性有界(grace + wall-clock backstop)。
