# User Guide — 用户使用指南

> 从"一句话目标"到"自主实验循环"的完整流程。所有标注「计划中」的功能
> 以本仓库当前实现为准(见 docs/EVAL_REPORT.md 与 git log)。

## 最小输入路径(推荐)

```bash
# 1. 配置 API key(唯一必须用户做的事)
export DEEPSEEK_API_KEY="sk-..."          # 或 dashboard 设置面板(页面 ⚙️ LLM 设置)

# 2. 一句话启动(自动生成 brief、自动建环境、自动跑)
python -m core.nodes --project ~/my-proj --goal "把 MNIST 训练到 99% 测试准确率"
```

agent 会自动:
1. 生成 `PROJECT_BRIEF.md`(Goal/Codebase/What to Try 决策树/Constraints/Current Status)
2. 钉死训练解释器(无则自动 uv/conda/venv 创建 + 装 torch;有 GPU 装 CUDA 版,
   进度可见于 workspace/.trainenv_install.log)
3. 进入自主循环:think(自动想 idea,先调研再动手)→ execute(自动写代码+干跑+launch)
   → monitor(零成本盯训练,发散自动提前终止)→ reflect(自动分析+假设结算)→ 循环
4. 达标自动停止(预算封顶/反卡死保护)

## 完整 7 阶段流程

### 1. 安装与配置
```bash
git clone <your-repo> && cd auto-deep-researcher
pip install -r requirements.txt          # 或 pip install -e .[agent,dashboard]
# key:export DEEPSEEK_API_KEY(或 dashboard 设置面板)
# 可选 config.yaml:provider/model/预算(agent.max_cycles,cost.daily_budget)/
# 审批模式(approval.mode)/沙箱(sandbox.mode)
```

### 2. 创建科研任务(三选一)
- **最小**:`--goal "一句话目标"`(自动生成 brief)
- **标准**:手写 PROJECT_BRIEF.md —— **What to Try 决策树是质量关键**:
  给了决策树 → agent 走"决策树纪律";没给 → 强制 idea 先调研(不凭空想)
- **RAG 任务**(如论文复现):`python scripts/ingest_papers.py --project <dir>
  --arxiv 1710.09412,1608.03983 --fulltext`(论文库自动入 workspace)

### 3. 启动
- CLI:`python -m core.nodes --project <dir> --max-cycles 5`
- GUI:`python -m core.dashboard --project <dir> --port 8000` → Start Agent
- 缺 key/brief 时给出明确提示,不会堆栈崩溃

### 4. 自主运行(零干预)
- 每 cycle:决策 → worker(idea 文献调研 / code 写码+干跑门+launch)→
  monitor(零 LLM)→ reflect(分析+假设结算+记忆)
- 内置保护:预算封顶 / 反卡死 / launch 幂等 / 工具循环熔断 / 发散提前终止 /
  checkpoint 自愈(损坏先备份) / 崩溃恢复上下文(自动续训建议)
- HITL(可选):approval.mode=exception 时,仅高危/重大方向才请求人工,
  等待零 LLM 成本,批一次后续复用

### 5. 监控与干预(不打断)
- dashboard:状态(PID/cycle/成本)/ 轨迹(事件流)/ 指令框(HUMAN_DIRECTIVE,
  下个 cycle 最高优先)/ 审批面板 / 工作区文件树 / 设置面板 / checkpoint·快照·下载
- 干预:发指令 / Stop / Rollback(文件快照回退)

### 6. 查看结果
- 账本 `workspace/experiments.jsonl`(每轮假设/指标/状态,失败带 ❌)
- 假设库 `workspace/hypotheses.db`(待验证/已否证,决策只能选未验证的)
- 交付:下载交付包(权重+报告)

### 7. 评测(可选)
```bash
python scripts/run_eval.py --dry          # 校验任务
python scripts/run_eval.py --scripted     # 零成本回归(无 key 可跑)
python scripts/run_eval.py --real --tasks T1   # 真实评测(需 key)
python scripts/run_eval.py --report       # 聚合报告
```

## 环境策略(自动,无需操心)

```
config.execution.python(用户提供)→ 用之,绝不动
.python_env.json 钉住记录 → 每 cycle 同一解释器
uv/conda/venv 自动创建(无可用时)→ 装 torch(Linux 默认 CUDA/Windows CPU)
借用现成(兜底,警告)
```
首次 launch 可能触发环境创建(几分钟,进度见 dashboard/install log)。

## 常见问题

| 现象 | 处理 |
|------|------|
| 启动报"缺少 PROJECT_BRIEF.md" | 加 `--goal` 或手写 brief |
| 启动报 key 未配置 | export key 或用 dashboard 设置面板 |
| agent 反复试同一个实验 | G4 计划评审会打回;或发指令换方向 |
| 训练发散不收敛 | G2 会提前终止(省 GPU);账本标 failed |
| 想人工接管 | approval.mode=exception + dashboard 审批 |
