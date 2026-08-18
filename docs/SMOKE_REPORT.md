# SMOKE_REPORT — 真实闭环冒烟验证(Phase 0.1)

> 时间:2026-08-18 · 模型:deepseek-chat(provider=deepseek) · 项目:`examples/smoke_project`
> 任务:合成 4 高斯簇分类,MLP 训练到 test_acc ≥ 0.90(禁止下载数据,torch 合成,固定 seed)
> 验证方式:**真实 LLM 端到端运行**(非 ScriptedLLM/非 dry mock)—— 用户此前明确指出
> "没有完整运行过,怎么就开始评估了",本报告即为补上的真实运行证明。

## 0. 一句话结论

**真实 LLM 全自动闭环跑通:输入 PROJECT_BRIEF.md → think 提出假设 → code agent 编写/修改
训练脚本 → 系统干跑门校验 → 真实训练启动 → monitor 零 LLM 轮询提取指标 → reflect 结算假设
与记忆 → writing agent 产出最终报告 → 正常 finish(exit 0)。**

两次真实运行共暴露 **7 个真实路径 bug/缺口**,全部修复并有回归测试;修复后的第二次运行
(Run B) 2.28 分钟、50 次 LLM 调用、$0.10、单次真实训练 test_acc=0.998 全绿收官。

| 运行 | 结果 | 退出码 | 耗时 | 成本(实测) |
|------|------|--------|------|------------|
| Run A(首次真实运行) | 闭环走通但暴露 7 处真实缺陷 | 1(打印 crash) | ~4 min | 仅记账 leader 部分($0.0046,worker 未记账) |
| Run B(修复后) | 全绿:实验→报告→finish | **0** | **2.28 min** | **$0.10169**(含 worker 记账) |

---

## 1. 验证了什么(闭环链路逐段)

```
PROJECT_BRIEF.md(人工/自动生成)
  → think_node   (deepseek-chat,结构化输出 400 → 自动降级文本模式,200)
  → supervisor   (规则路由,零 LLM)
  → execute_node (code agent ReAct 工具循环:read_file/write_file/launch_experiment)
      ├─ launch_experiment(dry_run=true) → 系统用绑定解释器执行 --dry-run
      │    写权威 dry_run_log.json(interpreter + script_hash + torch 指纹)
      └─ launch_experiment(真实) → 校验 解释器/脚本指纹/依赖指纹 一致 → 启动子进程
  → monitor_node (零 LLM:每 15s 轮询 log,解析 METRIC_JSON 契约行 + 正则兜底)
  → reflect_node (deepseek-chat:里程碑/决策 → 假设结算 + 记忆写入)
  → think_node   (下一轮决策:继续 / 报告 / 停止)
  → finish       (最终回答打印)
```

**真实产物(Run B,全部落盘可审计):**

| 产物 | 内容 |
|------|------|
| `workspace/logs/baseline_mlp.log` | 10 个 epoch 的 `METRIC_JSON` 契约行:loss 0.85→0.0188,test_acc 0.9935→**0.998** |
| `workspace/checkpoints/best_model.pth` + `checkpoint_epoch_*.pth` | 真实训练权重(6 文件,1 MB) |
| `workspace/experiments.jsonl` | cycle 1:`status=completed`,metrics {epoch,loss,test_acc,accuracy},pid 30756 |
| `workspace/hypotheses.db` | 假设 `a937a9f6` → **confirmed**,experiment_id=30756,证据=reflect decision |
| `workspace/costs.jsonl` | 50 次调用逐笔记账(含 worker 工具循环),总额 $0.10169 |
| `workspace/FINAL_REPORT.md` | writing agent 产出的完整实验报告(配置表/结果/分析/Next Steps) |
| `workspace/MEMORY_LOG.md` | 里程碑 + 决策逐轮记录 |
| `workspace/.snapshots/` | 每 cycle 快照(代码+权重),支持回滚 |

**关键机制验证:**

- ✅ 结构化输出 400 → 文本降级:deepseek 不支持 `response_format`,系统自动降级,全程无一次失败
- ✅ 干跑门:dry_run_log.json 由系统写(interpreter/script_hash/fingerprint 三者一致才放行真实启动)
- ✅ 指标契约:`METRIC_JSON` 契约行优先解析,test_acc 归一化(模板打 test_acc,正则兜底补 accuracy)
- ✅ 记忆即账本:假设/实验/成本全部落盘,无"聊天式记忆"
- ✅ 失败不伪装:Run A cycle 1 未产出实验 → ledger 记 `status=experiment, metrics={}` + 结论"无实验结果",而非假 completed

---

## 2. Run A — 首次真实运行:闭环能走通,但暴露 7 处真实缺陷

**运行轨迹(smoke_run.log):**

| 时间 | 阶段 | 发生了什么 |
|------|------|-----------|
| 14:15:12 | 启动 | 环境绑定 D:/Anaconda/python.exe(纯函数解析,0.16s) |
| 14:15:15 | think | 400 → 文本降级 → 200,提出假设 |
| 14:15:18 | execute | 快照 cycle0;code agent 开始工具循环 |
| 14:23:33 | execute | 重启后(前一次 run 被外部中断)重新执行:agent **自写 train.py**(非模板) |
| ~14:24:31 | execute | `launch_experiment(dry_run=true)` → **passed**(自写脚本干跑 ok) |
| 14:25:14 | execute | 真实 launch 被拒:缺 `save_every_n_epochs/best_model.pth/log_metrics` → agent 读模板重写 → **max_turns=40 耗尽** |
| 14:25:14 | reflect | "实验未完成,无法判断 baseline 性能…重新指示 code agent 执行" |
| 14:25:17 | think→execute | **自动纠错**:cycle 2 code agent 基于模板重写 train.py |
| 14:26:41 | launch | 真实训练启动 PID=7840 → monitor 每 15s 轮询 |
| 14:27:19 | reflect | 里程碑:`test_acc=0.999`,假设 → confirmed(exp=7840) |
| 14:27:44 | reflect | 报告完成 → supervisor 路由 finish |
| 14:27:47 | **crash** | `print(final_answer)` → `UnicodeEncodeError: 'gbk' codec can't encode '\u2705'` |

**暴露的 7 个真实缺陷(全部修复):**

| # | 缺陷 | 证据 | 修复 |
|---|------|------|------|
| B1 | **GBK 控制台打印 crash**:final_answer 含 ✅,Windows GBK stdout 抛 UnicodeEncodeError,整进程 exit 1(仅差最后一行打印没打出来) | `UnicodeEncodeError: 'gbk' codec can't encode character '\u2705'` | `main()` 开头 `sys.stdout/stderr.reconfigure(encoding="utf-8", errors="replace")` |
| B2 | **干跑与真实启动校验不一致**:自写脚本缺模板契约 → 干跑 `ok:true` 写 marker,真实 launch 才报"缺少必需结构" → agent 返工烧光 max_turns | events seq 111-114:dry_run passed → launch error 缺结构 | 模板结构硬校验提前到**干跑前**(两模式共用),缺契约不执行不写 marker |
| B3 | **workspace 无模板种子**:agent 从零写 train.py(而非改模板),Windows 无 `cp` 命令反复失败(WinError 2 × 7 次)浪费轮次 | events seq 12/32/34/36/38/40/42:`run_shell` WinError 2 | `ResearchGraph.__init__` 幂等预置 `train_template.py` + `train.py`(模板副本,不覆盖已改文件) |
| B4 | **run_shell 描述无 Windows 提示**:agent 不知道本机无 Unix 命令 | 同上 | 工具描述加 Windows 注意:cp/cat/mv 不存在,复制用 read_file+write_file,模板已预置 |
| B5 | **假设结论被无信息量更新覆盖**:cycle 2 确认 `923afa30 → confirmed`;cycle 3 报告轮(无实验)else 分支 `mark_testing` 把它**打回 testing**,结论丢失 | hypotheses.db:status=testing 但 evidence 仍是"目标已达成…" | `mark_testing` 守卫:confirmed/refuted/inconclusive 终态不被降级;只能被新实验结果 resolve 覆盖 |
| B6 | **worker 成本不记账**:costs.jsonl 只有 leader 6 次调用($0.0046),code agent 80+ 次调用(实际成本大头)完全不可审计 | Run A costs.jsonl 仅 6 行 | 工具循环每轮调用后记 `cost_tracker.record_call(actor=code/writing, action=tool_loop)` |
| B7 | **元陈述当假设入库**:think 收尾轮把「无需新假设,目标已达成。」写进 hypothesis 字段,被当假设存入 | hypotheses.db `49fbc88d` | `_settle_hypothesis` 过滤元陈述(正则:`无需新假设/目标已达成/…`) |

---

## 3. Run B — 修复后全绿运行

**运行轨迹(smoke_run_B.log,14:32:37 → 14:34:54 = 2.28 min):**

| 时间 | 阶段 | 发生了什么 |
|------|------|-----------|
| 14:32:37 | 启动 | **模板种子生效**:workspace 预置 train.py/train_template.py(模板副本) |
| 14:32:43 | execute | code agent 读模板 → 编辑 TODO 区域 → 写 train.py(14052 B,模板+实现) |
| ~14:33:30 | launch | dry_run **passed**(结构校验+真实执行双通过)→ 真实启动 PID=30756 |
| 14:33:5x | monitor | 10 epochs 轮询,`METRIC_JSON` 逐行提取,loss 0.85→0.0188,test_acc→**0.998** |
| 14:34:0x | reflect | 里程碑:`test_acc=0.9980,远超 0.90(达成度 110.9%)` → 假设 `a937a9f6` **confirmed**(exp=30756) |
| 14:34:23 | execute | writing agent 生成 FINAL_REPORT.md |
| 14:34:39 | reflect | 决策树 `if test_acc>=0.90: 停止,记录结果` 命中 → 路由 finish |
| 14:34:52 | finish | 最终回答(含 ✅)正常打印 → `ResearchGraph 停止` → **exit 0** |

**与 Run A 对比:cycle 1 即一次成功**(Run A 需要 2 个 cycle 自动纠错),不再有返工;
结构校验/模板种子/Windows 提示三个修复消除了整类"写脚本-被拒-重写"浪费。

### 成本明细(逐笔记账,costs.jsonl)

| actor, action | 调用数 | 输入 tokens | 输出 tokens | 成本 USD |
|---------------|-------:|------------:|------------:|---------:|
| leader, think | 3 | 3,958 | 716 | 0.001856 |
| leader, reflect | 3 | 7,828 | 570 | 0.002740 |
| code, tool_loop | 35 | 305,841 | 7,316 | 0.090623 |
| writing, tool_loop | 9 | 15,097 | 2,178 | 0.006471 |
| **合计** | **50** | **332,724** | **10,780** | **0.10169** |

- **worker 工具循环占成本 89%**(code 35 次调用 $0.09)—— B6 修复前这部分完全不可见
- 成本大头是**输入 token 增长**(code agent 上下文随工具结果累积到 300k+),这是
  `_manage_context_window` 压缩机制的用武之地,也是访谈可讲的优化点

---

## 4. 证据索引

```
docs/smoke_evidence/run_A/   ← 首次真实运行证据
  experiments.jsonl          ← cycle1 未完成(如实记录) / cycle2 completed(test_acc=0.999)
  costs.jsonl                ← 仅 leader 记账(B6 修复前,证明缺口)
  baseline.log               ← 10 个 epoch 的 METRIC_JSON 行
  FINAL_REPORT.md / MEMORY_LOG.md / state.json / dry_run_log.json
docs/smoke_evidence/run_B/   ← 修复后全绿运行证据
  experiments.jsonl          ← cycle1 completed(test_acc=0.998,pid 30756) + 2 报告轮
  costs.jsonl                ← 50 笔逐笔记账,总额 $0.10169
  baseline_mlp.log           ← METRIC_JSON 契约行(monitor 解析的事实源)
  train.py                   ← agent 基于模板编辑的最终脚本(14052 B)
  FINAL_REPORT.md / MEMORY_LOG.md / state.json / dry_run_log.json
smoke_run.log                ← Run A 完整控制台 trace(含 crash 现场)
smoke_run_B.log              ← Run B 完整控制台 trace(含最终回答)
```

## 5. 相关回归测试(439 passed / 7 skipped)

- `tests/test_dryrun_gate.py` — 新增:干跑缺模板契约 → 立即拒绝、不写 marker;干跑失败回传 stderr
- `tests/test_hypotheses.py` — 新增:报告轮不覆盖 confirmed 结论(集成+store 守卫)、元陈述不入库
- 全套件在修复后全绿:**439 passed, 7 skipped**(修复前 434)

## 6. 已知限制(非阻塞,后续可做)

1. **deepseek 无 json_schema 结构化输出** → 每次 400 后文本降级,多一次往返;可用
   `response_format={type:'json_object'}`(deepseek 支持)优化,或换支持结构化输出的 provider
2. **假设去重是精确文本匹配**:「任务中」/「任务上」会建两条;可做归一化去重
3. **max_turns=40 对极端情况仍会耗尽**:B2/B3 已消除主要浪费源;剩余场景靠 reflect 自动纠错(实测有效)
4. **成本审计粒度**:worker 按轮记账已到位;按 cycle 聚合报表(dashboard)可作为下一步
5. FINAL_REPORT.md 日期字段由 LLM 自行填写(写了 2025-04-17),非系统 bug,提示词可约束
