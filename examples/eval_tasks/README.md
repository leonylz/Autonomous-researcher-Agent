# Eval Tasks — Deep Researcher Agent 评测任务集

这是本项目的**具体场景与评测基准**:5 个难度递增的科研任务,每个任务 = 一份
`PROJECT_BRIEF.md`(给 agent 的目标/约束/决策树)+ `task.json`(机器可读的
目标指标与预算)+ 可选 `golden.jsonl`(人工标注的决策基准,用于
`scripts/eval_report.py` 的 action-match 评估)。

## 任务一览

| ID | 任务 | 目标 | 难度 | 依赖 |
|----|------|------|------|------|
| T1 | MNIST CNN 从 0 优化 | test_acc ≥ 99.5% | 中等 | CPU 可跑(数据本地缓存) |
| T2 | CIFAR-10 ResNet-18 从 0 优化 | test_acc ≥ 85% | 中等 | GPU |
| T3 | 修复不收敛的配置 | loss 收敛(不再发散) | 中等 | GPU |
| T4 | 论文方法复现(mixup + cosine) | 达到论文报告精度区间 | 难 | GPU + 论文库(RAG) |
| T5 | 训练崩溃检测与恢复 | 崩溃后自动恢复续训 | 难 | GPU |

## 打分维度

| 维度 | 说明 |
|------|------|
| task success | 是否在预算内达到目标指标(truthful:以实验账本 final 状态为准) |
| cycles | 用了多少个循环(越少越好) |
| LLM 成本 | CostTracker 记录的 token 成本(美元,可审计) |
| 墙钟时间 | 从启动到达标/预算耗尽 |
| 引用率(T4) | 假设中带论文引用的比例(RAG 闭环质量) |

## 运行方式

```bash
# 0) 校验任务配置(不需要 GPU/API key)
python scripts/run_eval.py --dry

# 1) 确定性回归:ScriptedLLM 驱动完整循环(不需要 API key)
python scripts/run_eval.py --scripted

# 2) 真实评测(需要 GPU + API key,配置见 scripts/run_eval.py 顶部)
python scripts/run_eval.py --real --provider deepseek --model deepseek-chat

# 3) 人工标注 golden(可选,提升报告可信度)
python scripts/eval_report.py --recording docs/eval_runs/T1/recording.jsonl --init-golden

# 4) 聚合报告
python scripts/run_eval.py --report
```

结果写入 `docs/EVAL_REPORT.md`;每次真实运行的录制保留在 `docs/eval_runs/<TASK>/`,
可复现、可审计。
