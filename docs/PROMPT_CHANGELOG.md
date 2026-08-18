# Prompt Changelog — Prompt ↔ Eval 挂钩记录

> 规则:任何 `agents/*.md` 或 `core/nodes.py` 内嵌 prompt 的修改,
> 必须在此登记,并在可行时对照 eval 任务集记录效果。
> 离线检查:`tests/test_prompt_quality.py`(结构不变量)。

| 日期 | PROMPT_VERSION | 修改 | 目的 | eval 对照 |
|------|----------------|------|------|-----------|
| — | leader 2.0 | 决策树纪律(brief What to Try 是约束非建议;偏离需说明已穷尽的分支) | 防 planner 天马行空发明创新点 | 待 T1-T3 真实跑(脚本就绪) |
| — | leader 2.0 | 假设必须带论文引用 [arxiv_id] | RAG 闭环:假设可溯源 | T4 golden 断言引用率 |
| — | idea 2.0 | Citation Traceability 段(假设/建议必须引用) | 同上 | T4 golden |
| — | code 2.0 | 干跑改为 launch_experiment(dry_run=true);git 工具 | 系统干跑门;复现快照 | test_dryrun_gate(自动化) |
| — | reflect 1.0 | 目标回顾段(防 specification drift) | 防目标漂移 | 待真实 eval |

## 如何登记一次 prompt 修改

1. `agents/<x>.md` frontmatter 的 `PROMPT_VERSION` +1;
2. 本表加一行:修改内容、目的、eval 对照(测试名或"待真实 eval");
3. 跑 `python -m pytest tests/test_prompt_quality.py`(结构不变量);
4. 有真实 eval 条件时,在改动前后各跑一次 `scripts/run_eval.py --real --tasks T1`,
   把数字填进"eval 对照"列。
