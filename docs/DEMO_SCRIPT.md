# Demo Script — 3 分钟面试演示

> 目标:让面试官在 3 分钟内理解"自主深度学习实验 agent"的完整闭环,
> 每段都能指路到代码/文档。

## 0:00–0:30 — 一句话定位

"这是一个 24/7 自主跑深度学习实验的 agent。你给它一句话目标,它自动想 idea、
写代码、跑实验、分析结果,然后基于结果优化下一轮,直到达标。"
(顺手展示:README 徽章 + 361→434 测试的 commit 历史)

## 0:30–1:10 — 代码级可信度(讲 2 个故事)

1. **系统干跑门**:`launch_experiment(dry_run=true)` —— 干跑由系统用绑定
   解释器执行,写权威 marker(解释器+脚本指纹+依赖指纹),环境不一致在机制上不可能。
   → 打开 `core/nodes.py` 搜 `dry_run` 指给面试官
2. **契约化指标**:`train_template.py` 的 `log_metrics()` 输出 `METRIC_JSON` 行,
   monitor 优先解析,字段名原样(test_acc 直达账本)—— 展示测试
   `tests/test_contract_metrics.py` 的"test_acc 断裂修复"注释

## 1:10–2:10 — 真跑一个循环(scripted,零成本)

```bash
python scripts/run_eval.py --scripted
```
展示输出:循环干净结束 / cycle_counter / 事件日志 / 锁释放 / 账本无实验误记。
强调:"**整个 agent 循环可以在零 API 成本下做确定性回归** —— 正确性靠回归,
质量靠真实评测,LLM 只负责质量不负责正确性。"

## 2:10–2:50 — 控制台(如果时间允许起 dashboard)

```bash
python -m core.dashboard --project examples/eval_tasks/T1_mnist --port 8000
```
展示:状态栏 / 指令框(发一条 HUMAN_DIRECTIVE)/ 工作区文件树 / ⚙️ LLM 设置面板。
(或只截图展示,节省时间)

## 2:50–3:00 — 收尾金句

"**记忆是实验账本不是聊天记忆;成本逐 cycle 可审计;失败训练不伪装成
completed;工具层协议无关(内嵌 bind_tools + 可选 MCP);HITL 是可选保险丝,
默认零打断。**"

## 备选深挖(面试官追问时)

- 安全:shell 注入/路径越界/符号链接泄漏 → `tests/test_tools_security.py`
- 架构:supervisor 为什么不用 LLM 路由 → `docs/architecture.md` §2
- 假设生命周期:已否证的假设不再提 → `core/hypotheses.py`
- 环境:自动创建/钉死 → `docs/USER_GUIDE.md` 环境策略
- 风险三修:工具熔断/crash-resume/checkpoint 备份 → `tests/test_tool_loop_fuse.py` 等
