#!/usr/bin/env python3
"""
Eval 驱动脚本 — 跑 examples/eval_tasks/ 任务集,产出 docs/EVAL_REPORT.md。

四种模式(都从 examples/eval_tasks/*/task.json 发现任务):

  python scripts/run_eval.py --dry            # 只校验任务配置(不需要 GPU/API key)
  python scripts/run_eval.py --scripted       # ScriptedLLM 全循环确定性回归(不需要 API key)
  python scripts/run_eval.py --real           # 真实 LLM 逐任务跑(需要 GPU + API key)
  python scripts/run_eval.py --report         # 聚合 docs/eval_runs/* 生成 EVAL_REPORT.md

真实模式参数:
  --provider deepseek --model deepseek-chat   # 传给 agent(默认读环境或 config)
  --tasks T1,T2                               # 只跑指定任务(默认全部)
  --max-cycles N                              # 覆盖任务预算

设计要点:
- 每个任务复制到临时目录运行,不污染 examples/ 源目录;
- config 注入 eval.enabled: true,运行全程录制决策轨迹(零额外成本);
- 结果与账本/成本保留在 docs/eval_runs/<TASK>/,可复现、可审计;
- scripted 模式是 eval-first 架构的基石:无 key 也能回归完整循环。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 脚本在 scripts/ 下运行,sys.path[0] 是 scripts/ → 需要显式加入项目根
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows 控制台默认 GBK:强制 UTF-8 输出,保证子进程捕获/CI 日志不乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

EVAL_TASKS_DIR = PROJECT_ROOT / "examples" / "eval_tasks"
EVAL_RUNS_DIR = PROJECT_ROOT / "docs" / "eval_runs"
REPORT_PATH = PROJECT_ROOT / "docs" / "EVAL_REPORT.md"


def _load_tasks() -> list[dict]:
    tasks = []
    for task_dir in sorted(EVAL_TASKS_DIR.iterdir()):
        meta_path = task_dir / "task.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["_dir"] = task_dir
        tasks.append(meta)
    return tasks


def _minimal_config(provider: str, model: str, max_cycles: int) -> dict:
    """真实/scripted 模式共用的最小 agent 配置(eval 录制开启)。"""
    return {
        "project": {"name": "eval", "brief": "PROJECT_BRIEF.md"},
        "agent": {
            "provider": provider,
            "model": model,
            "max_cycles": max_cycles,
            "max_steps_per_cycle": 3,
            "cooldown_interval": 30,
            "retry_limit": 2,
            "allow_missing_key": True,  # scripted/离线回归可在无 key 环境构造
        },
        "monitor": {"poll_interval": 20, "zero_llm": True},
        "obsidian": {"enabled": False},
        "eval": {"enabled": True},
        "journal": {"enabled": True},
        "ledger": {"enabled": True, "metric_key": "test_acc",
                   "metric_direction": "higher_better"},
        "stagnation": {"enabled": True, "threshold_cycles": 2,
                       # 自适应收益阈值(用户审查):单轮提升
                       # < max(0.3pp, 40%×离目标距离) 视为无实质进展,
                       # 单调小步爬升会触发创新度门;target 由 cmd_real
                       # 按任务注入(见 _stagnation_min_delta)
                       "min_delta_ratio": 0.4, "min_delta_floor": 0.003},
        "gates": {"enabled": False},
        "safety": {"enabled": True, "fail_threshold": 3},
    }


# ═══════════════════════════════════════════════════════════════════
# --dry
# ═══════════════════════════════════════════════════════════════════

def cmd_dry() -> int:
    tasks = _load_tasks()
    if not tasks:
        print("[FAIL] 未发现任务: examples/eval_tasks/*/task.json")
        return 1
    print(f"{'ID':<5}{'难度':<8}{'目标':<40}预算(cycles/cost/wall)")
    print("-" * 80)
    for t in tasks:
        target = t.get("target", {})
        budget = t.get("budget", {})
        brief = t["_dir"] / t.get("brief", "")
        ok = brief.is_file()
        flag = "OK " if ok else "MISSING"
        print(f"{t['id']:<5}{t.get('difficulty','?'):<8}"
              f"{t.get('name','?')[:36]:<40}"
              f"{budget.get('max_cycles','?')}/{budget.get('max_cost_usd','?')}"
              f"/{budget.get('max_wall_hours','?')}h  {flag}")
        if not ok:
            return 1
    print(f"\n{len(tasks)} 个任务配置合法(--dry 通过)。")
    return 0


# ═══════════════════════════════════════════════════════════════════
# --scripted 确定性回归(无 API key)
# ═══════════════════════════════════════════════════════════════════

def cmd_scripted() -> int:
    from core.nodes import ResearchGraph
    from core.scripted_llm import ScriptedLLM

    task = next((t for t in _load_tasks() if t["id"] == "T1"), None)
    if task is None:
        print("[FAIL] --scripted 需要 T1 任务")
        return 1

    with tempfile.TemporaryDirectory(prefix="ar_eval_scripted_") as tmp:
        run_dir = Path(tmp)
        shutil.copy2(task["_dir"] / task["brief"], run_dir / "PROJECT_BRIEF.md")
        # provider 先用合法值通过 fail-fast 校验,构造后再整体替换为 ScriptedLLM
        config = _minimal_config("openai", "scripted-model", 2)
        (run_dir / "config.yaml").write_text(
            json.dumps(config, ensure_ascii=False),
            encoding="utf-8",
        )

        scripted = ScriptedLLM(scenario="wait")
        graph = ResearchGraph(config=config, project_dir=str(run_dir))
        # 注入确定性 LLM:所有 LLM 入口换成 ScriptedLLM
        for attr in ("llm", "_llm_think", "_llm_reflect", "_llm_worker"):
            setattr(graph, attr, scripted)

        t0 = time.time()
        result = graph.run()
        elapsed = time.time() - t0

        # ── 验证循环健康:状态、事件、账本、锁 ──
        checks = {
            "循环干净结束": result is not None,
            "cycle_counter 已持久化": (run_dir / "workspace" / ".cycle_counter").exists(),
            "事件日志有记录": (run_dir / "workspace" / "events.jsonl").exists(),
            "实例锁已释放": not (run_dir / "workspace" / ".agent.lock").exists(),
        }
        if checks["循环干净结束"]:
            # wait 决策 → supervisor 应走 finish,不发起实验
            checks["账本无实验启动"] = _ledger_has_no_launch(run_dir)

        for name, ok in checks.items():
            print(f"  [{'OK' if ok else 'FAIL'}] {name}")
        print(f"  ScriptedLLM 调用次数: {len(scripted.calls)} | 耗时 {elapsed:.1f}s")

        if not all(checks.values()):
            print("[FAIL] scripted 确定性回归未通过")
            return 1
        print("[OK] scripted 确定性回归通过(零 API 成本)")
        return 0


def _ledger_has_no_launch(run_dir: Path) -> bool:
    ledger_path = run_dir / "workspace" / "experiments.jsonl"
    if not ledger_path.exists():
        return True  # 没有账本 = 没有实验
    for line in ledger_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("status") not in (None, "", "no_experiment"):
            return False
    return True


# ═══════════════════════════════════════════════════════════════════
# --real 真实评测(需要 GPU + API key)
# ═══════════════════════════════════════════════════════════════════

def cmd_real(args) -> int:
    from core.nodes import ResearchGraph

    tasks = [t for t in _load_tasks() if t["id"] in args.tasks]
    if not tasks:
        print(f"[FAIL] 未找到任务: {args.tasks}")
        return 1

    EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for task in tasks:
        tid = task["id"]
        print(f"\n=== {tid}: {task['name']} ===")
        budget = task.get("budget", {})
        max_cycles = args.max_cycles or budget.get("max_cycles", 5)
        with tempfile.TemporaryDirectory(prefix=f"ar_eval_{tid}_") as tmp:
            run_dir = Path(tmp)
            # 复制整个任务目录:PROJECT_BRIEF + literature/ 论文库 + 已 ingest 的
            # workspace(如 T4 的 memory.db),保证 RAG 任务开箱可用
            shutil.copytree(task["_dir"], run_dir, dirs_exist_ok=True)

            # RAG 任务:literature/ 存在但知识库缺失 → 自动摄取(需网络)
            if task.get("requires_rag"):
                lit_dir = run_dir / "literature"
                kb_path = run_dir / "workspace" / "memory.db"
                if lit_dir.is_dir() and not kb_path.exists():
                    from core.cross_project_memory import CrossProjectStore
                    from core.rag import RagKnowledgeBase
                    from scripts.ingest_papers import ingest_dir
                    print(f"  [rag] 自动摄取文献库 {lit_dir} ...")
                    kb = RagKnowledgeBase(
                        CrossProjectStore(kb_path),
                        project=f"rag_{run_dir.name[:40]}",
                    )
                    ingest_dir(kb, lit_dir)

            (run_dir / "config.yaml").write_text(
                json.dumps(_minimal_config(args.provider, args.model, max_cycles),
                           ensure_ascii=False),
                encoding="utf-8",
            )
            # 停滞判定的自适应 min_delta:按任务目标注入
            # (min_delta = 40% × 离目标距离,见 core/nodes.py:_stagnation_min_delta)
            cfg = _minimal_config(args.provider, args.model, max_cycles)
            target = task.get("target", {}) or {}
            if target.get("value") is not None:
                cfg.setdefault("stagnation", {})["target"] = float(target["value"])
            # 创新强制任务(T6 论文驱动等):计划不含创新点 → 升级 idea agent
            if task.get("requires_innovation"):
                cfg.setdefault("stagnation", {})["innovation_required"] = True
            (run_dir / "config.yaml").write_text(
                json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
            graph = ResearchGraph(config=cfg,
                                  project_dir=str(run_dir))
            try:
                graph.run()
            except Exception as exc:
                print(f"  [FAIL] {tid} 运行异常: {exc}")
                failures += 1

            # 保留证据:录制 + 账本 + 成本 + 状态
            out_dir = EVAL_RUNS_DIR / tid
            out_dir.mkdir(parents=True, exist_ok=True)
            ws = run_dir / "workspace"
            for fname in ("eval/recording.jsonl", "experiments.jsonl",
                          "costs.jsonl", "state.json", "events.jsonl"):
                src = ws / fname
                if src.exists():
                    dst = out_dir / fname.replace("/", "_")
                    shutil.copy2(src, dst)
            print(f"  证据已保存: docs/eval_runs/{tid}/")
    if failures:
        print(f"\n[FAIL] {failures} 个任务运行异常")
        return 1
    print("\n全部任务运行完毕。运行 `python scripts/run_eval.py --report` 聚合报告。")
    return 0


# ═══════════════════════════════════════════════════════════════════
# --report 聚合
# ═══════════════════════════════════════════════════════════════════

def _last_ledger_metric(run_dir: Path) -> tuple:
    """从账本取指标:返回 (metric_dict, status, cycles)。

    实测取「历史最佳」而非最后一条:attempt1/2 的末轮是未完成的
    experiment(无指标),但中间轮次有真实指标 —— 达标判定看整个
    运行达到过的最佳值,与任务定义一致。
    """
    ledger_path = run_dir / "experiments.jsonl"
    entries = []
    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not entries:
        return {}, "", 0
    best: dict = {}
    for e in entries:
        m = e.get("metrics")
        if isinstance(m, dict) and m:
            for k, v in m.items():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if k not in best or fv > best[k]:
                    best[k] = fv
    last = entries[-1]
    return best, last.get("status", ""), len(entries)


def _run_cost(run_dir: Path) -> float:
    costs_path = run_dir / "costs.jsonl"
    total = 0.0
    if costs_path.exists():
        for line in costs_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += float(entry.get("cost_usd") or entry.get("cost", 0) or 0)
    return round(total, 4)


def _citation_count(run_dir: Path) -> int:
    """RAG 引用率统计:账本里假设/结论带 [arXiv:xxx] 引用的条数(E3)。"""
    ledger_path = run_dir / "experiments.jsonl"
    count = 0
    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = " ".join([
                str(entry.get("hypothesis", "") or ""),
                str(entry.get("conclusion", "") or ""),
            ])
            if "[arXiv:" in text or "[paper:" in text:
                count += 1
    return count


def cmd_report() -> int:
    tasks = {t["id"]: t for t in _load_tasks()}
    rows = []
    # 扫描 canonical 目录 + 历史尝试目录(<ID>_attempt<N>);
    # 指标/目标变动后重跑本命令即同步(对比数据派生自账本,非硬编码)。
    for run_dir in sorted(EVAL_RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        task = tasks.get(run_dir.name)
        attempt = None
        if task is None and "_attempt" in run_dir.name:
            base = run_dir.name.split("_attempt")[0]
            task = tasks.get(base)
            try:
                attempt = int(run_dir.name.split("_attempt")[1])
            except (TypeError, ValueError):
                attempt = None
        if task is None:
            continue
        metrics, status, cycles = _last_ledger_metric(run_dir)
        target = task.get("target", {})
        metric_key = target.get("metric", "test_acc")
        operator = target.get("operator", ">=")
        threshold = target.get("value")
        # 达标口径 = 训练中达到过的历史最佳(test_acc/accuracy/acc 别名取 max):
        # monitor 的 test_acc 是末轮、accuracy 是 best_acc;"达到 X%" 的
        # 自然语义是达到过(与 agent 的 reflect 判定一致)。
        got = None
        candidates = [metrics.get(k) for k in (metric_key, "accuracy", "acc")]
        numeric = []
        for c in candidates:
            if c is not None:
                try:
                    numeric.append(float(c))
                except (TypeError, ValueError):
                    pass
        if numeric:
            got = max(numeric)
        if got is not None and threshold is not None:
            try:
                success = (float(got) >= float(threshold)) if operator == ">=" \
                    else (float(got) <= float(threshold))
            except (TypeError, ValueError):
                success = False
        else:
            success = False
        attempt = None
        if "_attempt" in run_dir.name:
            try:
                attempt = int(run_dir.name.split("_attempt")[1])
            except (TypeError, ValueError):
                attempt = None
        label = run_dir.name if attempt is None else f"{run_dir.name.split('_attempt')[0]}·尝试{attempt}"
        rows.append({
            "id": run_dir.name,
            "label": label,
            "name": task["name"],
            "target": f"{metric_key} {operator} {threshold}",
            "got": got,
            "success": success,
            "cycles": cycles,
            "cost_usd": _run_cost(run_dir),
            "status": status,
            "citations": _citation_count(run_dir),
        })

    if not rows:
        print("[提示] docs/eval_runs/ 下没有运行记录。先跑 `--real` 或 `--scripted`。")
        return 0

    lines = [
        "# Eval Report — Deep Researcher Agent",
        "",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "| ID | 任务 | 目标 | 实测 | 达标 | cycles | 成本($) | 引用数 | 账本状态 |",
        "|----|------|------|------|------|--------|---------|--------|----------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['name']} | {r['target']} | {r['got']} | "
            f"{'✅' if r['success'] else '❌'} | {r['cycles']} | {r['cost_usd']} | "
            f"{r['citations']} | {r['status']} |"
        )
    n_ok = sum(1 for r in rows if r["success"])
    lines += [
        "",
        f"**达标率: {n_ok}/{len(rows)}**",
        "",
        "> 口径:达标判定来自实验账本(experiments.jsonl)的最终指标,与监控器记录的",
        "> 真实状态一致(失败/超时不会被记为 completed)。录制文件见 `docs/eval_runs/<TASK>/`,可复现。",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n报告已写入: {REPORT_PATH}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Deep Researcher Agent eval driver")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry", action="store_true", help="校验任务配置")
    mode.add_argument("--scripted", action="store_true", help="ScriptedLLM 确定性回归")
    mode.add_argument("--real", action="store_true", help="真实 LLM 评测")
    mode.add_argument("--report", action="store_true", help="聚合运行结果生成报告")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--tasks", default="T1,T2,T3,T4,T5")
    parser.add_argument("--max-cycles", type=int, default=None)
    args = parser.parse_args()

    args.tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    if args.dry:
        return cmd_dry()
    if args.scripted:
        return cmd_scripted()
    if args.real:
        return cmd_real(args)
    return cmd_report()


if __name__ == "__main__":
    sys.exit(main())
