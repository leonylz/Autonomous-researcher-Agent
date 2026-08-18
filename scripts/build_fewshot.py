#!/usr/bin/env python3
"""
build_fewshot — 从真实运行录制中提取 few-shot 候选样例。

素材:docs/eval_runs/<TASK>*/eval_recording.jsonl(每次真实运行的
(prompt_snippet, output_snippet) 对 + code agent 的 tools_used 工具序列)。

用途(两阶段自举):
  阶段 1(种子):提取真实输出 → 人工挑选 + 英译 → 嵌入 agents/*.md 的
               Examples 段(结构真实,格式与解析器 100% 兼容)。
  阶段 2(自举):英文提示词跑通后,用新录制的英文真实轨迹刷新。

用法:
  python scripts/build_fewshot.py            # 汇总所有候选到 stdout
  python scripts/build_fewshot.py --json     # 输出结构化候选(便于脚本处理)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_RUNS_DIR = PROJECT_ROOT / "docs" / "eval_runs"


def _load_records() -> list[dict]:
    records = []
    for run_dir in sorted(EVAL_RUNS_DIR.iterdir()):
        rec_path = run_dir / "eval_recording.jsonl"
        if not rec_path.is_file():
            continue
        for line in rec_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _is_valid_leader_output(out: str) -> bool:
    """筛选标准:输出是合法 JSON,action 合法,关键字段完整。"""
    try:
        d = json.loads(out)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(d, dict):
        return False
    if d.get("action") not in ("experiment", "wait", "report"):
        return False
    if d.get("action") == "experiment" and not d.get("task"):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract few-shot candidates from real recordings")
    parser.add_argument("--json", action="store_true", help="emit structured JSON candidates")
    parser.add_argument("--leader-only", action="store_true",
                        help="only leader think/reflect records")
    args = parser.parse_args()

    records = _load_records()
    if not records:
        print("[提示] docs/eval_runs/*/eval_recording.jsonl 下没有录制。先跑 --real。")
        return 1

    candidates = []
    for r in records:
        actor = r.get("actor", "")
        action = r.get("action", "")
        if args.leader_only and actor != "leader":
            continue
        out = r.get("output_snippet", "")
        valid = _is_valid_leader_output(out) if actor == "leader" else True
        candidates.append({
            "run": str(Path(r.get("_run", "")).name) if r.get("_run") else "?",
            "actor": actor,
            "action": action,
            "cycle": r.get("cycle"),
            "chosen_action": r.get("chosen_action"),
            "chosen_agent": r.get("chosen_agent"),
            "tools_used": r.get("tools_used", []),
            "valid_json": valid,
            "output": out,
        })

    if args.json:
        print(json.dumps(candidates, ensure_ascii=False, indent=1))
        return 0

    n_valid = sum(1 for c in candidates if c["valid_json"])
    print(f"录制文件: {len(records)} 条记录;leader 合法 JSON 输出: {n_valid}")
    print("=" * 70)
    for c in candidates:
        tag = "OK " if c["valid_json"] else "BAD"
        tools = ",".join(c["tools_used"][:6]) if c["tools_used"] else "-"
        print(f"[{tag}] {c['actor']}/{c['action']} cycle={c['cycle']} "
              f"-> {c['chosen_action']}/{c['chosen_agent']} | tools: {tools}")
        if c["valid_json"]:
            try:
                d = json.loads(c["output"])
                print(f"      task: {str(d.get('task', ''))[:90]}")
                print(f"      hypo: {str(d.get('hypothesis', ''))[:90]}")
            except json.JSONDecodeError:
                pass
    print("=" * 70)
    print("挑选标准:valid_json=OK 且 task/hypothesis 完整;剔除元陈述"
          "(如 hypothesis='无需新假设')与散文输出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
