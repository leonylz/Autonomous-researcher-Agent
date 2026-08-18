#!/usr/bin/env python3
"""
extract_trace — 从 events.jsonl 提取指定 agent 的完整工具轨迹(观测"怎么找")。

用法:
  python scripts/extract_trace.py <run_dir> --agent idea [--out docs/idea_trace_T6.md]
  python scripts/extract_trace.py <run_dir> --summary     # 每 cycle 路由/工具统计

输出:工具调用序列(时间/工具/参数)+ 路由摘要;用于验证 idea agent
是否真的 list_files literature/ → read_file 论文 → write_file IDEA_NOTES,
以及它读了几篇、引用了哪些论文。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_events(run_dir: Path) -> list[dict]:
    # 兼容两种布局:docs/eval_runs/<TASK>/events.jsonl 与
    # 运行中临时目录 <tmp>/workspace/events.jsonl
    for candidate in (run_dir / "events.jsonl",
                      run_dir / "workspace" / "events.jsonl"):
        if candidate.is_file():
            path = candidate
            break
    else:
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def extract_agent_trace(run_dir: Path, agent: str) -> list[dict]:
    """指定 agent 的 (tool_call, tool_result) 序列。"""
    out = []
    for e in _load_events(run_dir):
        if e.get("type") in ("tool_call", "tool_result") and \
                (e.get("payload") or {}).get("agent") == agent:
            out.append(e)
    return out


def routing_summary(run_dir: Path) -> list[dict]:
    """每 cycle 的 worker 路由与工具统计(从 events 反推)。"""
    rows = []
    current = None
    for e in _load_events(run_dir):
        p = e.get("payload") or {}
        if e.get("type") == "tool_call":
            agent = p.get("agent", "?")
            key = (e.get("phase", ""), agent)
            if current is None or current["key"] != key:
                current = {"key": key, "phase": e.get("phase", ""),
                           "agent": agent, "tools": [], "n": 0}
                rows.append(current)
            current["tools"].append(p.get("tool", "?"))
            current["n"] += 1
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract agent tool traces from events.jsonl")
    parser.add_argument("run_dir", type=Path, help="docs/eval_runs/<TASK> directory")
    parser.add_argument("--agent", default="idea", help="agent to trace (idea/code/writing)")
    parser.add_argument("--out", type=Path, default=None,
                        help="write markdown trace to this file")
    parser.add_argument("--summary", action="store_true", help="print routing summary only")
    args = parser.parse_args()

    if not ((args.run_dir / "events.jsonl").exists()
            or (args.run_dir / "workspace" / "events.jsonl").exists()):
        print(f"[FAIL] 无 events.jsonl: {args.run_dir}")
        return 1

    if args.summary:
        print(f"# Routing Summary — {args.run_dir.name}\n")
        for r in routing_summary(args.run_dir):
            tools = ",".join(dict.fromkeys(r["tools"]))
            print(f"- {r['phase']} / {r['agent']}: {r['n']} calls [{tools}]")
        return 0

    trace = extract_agent_trace(args.run_dir, args.agent)
    if not trace:
        print(f"[提示] {args.run_dir.name} 中没有 agent='{args.agent}' 的工具调用"
              f"(该 agent 可能从未被路由)。")
        return 1

    lines = [f"# {args.agent} agent trace — {args.run_dir.name}", ""]
    reads = set()
    writes = set()
    for e in trace:
        p = e.get("payload") or {}
        tool = p.get("tool", "?")
        arg_str = str(p.get("args", ""))[:160]
        if tool == "read_file":
            reads.add(arg_str)
        if tool == "write_file":
            writes.add(arg_str)
        ts = e.get("ts", 0)
        stamp = f"{int(ts % 100000):06d}"
        lines.append(f"{stamp} `{tool}` {arg_str}")
    lines += [
        "",
        f"## Summary",
        f"- read_file targets: {len(reads)} unique",
        f"- write_file targets: {len(writes)} unique",
    ]
    text = "\n".join(lines)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"已写入: {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
