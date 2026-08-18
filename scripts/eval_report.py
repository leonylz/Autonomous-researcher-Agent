#!/usr/bin/env python3
"""
Agent Eval 报告工具 —— 对照 golden dataset 评估一次录制。

录制来源：config.yaml 中 `eval.enabled: true` 时，真实运行会把决策轨迹
写入 workspace/eval/recording.jsonl（零额外 LLM 成本）。

用法：
  python scripts/eval_report.py                  # 用默认路径评估
  python scripts/eval_report.py --init-golden    # 从录制生成 golden 标注模板
  python scripts/eval_report.py --recording PATH --golden PATH

指标（见 core/eval.py EvalReport）：
  - action_match_rate     动作精确匹配率（chosen vs expected 的 action+agent）
  - tool_select_accuracy  工具选择准确率（golden expected_tools 的命中率）
  - cycle_success_rate    cycle 成功率（录制 result 非 failed/error 的比例）

口径说明（写简历/面试前必读）：
  - 动作匹配是精确字符串匹配，无语义容错，数字偏保守
  - cycle 成功率来自录制 result 字段：worker 正常结束（done）也算成功，
    不等于「实验达标」
  - 没有 golden 标注的录制行不参与计分；golden 标注质量决定数字可信度
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

# Windows 控制台默认 GBK 会导致中文乱码，强制 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# core/eval.py 只依赖标准库；按文件路径直接加载，避免 import core 包
# 时连带触发 nodes.py 的 langchain/pytorch 等完整依赖链。
# 离线评分不需要安装任何项目依赖。
def _load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "autoresearcher_eval", PROJECT_ROOT / "core" / "eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["autoresearcher_eval"] = module  # dataclass 装饰器需要模块注册
    spec.loader.exec_module(module)
    return module


_eval = _load_eval_module()
AgentRecorder = _eval.AgentRecorder
evaluate_recording = _eval.evaluate_recording

DEFAULT_RECORDING = PROJECT_ROOT / "workspace" / "eval" / "recording.jsonl"
DEFAULT_GOLDEN = PROJECT_ROOT / "workspace" / "eval" / "golden.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def cmd_init_golden(golden: Path, recording: Path, force: bool) -> int:
    entries = AgentRecorder(recording).entries()
    if not entries:
        print("[FAIL] 未找到录制数据:", recording)
        print("       先开启 config.yaml 的 eval.enabled: true，跑几轮真实实验后再试。")
        return 1
    if golden.exists() and not force:
        print("[FAIL] golden 已存在:", golden)
        print("       加 --force 覆盖重生成（会丢失已有标注）。")
        return 1

    lines = []
    for i, e in enumerate(entries, 1):
        lines.append({
            "actor": e.get("actor"),
            "action": e.get("action"),
            "cycle": e.get("cycle"),
            "expected_action": "",
            "expected_agent": "",
            "expected_tools": [],
            "note": (
                f"#L{i} 实际: action={e.get('chosen_action')}, "
                f"agent={e.get('chosen_agent')}, tools={e.get('tools_used', [])} | "
                f"决策输出: {(e.get('output_snippet') or '')[:150]}"
            ),
        })
    with open(golden, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"[OK] 已生成 {len(lines)} 行标注模板:", golden)
    print()
    print("标注方法：")
    print("  1. 每行填三个字段：expected_action / expected_agent / expected_tools")
    print("     （即「这一步应该做什么、派给谁、用什么工具」）")
    print("  2. 不想评的行直接删除 —— 没有 golden 的录制行自动跳过，不参与计分")
    print("  3. 建议每个 cycle 保留 1 条 leader think + 1 条对应 worker execute，")
    print("     其余行删掉，标注量最小")
    print("  4. 标注完运行: python scripts/eval_report.py")
    return 0


def cmd_report(recording: Path, golden: Path) -> int:
    if not recording.exists():
        print("[FAIL] 未找到录制数据:", recording)
        print("       先开启 config.yaml 的 eval.enabled: true，跑几轮真实实验后再试。")
        return 1
    if not golden.exists():
        print("[FAIL] 未找到 golden dataset:", golden)
        print("       先运行: python scripts/eval_report.py --init-golden")
        return 1

    report = evaluate_recording(recording, golden)

    print("=" * 60)
    print("Agent Eval 评估报告")
    print("=" * 60)
    print("录制文件:", recording)
    print("golden   :", golden)
    print(f"参与计分条目: {report.total}")
    print("-" * 60)
    print(f"动作匹配率     : {report.action_match_rate * 100:.1f}%  ({report.action_match}/{report.total})")
    print(f"工具选择准确率 : {report.tool_select_accuracy * 100:.1f}%  ({report.tool_hits}/{report.tool_checks})")
    print(f"cycle 成功率   : {report.cycle_success_rate * 100:.1f}%  ({report.cycles_success}/{report.cycles_total})")
    print("=" * 60)

    mismatches = []
    for d in report.details:
        if not d.get("action_match"):
            mismatches.append(d)
        elif d.get("expected_tools") and any(
            t not in set(d.get("tools_used", [])) for t in d["expected_tools"]
        ):
            mismatches.append(d)
    if mismatches:
        print()
        print(f"不匹配明细（{len(mismatches)} 条）：")
        for d in mismatches:
            print(f"  {d.get('actor')}/{d.get('action')}")
            print(f"    实际: action={d.get('chosen_action')}, agent={d.get('chosen_agent')}, tools={d.get('tools_used')}")
            print(f"    期望: action={d.get('expected_action')}, agent={d.get('expected_agent')}, tools={d.get('expected_tools')}")

    if report.total == 0:
        print()
        print("[!] 没有任何录制条目与 golden 匹配计分，请检查 golden 的 (actor, action) 键。")
        return 0

    # ── 警示 ──
    golden_lines = load_jsonl(golden)
    unannotated = sum(1 for g in golden_lines if not g.get("expected_action"))
    if unannotated:
        print()
        print(f"[!] golden 中有 {unannotated} 行未标注（expected_action 为空），已计为不匹配。")
        print("    不想评的行请删除，想评的请补全后重跑。")
    rec_keys = {(e.get("actor"), e.get("action")) for e in AgentRecorder(recording).entries()}
    orphans = [g for g in golden_lines if (g.get("actor"), g.get("action")) not in rec_keys]
    if orphans:
        print()
        print(f"[!] golden 中有 {len(orphans)} 行的 (actor, action) 在录制中不存在，永远不会被计分。")
    if report.total < 30:
        print()
        print(f"[!] 参与计分仅 {report.total} 条（<30），统计噪声大，只能当初步评测；")
        print("    想有可信数字，继续攒 cycle 到 30+ 条再评。")

    print()
    print("口径说明（写简历/面试前必读）：")
    print("  - 动作匹配是精确字符串匹配，无语义容错，数字偏保守")
    print("  - cycle 成功率来自录制 result 字段：worker 正常结束（done）也算成功，")
    print("    不等于「实验达标」")
    print("  - 未标注的录制行不参与计分；golden 标注质量决定数字可信度")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent Eval 报告工具")
    parser.add_argument("--init-golden", action="store_true",
                        help="从录制生成 golden 标注模板")
    parser.add_argument("--force", action="store_true",
                        help="--init-golden 时覆盖已有 golden")
    parser.add_argument("--recording", type=Path, default=DEFAULT_RECORDING)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    args = parser.parse_args()

    if args.init_golden:
        return cmd_init_golden(args.golden, args.recording, args.force)
    return cmd_report(args.recording, args.golden)


if __name__ == "__main__":
    sys.exit(main())
