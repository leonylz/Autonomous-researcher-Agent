"""
Agent Eval 框架：录制 → 回放 → 报告。

解决"Agent 决策质量无法评估"的问题：
  - AgentRecorder  — 运行期录制 LLM 调用/工具调用/cycle 结果到 JSONL
  - AgentReplayer  — 离线回放录制数据（输入重新喂给 LLM 对比，或纯分析）
  - evaluate_recording — 对照 golden dataset 生成 EvalReport

指标：
  - action_match_rate   动作精确匹配率（recorded vs golden 的 expected_action）
  - tool_select_accuracy 工具选择正确率（golden 标注 expected_tools 的命中率）
  - cycle_success_rate  cycle 成功率（recorded 的 result 含 success/failed）

录制文件格式（JSONL，每行一条）:
  {"actor": "leader", "action": "think", "prompt_snippet": "...",
   "output_snippet": "...", "chosen_action": "experiment", "chosen_agent": "code",
   "tools_used": ["run_shell"], "cycle": 1, "ts": 1700000000.0}

Golden dataset 格式（JSONL）:
  {"actor": "leader", "action": "think", "expected_action": "experiment",
   "expected_agent": "code", "expected_tools": ["launch_experiment"], "note": "..."}
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger("autoresearcher.eval")


# ═══════════════════════════════════════════════════════════════════
# AgentRecorder — 运行期录制
# ═══════════════════════════════════════════════════════════════════

class AgentRecorder:
    """录制 Agent 决策轨迹到 JSONL（append-only，单写者）。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record_llm(self, actor: str, action: str, prompt_snippet: str,
                   output_snippet: str, chosen_action: str = "",
                   chosen_agent: str = "", cycle: Optional[int] = None) -> None:
        """录制一次 LLM 决策调用。"""
        self._append({
            "actor": actor, "action": action,
            "prompt_snippet": (prompt_snippet or "")[:1000],
            "output_snippet": (output_snippet or "")[:1000],
            "chosen_action": chosen_action, "chosen_agent": chosen_agent,
            "cycle": cycle, "ts": time.time(),
        })

    def record_worker(self, agent_type: str, task_snippet: str,
                      tools_used: list, result: dict,
                      cycle: Optional[int] = None) -> None:
        """录制一次 worker 工具循环。"""
        self._append({
            "actor": f"{agent_type}_agent", "action": "execute",
            "prompt_snippet": (task_snippet or "")[:1000],
            "output_snippet": (str(result.get("response", "")) or "")[:1000],
            "chosen_action": "experiment", "chosen_agent": agent_type,
            "tools_used": [t.get("name", "") for t in tools_used][:20],
            "result": result.get("error") and "failed" or
                      ("completed" if result.get("experiment_launched") else "done"),
            "cycle": cycle, "ts": time.time(),
        })

    def _append(self, entry: dict) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("eval recorder append failed: %s", exc)

    def entries(self) -> list[dict]:
        """读取全部录制条目（跳过坏行）。"""
        if not self.path.exists():
            return []
        out = []
        with open(self.path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


# ═══════════════════════════════════════════════════════════════════
# AgentReplayer — 离线回放
# ═══════════════════════════════════════════════════════════════════

class AgentReplayer:
    """离线回放录制数据。

    两种模式：
      - 纯分析（analyze=True）：直接读取录制条目，不重新调用 LLM
      - 重放对比：逐条把 prompt_snippet 喂给 llm_fn(prompt) 得到新输出，
        与录制输出对比（用于验证模型升级/提示词改动后的行为漂移）
    """

    def __init__(self, recording_path: Path):
        self.recorder = AgentRecorder(recording_path)

    def replay(self, llm_fn=None, sample: Optional[int] = None) -> list[dict]:
        """回放全部（或前 sample 条）录制条目。

        llm_fn 为 None → 纯分析模式，返回原条目。
        llm_fn 提供 → 每条重放 LLM 调用，附 replayed_output 与 match 标记。
        """
        entries = self.recorder.entries()
        if sample:
            entries = entries[:sample]
        out = []
        for e in entries:
            row = dict(e)
            if llm_fn is not None:
                try:
                    replayed = llm_fn(e.get("prompt_snippet", ""))
                    row["replayed_output"] = (replayed or "")[:1000]
                    row["replay_match"] = (
                        str(replayed).strip() == str(e.get("output_snippet", "")).strip()
                    )
                except Exception as exc:
                    row["replay_error"] = str(exc)[:200]
            out.append(row)
        return out


# ═══════════════════════════════════════════════════════════════════
# EvalReport — 对照 golden dataset 评估
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EvalReport:
    total: int = 0
    action_match: int = 0
    tool_checks: int = 0
    tool_hits: int = 0
    cycles_total: int = 0
    cycles_success: int = 0
    details: list[dict] = field(default_factory=list)

    @property
    def action_match_rate(self) -> float:
        return round(self.action_match / self.total, 4) if self.total else 0.0

    @property
    def tool_select_accuracy(self) -> float:
        return round(self.tool_hits / self.tool_checks, 4) if self.tool_checks else 0.0

    @property
    def cycle_success_rate(self) -> float:
        return round(self.cycles_success / self.cycles_total, 4) if self.cycles_total else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "action_match_rate": self.action_match_rate,
            "tool_select_accuracy": self.tool_select_accuracy,
            "cycle_success_rate": self.cycle_success_rate,
            "action_match": self.action_match,
            "tool_checks": self.tool_checks,
            "tool_hits": self.tool_hits,
            "cycles_total": self.cycles_total,
            "cycles_success": self.cycles_success,
        }


def _load_golden(path: Path) -> list[dict]:
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


def evaluate_recording(recording_path: Path, golden_path: Path) -> EvalReport:
    """对照 golden dataset 评估一次录制。

    匹配规则：按 (actor, action) 键匹配 golden 条目（同 actor+action 的
    golden 条目按出现顺序消费）；没有匹配 golden 的录制条目不参与计分。
    """
    entries = AgentRecorder(recording_path).entries()
    golden = _load_golden(golden_path)
    # 按 (actor, action) 分组，指针消费
    golden_by_key: dict[tuple, list[dict]] = {}
    for g in golden:
        golden_by_key.setdefault((g.get("actor"), g.get("action")), []).append(g)

    report = EvalReport()
    for e in entries:
        key = (e.get("actor"), e.get("action"))
        g_list = golden_by_key.get(key)
        g = g_list.pop(0) if g_list else None
        if g is None:
            continue  # 无 golden 标注的条目不参与计分

        detail = {
            "actor": e.get("actor"), "action": e.get("action"),
            "chosen_action": e.get("chosen_action"),
            "expected_action": g.get("expected_action"),
            "chosen_agent": e.get("chosen_agent"),
            "expected_agent": g.get("expected_agent"),
            "tools_used": e.get("tools_used", []),
            "expected_tools": g.get("expected_tools", []),
        }
        report.total += 1

        # 动作精确匹配（action + agent 双字段）
        action_ok = (e.get("chosen_action") == g.get("expected_action")
                     and (not g.get("expected_agent")
                          or e.get("chosen_agent") == g.get("expected_agent")))
        detail["action_match"] = bool(action_ok)
        if action_ok:
            report.action_match += 1

        # 工具选择正确率（golden 标注的 expected_tools 是否都在录制里用过）
        expected_tools = g.get("expected_tools") or []
        if expected_tools:
            used = set(e.get("tools_used", []))
            for t in expected_tools:
                report.tool_checks += 1
                if t in used:
                    report.tool_hits += 1

        # cycle 成功率（录制条目带 cycle 的按 cycle 去重统计）
        if e.get("result"):
            report.cycles_total += 1
            if e.get("result") not in ("failed", "error"):
                report.cycles_success += 1

        detail["result"] = e.get("result")
        report.details.append(detail)

    return report
