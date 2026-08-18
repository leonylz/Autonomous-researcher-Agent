"""
Bad Case 采集和闭环分析。

每次安全拦截或异常输出都记录到 bad_cases.jsonl，
定期人工抽样复盘 → 更新规则 → 重新部署。

闭环：
  collect → review → classify → update_rules → deploy → collect

面试价值：ML 系统的 Bad Case 闭环是 MLOps 的核心实践。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Optional

logger = logging.getLogger("autoresearcher.bad_cases")


class BadCaseCollector:
    """Bad Case 采集器。"""

    def __init__(self, log_dir: Path, filename: str = "bad_cases.jsonl"):
        self.path = log_dir / filename

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------

    def record(self, *,
               stage: str,
               rule: str,
               content_snippet: str,
               model: str = "",
               action: str = "blocked",
               metadata: Optional[dict] = None,
               ) -> str:
        """记录一次拦截/异常。

        Parameters
        ----------
        stage : str
            "input" | "output" | "stream" | "tool_call"
        rule : str
            触发的规则名（如 "api_key_leak", "injection_pattern"）。
        content_snippet : str
            触发内容（自动截断到 300 字符，保护隐私）。
        model : str
            当前使用的模型名（方便分析模型差异）。
        action : str
            "blocked" | "flagged" | "passed" | "degraded"
        metadata : dict or None
            额外上下文（cycle, project_name 等）。

        Returns
        -------
        case_id : str
            8 字符 case ID，方便引用。
        """
        case_id = str(uuid.uuid4())[:8]
        entry = {
            "id": case_id,
            "ts": time.time(),
            "stage": stage,
            "rule": rule,
            "model": model,
            "content_snippet": content_snippet[:300],
            "action": action,
            "metadata": metadata or {},
            "reviewed": False,
            "verdict": "",          # 人工填: true_positive / false_positive / needs_new_rule
            "reviewer_notes": "",   # 人工填
        }

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Failed to write bad case: %s", exc)

        return case_id

    def record_tool_schema_error(self, tool_name: str, errors: list[str],
                                  raw_args: dict, model: str = "") -> str:
        """记录一次 tool schema 校验失败（方便追踪 LLM 的常见错误模式）。"""
        return self.record(
            stage="tool_call",
            rule=f"schema_validation:{tool_name}",
            content_snippet=json.dumps({"errors": errors, "raw_args": raw_args}, ensure_ascii=False),
            model=model,
            action="blocked",
            metadata={"tool_name": tool_name, "error_count": len(errors)},
        )

    def record_stream_block(self, rule: str, content_snippet: str, model: str = "") -> str:
        """记录一次流式输出阻断。"""
        return self.record(
            stage="stream",
            rule=rule,
            content_snippet=content_snippet,
            model=model,
            action="blocked",
        )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def stats(self, days: int = 7) -> dict:
        """最近 N 天的拦截统计。"""
        cutoff = time.time() - days * 86400
        total = 0
        blocked = 0
        by_stage = Counter()
        by_rule = Counter()
        by_model = Counter()
        reviewed_count = 0

        if not self.path.exists():
            return {
                "total_events": 0, "blocked": 0, "by_stage": {}, "by_rule": {},
                "by_model": {}, "reviewed": 0, "days": days,
            }

        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if e.get("ts", 0) < cutoff:
                    continue

                total += 1
                if e.get("action") == "blocked":
                    blocked += 1
                by_stage[e.get("stage", "unknown")] += 1
                by_rule[e.get("rule", "unknown")] += 1
                by_model[e.get("model", "unknown")] += 1
                if e.get("reviewed"):
                    reviewed_count += 1

        return {
            "days": days,
            "total_events": total,
            "blocked": blocked,
            "block_rate": round(blocked / max(total, 1), 4),
            "by_stage": dict(by_stage.most_common(10)),
            "by_rule": dict(by_rule.most_common(15)),
            "by_model": dict(by_model.most_common(10)),
            "reviewed": reviewed_count,
        }

    def get_unreviewed(self, limit: int = 50) -> list[dict]:
        """获取未人工审核的 Bad Case。"""
        cases: list[dict] = []
        if not self.path.exists():
            return cases

        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not e.get("reviewed"):
                    cases.append(e)

        return sorted(cases, key=lambda c: c.get("ts", 0), reverse=True)[:limit]

    def mark_reviewed(self, case_id: str, verdict: str, notes: str = "") -> bool:
        """标记一个 Bad Case 为已审核。

        Parameters
        ----------
        case_id : str
            8 字符 case ID。
        verdict : str
            "true_positive" | "false_positive" | "needs_new_rule"
        notes : str
            审核备注。
        """
        if not self.path.exists():
            return False

        updated = False
        lines = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    lines.append(line)
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    lines.append(line)
                    continue

                if e.get("id") == case_id:
                    e["reviewed"] = True
                    e["verdict"] = verdict
                    e["reviewer_notes"] = notes
                    updated = True
                lines.append(json.dumps(e, ensure_ascii=False))

        if updated:
            self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            logger.info("Marked case %s as reviewed: %s", case_id, verdict)

        return updated

    def generate_false_positive_report(self) -> list[dict]:
        """提取所有误拦案例，用于更新规则。"""
        fps: list[dict] = []
        if not self.path.exists():
            return fps

        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("verdict") == "false_positive":
                    fps.append(e)

        return sorted(fps, key=lambda c: c.get("ts", 0), reverse=True)
