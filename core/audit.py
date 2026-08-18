"""
审计日志系统（append-only，与 ExperimentLedger 设计一致）。

记录 Agent 所有操作：工具调用、LLM 调用、人工指令、审批决策。
在企业安全合规场景下，这是必须的。"谁在什么时候做了什么" 全部可追溯。

面试价值：直接命中企业安全合规需求。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("autoresearcher.audit")


@dataclass
class AuditEntry:
    timestamp: float = field(default_factory=time.time)
    actor: str = ""               # "leader" | "user:HUMAN" | "code_agent" | "idea_agent" | "writing_agent"
    action: str = ""              # "tool:launch_experiment" | "tool:write_file" | "llm:think" | "llm:reflect"
    target: str = ""              # 操作对象（文件路径、命令、模型名）
    detail: dict = field(default_factory=dict)  # {"command": "...", "log_file": "..."}
    result: str = ""              # "success" | "failed" | "blocked"
    cost_estimate: float = 0.0
    session_id: str = ""


class AuditLogger:
    """Append-only 审计日志，存储到 workspace/audit.jsonl。"""

    def __init__(self, workspace: Path, filename: str = "audit.jsonl"):
        self.path = workspace / filename
        self._session_id = time.strftime("%Y%m%d_%H%M%S")

    def record(self, *, actor: str, action: str, target: str = "",
               detail: Optional[dict] = None, result: str = "success",
               cost_estimate: float = 0.0) -> None:
        entry = AuditEntry(
            actor=actor,
            action=action,
            target=target,
            detail=detail or {},
            result=result,
            cost_estimate=cost_estimate,
            session_id=self._session_id,
        )
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning(f"Audit write failed: {exc}")

    def recent(self, n: int = 20) -> list[dict]:
        """返回最近 n 条记录。"""
        if not self.path.exists():
            return []
        entries = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries[-n:]

    def summary(self) -> dict:
        """统计摘要：各 action 次数、失败次数、总花费。"""
        entries = self.recent(1000)
        actions = {}
        failures = 0
        total_cost = 0.0
        for e in entries:
            action = e.get("action", "unknown")
            actions[action] = actions.get(action, 0) + 1
            if e.get("result") in ("failed", "blocked"):
                failures += 1
            total_cost += float(e.get("cost_estimate", 0))
        return {
            "total_entries": len(entries),
            "by_action": actions,
            "failures": failures,
            "total_cost_estimate": round(total_cost, 4),
        }
