"""
Human-in-the-Loop 审批系统。

高危操作（写文件、启动实验、删除）可配置需要人工审批。
审批流程：Agent 生成 PENDING_APPROVALS.md → 人工写 APPROVE/DENY → Agent 继续。

面试价值："人在回路"是企业 Agent 安全合规的基本要求。
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("autoresearcher.approval")

# 默认风险等级映射
DEFAULT_RISK_RULES = {
    "launch_experiment": "medium",
    "write_file": "low",
    "run_shell": "medium",
    "run_shell:rm": "high",
    "run_shell:pip install": "medium",
}


@dataclass
class ApprovalRequest:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    action: str = ""              # 工具名
    detail: dict = field(default_factory=dict)
    estimated_cost: float = 0.0
    risk_level: str = "low"       # "low" | "medium" | "high"
    status: str = "pending"       # "pending" | "approved" | "denied"
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None


class ApprovalGate:
    """审批门控。

    配置文件 config.yaml：
        approval:
          enabled: true
          threshold_cost: 10.0        # 超 $10 需审批
          require_for_actions:         # 必须审批的操作列表
            - launch_experiment
          auto_approve_below: 1.0      # $1 以下自动通过
          risk_threshold: "medium"     # 低于此等级自动通过
    """

    RISK_ORDER = {"low": 0, "medium": 1, "high": 2}

    def __init__(self, workspace: Path, config: Optional[dict] = None):
        self.workspace = workspace
        cfg = config or {}
        # B0 三段式模式:off(默认,完全自主)/ exception(仅异常拦截,推荐)/
        # all(每步都批,调试用)
        self.mode = cfg.get("mode", "off")
        self.enabled = cfg.get("enabled", False) or self.mode != "off"
        self.threshold_cost = float(cfg.get("threshold_cost", 10.0))
        self.require_for = set(cfg.get("require_for_actions", ["launch_experiment"]))
        self.auto_approve_below = float(cfg.get("auto_approve_below", 1.0))
        self.risk_threshold = cfg.get("risk_threshold", "medium")
        self._pending_path = workspace / "PENDING_APPROVALS.md"
        # 审批结果缓存:同一操作批一次,后续复用(不打断自主节奏)
        self._decision_cache: dict[str, str] = {}

    # ── 缓存(批一次,后续复用)──

    @staticmethod
    def cache_key_for(action: str, detail: Optional[dict] = None) -> str:
        import json as _json
        return f"{action}:{_json.dumps(detail or {}, sort_keys=True)[:200]}"

    def cached_decision(self, key: str) -> Optional[str]:
        return self._decision_cache.get(key)

    def cache_decision(self, key: str, decision: str) -> None:
        self._decision_cache[key] = decision

    def needs_approval(self, action: str, detail: Optional[dict] = None,
                       estimated_cost: float = 0.0) -> tuple[bool, str]:
        """检查是否需要审批。返回 (needs, reason)。"""
        if not self.enabled or self.mode == "off":
            return False, "approval disabled"

        # all 模式:每个工具调用都批(调试/教学用)
        if self.mode == "all":
            return True, f"approval mode 'all' — action '{action}' requires approval"

        # 在必须审批列表里 → 跳过成本检查
        if action in self.require_for:
            return True, f"action '{action}' requires approval"

        # 成本超阈值 → 必须审批
        if estimated_cost >= self.threshold_cost:
            return True, f"cost ${estimated_cost:.2f} >= threshold ${self.threshold_cost:.2f}"

        # 风险等级检查
        risk = self._assess_risk(action, detail or {})
        if self.RISK_ORDER.get(risk, 0) >= self.RISK_ORDER.get(self.risk_threshold, 1):
            return True, f"risk level '{risk}' >= threshold '{self.risk_threshold}'"

        return False, "auto-approved"

    def _assess_risk(self, action: str, detail: dict) -> str:
        # 特殊命令匹配
        if action == "run_shell":
            cmd = str(detail.get("command", "")).lower()
            if any(d in cmd for d in ("rm ", "rmdir", "del ", "sudo", "chmod 777")):
                return "high"
            if "pip install" in cmd or "apt" in cmd:
                return "medium"
        return DEFAULT_RISK_RULES.get(action, "low")

    def create_request(self, action: str, detail: Optional[dict] = None,
                       estimated_cost: float = 0.0) -> ApprovalRequest:
        req = ApprovalRequest(
            action=action,
            detail=detail or {},
            estimated_cost=estimated_cost,
            risk_level=self._assess_risk(action, detail or {}),
            status="pending",
        )
        self._write_pending_file(req)
        return req

    def check_response(self, request_id: str) -> Optional[str]:
        """扫描 PENDING_APPROVALS.md 中的人工回复。返回 'approved' | 'denied' | None（未回复）。"""
        if not self._pending_path.exists():
            return None
        content = self._pending_path.read_text(encoding="utf-8")
        look_for = f"[{request_id}]"
        if look_for not in content:
            return None
        # 找到对应行，检查其后的行直到下一个请求块（或文件尾）。
        # 用块边界而非固定行数：决定可能写在 "Your decision" 提示后若干行
        # （如 UI 在 marker 前插入的 **APPROVE**），固定窗口会漏掉。
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if look_for not in line.lower():
                continue
            for j in range(i + 1, len(lines)):
                if lines[j].strip().startswith("## ["):
                    break  # 进入下一个请求块，停止
                # 精确匹配：行首的独立决定词（容忍 **APPROVE** 加粗、前后空白）。
                # 不用子串/前缀匹配，避免 "Approved by admin" 等评论文本误判。
                m = re.search(
                    r"^\s*\*?\*?(APPROVE|DENY|REJECT)\b",
                    lines[j], re.IGNORECASE)
                if m:
                    word = m.group(1).upper()
                    if word == "APPROVE":
                        return "approved"
                    return "denied"
        return None

    def wait_for_approval(self, request_id: str, poll_interval: int = 30,
                          timeout: int = 3600) -> str:
        """轮询等待人工审批。超时返回 'denied'。"""
        elapsed = 0
        while elapsed < timeout:
            result = self.check_response(request_id)
            if result:
                return result
            logger.info(f"Waiting for approval of [{request_id}]... ({elapsed}s)")
            time.sleep(poll_interval)
            elapsed += poll_interval
        logger.warning(f"Approval timeout for [{request_id}]")
        return "denied"

    def _write_pending_file(self, req: ApprovalRequest):
        marker = "========================================"
        if not self._pending_path.exists():
            self._pending_path.write_text(
                "# Pending Approvals\n\n"
                "Write **APPROVE** or **DENY** below and save; the agent will read your decision.\n\n"
                + marker + "\n\n",
                encoding="utf-8",
            )
        with open(self._pending_path, "a", encoding="utf-8") as f:
            f.write(f"## [{req.id}] {req.action} (risk: {req.risk_level})\n\n")
            f.write(f"- Cost estimate: ${req.estimated_cost:.2f}\n")
            f.write(f"- Detail: {json.dumps(req.detail, ensure_ascii=False)}\n")
            f.write(f"- Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("**Your decision (write APPROVE or DENY below):**\n\n")
            f.write(marker + "\n\n")
