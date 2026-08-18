"""
HypothesisStore — 假设生命周期状态机。

科研循环围绕"假设生命周期"组织的核心:
  提出(proposed)→ 验证中(testing)→ 证实/否证/不确定(confirmed/refuted/inconclusive)

价值:
- **防重复实验**:已否证的假设不再被 think 提出(注入"已否证列表"给决策)
- **防乱想**:think 只能选"待验证"的假设(注入待验证列表)
- **可溯源**:每个结论带依据(实验 id + 证据摘要)

存储:workspace/hypotheses.db(SQLite,与 ledger 互补 —— ledger 记实验,
本模块记假设及其结论)。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger("autoresearcher.hypotheses")

STATUSES = ("proposed", "testing", "confirmed", "refuted", "inconclusive")


class HypothesisStore:
    """假设生命周期存储(proposed → testing → resolved)。"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hypotheses (
                    id           TEXT PRIMARY KEY,
                    text         TEXT NOT NULL,
                    status       TEXT NOT NULL DEFAULT 'proposed',
                    experiment_id TEXT DEFAULT '',
                    evidence     TEXT DEFAULT '',
                    created_at   REAL NOT NULL,
                    updated_at   REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hyp_status "
                "ON hypotheses(status)")

    # ── 写入 ──

    def add(self, text: str) -> str:
        """提出假设。内容去重(相同文本 → 返回已有 id,不重复建)。"""
        text = text.strip()
        existing = self.find_by_text(text)
        if existing:
            return existing["id"]
        hid = str(uuid.uuid4())[:8]
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO hypotheses (id, text, status, created_at, updated_at) "
                "VALUES (?, ?, 'proposed', ?, ?)",
                (hid, text, now, now))
        logger.info("hypothesis proposed: %s — %s", hid, text[:80])
        return hid

    def mark_testing(self, hid: str, experiment_id: str = "") -> None:
        """标记为验证中。已有结论的假设(confirmed/refuted/inconclusive)
        不被无信息量的更新降级 —— 结论一旦成立,只能由新的实验结果
        resolve 覆盖(冒烟实测:报告轮无实验信号,else 分支 mark_testing
        会把上一轮已 confirmed 的假设打回 testing,结论丢失)。"""
        row = self.get(hid)
        if row and row["status"] in ("confirmed", "refuted", "inconclusive"):
            return
        self._update(hid, status="testing", experiment_id=experiment_id)

    def resolve(self, hid: str, outcome: str, experiment_id: str = "",
                evidence: str = "") -> None:
        """按实验结果结算假设。outcome ∈ confirmed/refuted/inconclusive。"""
        if outcome not in ("confirmed", "refuted", "inconclusive"):
            raise ValueError(f"invalid outcome: {outcome}")
        self._update(hid, status=outcome, experiment_id=experiment_id,
                     evidence=evidence)
        logger.info("hypothesis %s → %s (exp=%s)", hid, outcome, experiment_id)

    def _update(self, hid: str, **fields) -> None:
        if not fields:
            return
        fields["updated_at"] = time.time()
        sets = ", ".join(f"{k}=?" for k in fields)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE hypotheses SET {sets} WHERE id=?",
                (*fields.values(), hid))

    # ── 读取 ──

    def find_by_text(self, text: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM hypotheses WHERE text=? ORDER BY created_at DESC LIMIT 1",
                (text.strip(),)).fetchone()
        return dict(row) if row else None

    def get(self, hid: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM hypotheses WHERE id=?", (hid,)).fetchone()
        return dict(row) if row else None

    def pending(self, limit: int = 10) -> list[dict]:
        """待验证假设(proposed/testing),按时间倒序。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM hypotheses WHERE status IN ('proposed','testing') "
                "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def resolved(self, limit: int = 20) -> list[dict]:
        """已结算假设(confirmed/refuted/inconclusive),按更新时间倒序。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM hypotheses WHERE status IN "
                "('confirmed','refuted','inconclusive') "
                "ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def refuted(self, limit: int = 10) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM hypotheses WHERE status='refuted' "
                "ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def all(self, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM hypotheses ORDER BY created_at DESC LIMIT ?",
                (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ── 上下文注入(think 决策输入)──

    def to_context(self, pending_limit: int = 8, refuted_limit: int = 6) -> str:
        """渲染成 think 上下文片段:待验证假设 + 已否证假设。

        决策约束提示:只能选"待验证"中的假设;已否证的绝不重复提出。
        """
        lines = []
        pend = self.pending(pending_limit)
        if pend:
            lines.append("## 待验证假设(下一步决策只能从这里选):")
            for h in pend:
                tag = "验证中" if h["status"] == "testing" else "待验证"
                lines.append(f"- [{tag}] {h['text'][:150]}")
        ref = self.refuted(refuted_limit)
        if ref:
            lines.append("## 已否证假设(禁止再次提出):")
            for h in ref:
                ev = f" — {h['evidence'][:80]}" if h.get("evidence") else ""
                lines.append(f"- ❌ {h['text'][:120]}{ev}")
        return "\n".join(lines)
