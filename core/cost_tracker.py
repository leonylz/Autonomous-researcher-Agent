"""
费用追踪系统。

记录每次 LLM 调用的 token 消耗和费用，支持：
  - 按模型计费（不同模型价格不同）
  - 按天统计（daily_summary）
  - 预算告警（budget_alert）
  - 项目累计（project_total）

存储：workspace/costs.jsonl（append-only）

面试价值：有数字可讲的都是好故事。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("autoresearcher.cost")

# 模型价格表（$/1M tokens — 2026 Q2 参考价）
# 可通过 config.yaml 覆盖
MODEL_PRICES = {
    # Anthropic
    "claude-opus-4-6":     (15.0, 75.0),
    "claude-sonnet-4-6":   (3.0, 15.0),
    "claude-haiku-4-5":    (0.8, 4.0),
    # OpenAI
    "gpt-5.4":             (15.0, 75.0),
    "gpt-5.3":             (3.0, 15.0),
    "codex-5.3":           (3.0, 15.0),
    # Qwen / DashScope
    "qwen-max":            (2.0, 8.0),
    "qwen-plus":           (0.8, 2.0),
    "qwen-turbo":          (0.3, 0.6),
    # DeepSeek
    "deepseek-chat":       (0.27, 1.10),
    "deepseek-reasoner":   (0.55, 2.19),
    # GLM / Zhipu
    "glm-4.5":             (0.1, 0.1),
    "glm-4-flash":         (0.001, 0.001),
    # Moonshot / Kimi
    "moonshot-v1":         (0.6, 1.2),
    # MiniMax
    "minimax-m1":          (0.5, 2.0),
    # Default fallback
    "_default_":           (1.0, 4.0),
}


class CostTracker:
    """Append-only 费用追踪器。"""

    def __init__(self, workspace: Path, config_or_filename: object = None,
                 price_overrides: Optional[dict] = None):
        # Accept either a config dict (from nodes.py) or a filename string
        if isinstance(config_or_filename, dict):
            cfg = config_or_filename
            filename = cfg.get("filename", "costs.jsonl")
            price_overrides = price_overrides or cfg.get("price_overrides", {})
        elif isinstance(config_or_filename, str):
            filename = config_or_filename
        else:
            filename = "costs.jsonl"
        self.path = workspace / filename
        self._prices = {**MODEL_PRICES, **(price_overrides or {})}

    def record_call(self, model: str, input_tokens: int, output_tokens: int,
                    actor: str = "", action: str = "") -> float:
        """记录一次 LLM 调用，返回费用（美元）。"""
        price_in, price_out = self._prices.get(model, self._prices["_default_"])
        cost = (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out

        entry = {
            "ts": time.time(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
            "actor": actor,
            "action": action,
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning(f"Cost write failed: {exc}")
        return cost

    def all_entries(self) -> list[dict]:
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
        return entries

    def project_total(self) -> float:
        return sum(e.get("cost_usd", 0) for e in self.all_entries())

    def daily_summary(self, days: int = 7) -> dict:
        """按天汇总最近 N 天费用。"""
        now = time.time()
        cutoff = now - days * 86400
        entries = [e for e in self.all_entries() if e.get("ts", 0) >= cutoff]

        by_day = {}
        by_model = {}
        total = 0.0
        for e in entries:
            day = time.strftime("%Y-%m-%d", time.localtime(e.get("ts", 0)))
            cost = e.get("cost_usd", 0)
            by_day[day] = by_day.get(day, 0.0) + cost
            model = e.get("model", "unknown")
            by_model[model] = by_model.get(model, 0.0) + cost
            total += cost

        return {
            "days": days,
            "total_calls": len(entries),
            "total_cost_usd": round(total, 4),
            "by_day": {k: round(v, 4) for k, v in sorted(by_day.items())},
            "by_model": {k: round(v, 4) for k, v in sorted(by_model.items())},
        }

    def budget_alert(self, daily_budget: float) -> tuple[bool, str]:
        """检查今日是否超预算。返回 (alert, message)。"""
        today = time.strftime("%Y-%m-%d")
        summary = self.daily_summary(1)
        today_cost = summary["by_day"].get(today, 0.0)
        if today_cost >= daily_budget:
            return True, f"Budget exceeded: ${today_cost:.2f} >= ${daily_budget:.2f}"
        if today_cost >= daily_budget * 0.8:
            return True, f"Budget warning: ${today_cost:.2f} / ${daily_budget:.2f} (80%)"
        return False, f"Budget OK: ${today_cost:.2f} / ${daily_budget:.2f}"
