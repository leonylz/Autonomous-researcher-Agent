"""
Append-only 事件日志（跨进程观测通道）。

设计动机（对齐 Anthropic 事件协议思想）：
  - Agent 与 Dashboard 是独立进程，进程内事件总线无效；
    事件日志是唯一的跨进程实时通道。
  - append-only JSONL：单行追加天然原子（<PIPE_BUF 时 POSIX 原子，
    Windows 下受文件缓冲保护），崩溃不损坏已有事件。
  - 每条事件带单调递增 seq + run_id + phase + type + timestamp，
    Dashboard 可断点续读（Last-Event-ID 语义）。

事件类型（type）约定：
  - node_start / node_end      — 节点生命周期（think/execute/monitor/reflect）
  - monitor_progress           — 训练进度（epoch/loss）
  - token_usage                — LLM token 统计
  - stream_text_delta          — worker 流式文本增量（可选）
  - tool_call / tool_result    — 工具生命周期
  - cycle_end                  — 一轮实验结束

用法（agent 侧单写者）:
    from .event_journal import EventJournal
    journal = EventJournal(workspace / "events.jsonl")
    journal.start(run_id="proj_a")       # 幂等：同 run 续写
    journal.emit("node_start", phase="think", payload={...})
    seq = journal.last_seq()             # 供断点续读

用法（dashboard 侧只读）:
    journal = EventJournal(workspace / "events.jsonl")
    for event in journal.read_from(after_seq=123, limit=200):
        ...
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger("autoresearcher.event_journal")

# 单次追加的最大行长度（超长 payload 截断，防日志膨胀拖垮读取端）
_MAX_LINE_CHARS = 16_000


class EventJournal:
    """Append-only 事件日志（线程安全单写者 + 只读尾随）。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._seq = 0
        self._run_id: Optional[str] = None
        # 加载已有 seq（续写不重置）
        self._seq = self._load_last_seq()

    # ── 写入侧 ──

    def start(self, run_id: str) -> int:
        """开始/续写一个 run。返回当前 seq。幂等：同 run 重复调用不重置。"""
        with self._lock:
            if self._run_id != run_id:
                self._run_id = run_id
            return self._seq

    def emit(self, type: str, phase: str = "", payload: Optional[dict] = None,
             run_id: Optional[str] = None) -> int:
        """追加一条事件。返回其 seq。

        单行 JSON 追加：即使进程崩溃，已写行不损坏（append-only 语义）。
        """
        with self._lock:
            self._seq += 1
            event = {
                "seq": self._seq,
                "run_id": run_id or self._run_id or "",
                "phase": phase,
                "type": type,
                "ts": time.time(),
                "payload": payload or {},
            }
            line = json.dumps(event, ensure_ascii=False)
            if len(line) > _MAX_LINE_CHARS:
                event["payload"] = {"_truncated": True, **dict(payload or {})}
                line = json.dumps(event, ensure_ascii=False)
                if len(line) > _MAX_LINE_CHARS:
                    event["payload"] = {"_truncated": True}
                    line = json.dumps(event, ensure_ascii=False)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())  # 崩溃安全：落盘再返回
            except OSError as exc:
                # 事件日志失败不中断 agent 主流程（观测通道降级）
                logger.warning("event journal append failed (seq=%d): %s",
                               self._seq, exc)
            return self._seq

    def last_seq(self) -> int:
        with self._lock:
            return self._seq

    # ── 读取侧 ──

    def read_from(self, after_seq: int = 0, limit: int = 200,
                  types: Optional[set] = None) -> list[dict]:
        """读取 seq > after_seq 的事件（最多 limit 条）。断点续读语义。"""
        if not self.path.exists():
            return []
        out: list[dict] = []
        with open(self.path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 跳过损坏行（崩溃残留），不中断读取
                if event.get("seq", 0) <= after_seq:
                    continue
                if types and event.get("type") not in types:
                    continue
                out.append(event)
                if len(out) >= limit:
                    break
        return out

    def _load_last_seq(self) -> int:
        """启动时扫描一次文件，恢复 seq 计数（O(n)，启动一次性）。"""
        if not self.path.exists():
            return 0
        last = 0
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        last = int(json.loads(line).get("seq", 0))
                    except (json.JSONDecodeError, ValueError):
                        continue
        except OSError:
            pass
        return last

    def stats(self) -> dict:
        return {
            "path": str(self.path),
            "last_seq": self._seq,
            "size_kb": round(self.path.stat().st_size / 1024, 1)
            if self.path.exists() else 0,
        }
