"""EventJournal（append-only 跨进程事件日志）单元测试。"""
import json
from pathlib import Path

from core.event_journal import EventJournal


def test_emit_and_read(tmp_path: Path):
    j = EventJournal(tmp_path / "events.jsonl")
    j.start(run_id="proj_a")
    s1 = j.emit("node_start", phase="think", payload={"cycle": 1})
    s2 = j.emit("node_end", phase="think", payload={"cycle": 1})
    assert s1 == 1 and s2 == 2
    events = j.read_from(after_seq=0)
    assert len(events) == 2
    assert events[0]["type"] == "node_start"
    assert events[0]["phase"] == "think"
    assert events[0]["run_id"] == "proj_a"
    assert events[0]["payload"]["cycle"] == 1
    assert events[0]["seq"] < events[1]["seq"]


def test_read_from_after_seq(tmp_path: Path):
    j = EventJournal(tmp_path / "events.jsonl")
    for i in range(5):
        j.emit("evt", payload={"i": i})
    events = j.read_from(after_seq=2)
    assert [e["seq"] for e in events] == [3, 4, 5]
    assert len(j.read_from(after_seq=5)) == 0


def test_limit(tmp_path: Path):
    j = EventJournal(tmp_path / "events.jsonl")
    for i in range(10):
        j.emit("evt")
    assert len(j.read_from(after_seq=0, limit=3)) == 3


def test_type_filter(tmp_path: Path):
    j = EventJournal(tmp_path / "events.jsonl")
    j.emit("node_start", phase="think")
    j.emit("monitor_progress", phase="monitor")
    events = j.read_from(after_seq=0, types={"monitor_progress"})
    assert len(events) == 1 and events[0]["type"] == "monitor_progress"


def test_seq_persists_across_reopen(tmp_path: Path):
    """续写不重置 seq（断点续读的关键）。"""
    j1 = EventJournal(tmp_path / "events.jsonl")
    j1.start(run_id="r1")
    j1.emit("evt")
    j2 = EventJournal(tmp_path / "events.jsonl")  # 重新打开
    j2.start(run_id="r1")
    s = j2.emit("evt")
    assert s == 2  # 从上次的 seq=1 续写


def test_corrupt_line_skipped(tmp_path: Path):
    """崩溃残留的损坏行不中断读取。"""
    p = tmp_path / "events.jsonl"
    p.write_text('{"seq": 1, "type": "a"}\nNOT_JSON_LINE\n{"seq": 2, "type": "b"}\n',
                 encoding="utf-8")
    j = EventJournal(p)
    events = j.read_from(after_seq=0)
    assert [e["seq"] for e in events] == [1, 2]
    assert j.last_seq() == 2


def test_oversized_payload_truncated(tmp_path: Path):
    j = EventJournal(tmp_path / "events.jsonl")
    j.emit("evt", payload={"big": "x" * 100_000})
    events = j.read_from()
    assert len(events) == 1
    assert events[0]["payload"].get("_truncated") is True


def test_monotonic_seq_under_threads(tmp_path: Path):
    """多线程写入 seq 单调不重复。"""
    import threading
    j = EventJournal(tmp_path / "events.jsonl")
    seqs = []

    def writer():
        for _ in range(20):
            seqs.append(j.emit("evt"))

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(seqs) == 80
    assert len(set(seqs)) == 80  # 无重复
    assert j.last_seq() == 80
    events = j.read_from()
    assert [e["seq"] for e in events] == list(range(1, 81))  # 单调


def test_stats(tmp_path: Path):
    j = EventJournal(tmp_path / "events.jsonl")
    j.emit("evt")
    st = j.stats()
    assert st["last_seq"] == 1
    assert st["size_kb"] > 0
