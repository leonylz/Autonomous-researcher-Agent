"""read_file 重复读取去重测试（对齐大厂 Context Editing 思路）。"""
import time
from pathlib import Path

import core.nodes as N


def _reset(workspace: Path):
    N._tool_workspace = workspace
    N._file_read_cache.clear()


def test_repeated_read_dedup(tmp_path: Path):
    _reset(tmp_path)
    p = tmp_path / "train.py"
    p.write_text("line1\nline2\n", encoding="utf-8")
    rf = N.read_file.func

    r1 = rf("train.py")
    assert "line1" in r1 and "line2" in r1
    r2 = rf("train.py")
    assert "already read before" in r2, f"重复读取未去重: {r2[:80]}"


def test_changed_file_re_read(tmp_path: Path):
    _reset(tmp_path)
    p = tmp_path / "train.py"
    p.write_text("line1\n", encoding="utf-8")
    rf = N.read_file.func
    assert "already read before" not in rf("train.py")
    time.sleep(0.01)
    p.write_text("line1\nline2\nline3\n", encoding="utf-8")
    r = rf("train.py")
    assert "already read before" not in r and "line3" in r


def test_line_range_not_deduped(tmp_path: Path):
    _reset(tmp_path)
    p = tmp_path / "train.py"
    p.write_text("line1\nline2\nline3\n", encoding="utf-8")
    rf = N.read_file.func
    rf("train.py")  # 全量读一次
    r = rf("train.py", start_line=1, end_line=1)  # 行范围读
    assert "already read before" not in r
    assert "line1" in r


def test_missing_file(tmp_path: Path):
    _reset(tmp_path)
    r = N.read_file.func("nope.py")
    assert "file not found" in r


def test_max_lines_limit(tmp_path: Path):
    """对齐 Claude Code MAX_LINES_TO_READ=2000：超大文件不全量返回。"""
    _reset(tmp_path)
    p = tmp_path / "big.py"
    p.write_text("\n".join(f"line{i}" for i in range(3000)), encoding="utf-8")
    r = N.read_file.func("big.py")
    assert "only the first 2000" in r
    assert "line2999" not in r  # 尾部未返回
    # 局部精读不受限
    r2 = N.read_file.func("big.py", start_line=2500, end_line=2505)
    assert "line2500" in r2 and "仅显示" not in r2


def test_big_file_not_deduped_when_not_fully_delivered(tmp_path: Path):
    """冒烟实测修复:去重只在「完整交付」时生效。超过回灌阈值的文件,
    agent 实际只看到截断内容,不应提示"不要重复读取"(那会逼它用
    shell 转储)。"""
    _reset(tmp_path)
    p = tmp_path / "train.py"
    # 20K 字符,≤2000 行:read_file 全量返回,但回灌阈值(16K)会截断
    p.write_text(("x" * 80 + "\n") * 250, encoding="utf-8")
    rf = N.read_file.func
    r1 = rf("train.py")
    assert "already read before" not in r1
    # 上次未完整交付 → 再次整文件读取不触发去重
    r2 = rf("train.py")
    assert "already read before" not in r2, "未完整交付的文件不应去重"
    assert "truncated" not in r2  # read_file 自身未截断(≤2000 行)


def test_truncated_head_read_not_deduped(tmp_path: Path):
    """超过 MAX_LINES 的文件:只返回头部 → 不完整交付 → 不触发去重。"""
    _reset(tmp_path)
    p = tmp_path / "big.py"
    p.write_text("\n".join(f"line{i}" for i in range(3000)), encoding="utf-8")
    rf = N.read_file.func
    rf("big.py")  # 头部截断读取
    r2 = rf("big.py")
    assert "already read before" not in r2, "截断读取后不应去重"
