"""Agent Eval 框架单元测试：录制 → 回放 → 报告。"""
import json
from pathlib import Path

from core.eval import (
    AgentRecorder,
    AgentReplayer,
    EvalReport,
    evaluate_recording,
)

GOLDEN = Path(__file__).parent / "fixtures" / "golden_actions.jsonl"


# ── AgentRecorder ──

def test_recorder_append_and_read(tmp_path: Path):
    r = AgentRecorder(tmp_path / "rec.jsonl")
    r.record_llm("leader", "think", "prompt...", "output...",
                 chosen_action="experiment", chosen_agent="code", cycle=1)
    r.record_worker("code", "task...", [{"name": "run_shell"}, {"name": "launch_experiment"}],
                    {"response": "done", "experiment_launched": True}, cycle=1)
    entries = r.entries()
    assert len(entries) == 2
    assert entries[0]["actor"] == "leader"
    assert entries[0]["chosen_action"] == "experiment"
    assert entries[1]["tools_used"] == ["run_shell", "launch_experiment"]


def test_recorder_corrupt_line_skipped(tmp_path: Path):
    p = tmp_path / "rec.jsonl"
    p.write_text('{"actor": "a"}\nBAD_LINE\n{"actor": "b"}\n', encoding="utf-8")
    r = AgentRecorder(p)
    assert [e["actor"] for e in r.entries()] == ["a", "b"]


# ── AgentReplayer ──

def test_replayer_analyze_mode(tmp_path: Path):
    r = AgentRecorder(tmp_path / "rec.jsonl")
    r.record_llm("leader", "think", "p", "o", chosen_action="experiment")
    replayer = AgentReplayer(tmp_path / "rec.jsonl")
    rows = replayer.replay()  # llm_fn=None → 纯分析
    assert len(rows) == 1
    assert rows[0]["chosen_action"] == "experiment"
    assert "replayed_output" not in rows[0]


def test_replayer_with_llm_fn(tmp_path: Path):
    r = AgentRecorder(tmp_path / "rec.jsonl")
    r.record_llm("leader", "think", "p", "same_output")
    replayer = AgentReplayer(tmp_path / "rec.jsonl")
    rows = replayer.replay(llm_fn=lambda prompt: "same_output")
    assert rows[0]["replay_match"] is True
    rows2 = replayer.replay(llm_fn=lambda prompt: "different")
    assert rows2[0]["replay_match"] is False


def test_replayer_sample(tmp_path: Path):
    r = AgentRecorder(tmp_path / "rec.jsonl")
    for i in range(5):
        r.record_llm("leader", "think", f"p{i}", "o")
    replayer = AgentReplayer(tmp_path / "rec.jsonl")
    assert len(replayer.replay(sample=2)) == 2


# ── evaluate_recording ──

def _build_recording(tmp_path: Path) -> Path:
    r = AgentRecorder(tmp_path / "rec.jsonl")
    # golden 1: experiment/code
    r.record_llm("leader", "think", "p", "o", chosen_action="experiment",
                 chosen_agent="code", cycle=1)
    # golden 2: experiment/code + launch_experiment（工具命中）
    r.record_llm("leader", "think", "p", "o", chosen_action="experiment",
                 chosen_agent="code", cycle=2)
    r.record_worker("code", "t", [{"name": "launch_experiment"}],
                    {"experiment_launched": True}, cycle=2)
    # golden 3: idea/search_papers（动作错误：选了 code）
    r.record_llm("leader", "think", "p", "o", chosen_action="experiment",
                 chosen_agent="code", cycle=3)
    # golden 5: wait（正确）
    r.record_llm("leader", "reflect", "p", "o", chosen_action="wait", cycle=4)
    # 无 golden 标注的条目不参与计分
    r.record_llm("leader", "think", "p", "o", chosen_action="experiment", cycle=99)
    return tmp_path / "rec.jsonl"


def test_evaluate_recording_metrics(tmp_path: Path):
    rec = _build_recording(tmp_path)
    report = evaluate_recording(rec, GOLDEN)
    assert isinstance(report, EvalReport)
    # 5 条有 golden 的条目参与计分（cycle 99 无 golden 跳过）
    assert report.total == 5
    # 动作匹配：golden1✓ golden2✓ golden3✗ golden5✓ = 3/5（code_agent execute 无记录不参与）
    assert report.action_match == 3
    assert report.action_match_rate == 0.6
    # 工具选择：golden2 think(launch_experiment 未用✗) + golden3 think(search_papers 未用✗)
    #          + golden6 worker(launch_experiment 命中✓) = 3 checks / 1 hit
    assert report.tool_checks == 3
    assert report.tool_hits == 1
    assert report.tool_select_accuracy == round(1 / 3, 4)
    # cycle 成功率：1 条 worker result=completed → 1/1
    assert report.cycles_total == 1
    assert report.cycles_success == 1
    assert report.cycle_success_rate == 1.0


def test_evaluate_empty_recording(tmp_path: Path):
    rec = tmp_path / "empty.jsonl"
    rec.write_text("", encoding="utf-8")
    report = evaluate_recording(rec, GOLDEN)
    assert report.total == 0
    assert report.action_match_rate == 0.0
    assert report.tool_select_accuracy == 0.0
    assert report.cycle_success_rate == 0.0


def test_to_dict_shape(tmp_path: Path):
    rec = _build_recording(tmp_path)
    d = evaluate_recording(rec, GOLDEN).to_dict()
    assert set(d) >= {"total", "action_match_rate", "tool_select_accuracy",
                      "cycle_success_rate"}
