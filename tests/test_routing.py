"""路由语义回归测试：wait（永久停止）vs retry（临时重试）。

锁定的语义（修复背景）：
- 解析失败/LLM 故障 → action="retry" → 下轮重试，连续达上限才 finish
- Leader 主动 wait（目标达成/等人类）→ finish（永久停止）
- 旧实现把解析失败映射到 wait → 一次格式错误停摆整个 agent
"""
import json

from core.nodes import _parse_json_response, ResearchGraph


def _mk_graph(retry_limit: int = 3):
    """最小 ResearchGraph 实例（仅用于 _deterministic_next 的路由测试）。"""
    g = object.__new__(ResearchGraph)
    g._retry_streak = 0
    g._retry_limit = retry_limit
    return g


class TestParseFallbackIsRetry:
    def test_prose_fallback_is_retry(self):
        """散文 → retry（不执行，也不永久停止）。"""
        r = _parse_json_response("好的，我将开始实验，先写一个 CNN")
        assert r.get("action") == "retry", r

    def test_wait_word_not_misparsed(self):
        """含 wait 单词的非 JSON → retry（不是 wait）。"""
        r = _parse_json_response("we should not wait for this")
        assert r.get("action") == "retry"

    def test_valid_json_unchanged(self):
        r = _parse_json_response('{"action": "experiment", "agent": "code", "task": "t"}')
        assert r["action"] == "experiment"


class TestRetryRouting:
    def test_retry_returns_think(self):
        g = _mk_graph()
        state = {"think_result": json.dumps({"action": "retry", "reason": "x"}),
                 "execute_result": "", "reflect_result": ""}
        assert g._deterministic_next(state) == "think"

    def test_retry_streak_reaches_limit_finish(self):
        g = _mk_graph(retry_limit=2)
        state = {"think_result": json.dumps({"action": "retry"}),
                 "execute_result": "", "reflect_result": ""}
        assert g._deterministic_next(state) == "think"   # 第 1 次 retry
        assert g._deterministic_next(state) == "finish"  # 第 2 次达上限
        assert g._retry_streak == 2

    def test_wait_still_finishes(self):
        """Leader 主动 wait（目标达成）→ finish 语义保留。"""
        g = _mk_graph()
        state = {"think_result": json.dumps({"action": "wait", "reason": "goal reached"}),
                 "execute_result": "", "reflect_result": ""}
        assert g._deterministic_next(state) == "finish"

    def test_normal_decision_resets_streak(self):
        g = _mk_graph()
        g._retry_streak = 2  # 之前连续失败
        state = {"think_result": json.dumps({"action": "experiment", "task": "t"}),
                 "execute_result": "", "reflect_result": ""}
        assert g._deterministic_next(state) == "execute"
        assert g._retry_streak == 0  # 正常决策重置


def _mk_supervisor_graph():
    """最小 ResearchGraph 实例（supervisor_node 路由测试，无 LLM 副作用）。"""
    g = object.__new__(ResearchGraph)
    g._retry_streak = 0
    g._retry_limit = 3
    g.max_cycles = -1  # 关闭轮数闸门，避免干扰路由断言
    g._emit_event = lambda *a, **k: None
    g._update_state = lambda *a, **k: None
    return g


class TestMonitorExecuteRetryLoopRegression:
    """monitor→execute 死循环回归（2026-08-13 真实事故）。

    事故现场：monitor 检测到训练崩溃后，上一轮残留的 reflect_result 让
    _deterministic_next 跳过 reflect 说「think」，旧 think 的
    next_stage="execute" 又把规则覆盖回 execute —— 形成 monitor→execute
    无限重试（跳过 reflect/think，HUMAN_DIRECTIVE 永远读不到）。

    双层修复：
      A. think_node 返回时清空 execute_result/reflect_result（防泄漏）
      B. next_stage 覆盖加守卫：execute_result 非空时不覆盖（防复发）
    本组测试直接钉死 B（事故现场的状态形态）。
    """

    def test_stale_reflect_does_not_override_to_execute(self):
        """事故现场：monitor 结束后带残留 reflect_result → 必须回 think，不是 execute。"""
        g = _mk_supervisor_graph()
        state = {
            "think_result": json.dumps({"action": "experiment", "agent": "code",
                                        "task": "t", "next_stage": "execute"}),
            "execute_result": json.dumps({"experiment_launched": True,
                                          "training_logs": "ModuleNotFoundError",
                                          "experiment_status": "failed"}),
            "reflect_result": json.dumps({"decision": "stale from last cycle"}),
        }
        out = g.supervisor_node(state)
        # 修复前此处返回 "execute"（死循环）；修复后必须回 think（读指令、重新决策）
        assert out["next_agent"] == "think", out

    def test_fresh_cycle_next_stage_still_works(self):
        """正常新轮：think 决策完、execute 为空 → 仍按 next_stage 去 execute。"""
        g = _mk_supervisor_graph()
        state = {
            "think_result": json.dumps({"action": "experiment", "agent": "code",
                                        "task": "t", "next_stage": "execute"}),
            "execute_result": "",
            "reflect_result": "",
        }
        out = g.supervisor_node(state)
        assert out["next_agent"] == "execute"

    def test_monitored_result_with_clean_reflect_goes_reflect(self):
        """monitor 正常结束后 reflect 为空 → 必须走 reflect（不是 think/execute）。"""
        g = _mk_supervisor_graph()
        state = {
            "think_result": json.dumps({"action": "experiment", "agent": "code",
                                        "task": "t", "next_stage": "execute"}),
            "execute_result": json.dumps({"experiment_launched": True,
                                          "training_logs": "Training finished",
                                          "experiment_status": "completed"}),
            "reflect_result": "",
        }
        out = g.supervisor_node(state)
        assert out["next_agent"] == "reflect", out
