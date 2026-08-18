"""EXECUTE → MONITOR 交接回归测试:pid/log_file 必须来自工具结果,而非模型散文。

从已删除的旧引擎测试 (tests/test_tool_use_loop.py) 保留的核心保证:
launch_experiment 是唯一返回 pid+log_file 的工具,worker 的散文可能说谎
(例如 "PID=99999" 只是模型幻觉),但后续监控必须以工具结果为权威。
"""

import json
import unittest
from types import SimpleNamespace

from core.nodes import ResearchGraph


def _msg(content: str):
    return SimpleNamespace(content=content)


def _graph() -> ResearchGraph:
    """_extract_execute_result 是实例方法(不使用 self),用轻量实例调用。"""
    return object.__new__(ResearchGraph)


class ExtractExecuteResultTests(unittest.TestCase):
    def test_pid_from_tool_result_beats_lying_prose(self):
        """ToolMessage 里的 launch JSON 是权威;模型散文里的假 PID 必须被忽略。"""
        agent_result = {
            "agent": "code",
            "messages": [
                _msg(json.dumps({"pid": 4321, "log_file": "/tmp/exp.log",
                                 "status": "launched"})),
            ],
        }
        result = _graph()._extract_execute_result(
            agent_result, "code", "Training started, PID=99999 (this number is wrong)."
        )
        self.assertTrue(result["experiment_launched"])
        self.assertEqual(result["pid"], 4321)
        self.assertEqual(result["log_file"], "/tmp/exp.log")

    def test_pid_extracted_from_nested_json_in_message(self):
        """pid 嵌套在 prose 包裹的 JSON 里也要能提取(模型可能复述工具结果)。"""
        agent_result = {
            "agent": "code",
            "messages": [
                _msg('launched ok: {"pid": 7, "log_file": "logs/a.log"} done'),
            ],
        }
        result = _graph()._extract_execute_result(agent_result, "code", "ok")
        self.assertTrue(result["experiment_launched"])
        self.assertEqual(result["pid"], 7)

    def test_no_pid_means_not_launched(self):
        agent_result = {"agent": "code", "messages": [_msg("I could not launch it")]}
        result = _graph()._extract_execute_result(agent_result, "code", "nope")
        self.assertFalse(result["experiment_launched"])
        self.assertEqual(result["response"], "nope")

    def test_handles_empty_result(self):
        result = _graph()._extract_execute_result(None, "code", "")
        self.assertFalse(result["experiment_launched"])
        self.assertEqual(result["agent"], "code")


if __name__ == "__main__":
    unittest.main()
