"""0.5-B crash-resume 语义测试:checkpoint 恢复的合并行为。

审查项 30 实测证实:同 thread 二次 invoke 时,**全量 input 会覆盖 checkpoint
的执行状态**(cycle 回退)。修复:run() 检测到既有 checkpoint 时只传增量
输入(外部信号 directive),保留执行状态 —— 本测试固化该语义。
"""
import gc
import tempfile
import unittest
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from core.rollback import SqliteCheckpointer


class S(TypedDict):
    cycle: int


def _mk_compiled(ckpt):
    g = StateGraph(S)
    g.add_node("bump", lambda s: {"cycle": int(s.get("cycle", 0)) + 1})
    g.add_edge(START, "bump")
    g.add_edge("bump", END)
    return g.compile(checkpointer=ckpt)


class CrashResumeSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.ckpt = SqliteCheckpointer(Path(self.tempdir.name) / "checkpoints.db")
        self.config = {"configurable": {"thread_id": "t1"}}

    def tearDown(self):
        del self.ckpt
        gc.collect()
        self.tempdir.cleanup()

    def test_incremental_input_keeps_checkpoint_state(self):
        """run() 修复后的调用模式:已有 checkpoint → 只传增量(空/部分)→ 状态继续。"""
        compiled = _mk_compiled(self.ckpt)
        compiled.invoke({"cycle": 0}, self.config)  # checkpoint: cycle=1
        r2 = compiled.invoke({}, self.config)       # 增量输入
        self.assertEqual(r2["cycle"], 2)            # 保留 checkpoint,不回退

    def test_full_input_overrides_checkpoint(self):
        """全量 input 会覆盖 checkpoint(回退) —— run() 必须避免走这条路径。"""
        compiled = _mk_compiled(self.ckpt)
        compiled.invoke({"cycle": 0}, self.config)  # checkpoint: cycle=1
        r3 = compiled.invoke({"cycle": 0}, self.config)  # 全量覆盖
        self.assertEqual(r3["cycle"], 1)            # 回退(证明 bug 存在)

    def test_fresh_thread_starts_from_input(self):
        compiled = _mk_compiled(self.ckpt)
        r = compiled.invoke({"cycle": 10}, {"configurable": {"thread_id": "t2"}})
        self.assertEqual(r["cycle"], 11)

    def test_checkpoint_detection_for_resume_decision(self):
        """run() 用 get_tuple 判断是否走增量恢复。"""
        compiled = _mk_compiled(self.ckpt)
        self.assertIsNone(self.ckpt.get_tuple(self.config))
        compiled.invoke({"cycle": 0}, self.config)
        self.assertIsNotNone(self.ckpt.get_tuple(self.config))


if __name__ == "__main__":
    unittest.main()
