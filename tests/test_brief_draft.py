"""A5/F8 测试:--goal 自动生成 brief;缺失时的友好行为。"""
import tempfile
import unittest
from pathlib import Path

from core.brief_draft import (
    ensure_brief,
    generate_brief,
    parse_draft_json,
    render_brief,
)

GOOD_DRAFT = (
    '```json\n'
    '{"goal": "把 MNIST 训练到 99% 测试准确率",\n'
    ' "success_criteria": "test_acc >= 0.99",\n'
    ' "constraints": "PyTorch, CPU, max 10 epochs",\n'
    ' "what_to_try": "if acc < 0.97: 增大模型容量\\nif acc 0.97-0.985: 加 dropout"}\n'
    '```'
)


class BriefDraftTests(unittest.TestCase):
    def test_parse_tolerates_code_fence(self):
        draft = parse_draft_json(GOOD_DRAFT)
        self.assertEqual(draft["goal"], "把 MNIST 训练到 99% 测试准确率")
        self.assertIn("0.97", draft["what_to_try"])

    def test_parse_accepts_what_to_try_as_array(self):
        """真实 LLM(deepseek)返回 what_to_try 为数组 —— 必须兼容(真实冒烟 bug #2)。"""
        draft = parse_draft_json(
            '{"goal": "g", "success_criteria": "s", "constraints": "c",'
            ' "what_to_try": ["if acc < 0.97: A", "if acc > 0.97: B"]}')
        self.assertIn("if acc < 0.97: A", draft["what_to_try"])
        self.assertIn("B", draft["what_to_try"])

    def test_generate_brief_with_default_llm_none(self):
        """llm_call=None 时用默认 call_draft_llm(真实冒烟 bug #1)。"""
        with unittest.mock.patch(
                "core.brief_draft.call_draft_llm", return_value=GOOD_DRAFT) as mock:
            content = generate_brief("MNIST 99%", Path("."), llm_call=None)
        self.assertIsNotNone(content)
        mock.assert_called_once_with("MNIST 99%")

    def test_parse_rejects_missing_field(self):
        with self.assertRaises(ValueError):
            parse_draft_json('{"goal": "x", "success_criteria": "y", "constraints": "z"}')

    def test_render_brief_has_all_sections(self):
        draft = parse_draft_json(GOOD_DRAFT)
        brief = render_brief(draft)
        for section in ("## Goal", "## Codebase", "## What to Try",
                        "## Constraints", "## Current Status"):
            self.assertIn(section, brief)
        self.assertIn("- if acc < 0.97: 增大模型容量", brief)

    def test_generate_brief_with_fake_llm(self):
        content = generate_brief("MNIST 99%", Path("."),
                                 llm_call=lambda goal: GOOD_DRAFT)
        self.assertIsNotNone(content)
        self.assertIn("## What to Try", content)

    def test_generate_brief_failure_returns_none(self):
        content = generate_brief("x", Path("."),
                                 llm_call=lambda goal: (_ for _ in ()).throw(
                                     RuntimeError("no key")))
        self.assertIsNone(content)


class EnsureBriefTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.project = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_existing_brief_not_overwritten(self):
        (self.project / "PROJECT_BRIEF.md").write_text("user brief", encoding="utf-8")
        created, msg = ensure_brief(self.project, goal="MNIST 99%")
        self.assertFalse(created)
        self.assertIn("已存在", msg)
        self.assertEqual((self.project / "PROJECT_BRIEF.md").read_text(), "user brief")

    def test_missing_brief_with_goal_generates(self):
        created, msg = ensure_brief(
            self.project, goal="MNIST 99%",
            llm_call=lambda goal: GOOD_DRAFT)
        self.assertTrue(created)
        self.assertTrue((self.project / "PROJECT_BRIEF.md").exists())
        content = (self.project / "PROJECT_BRIEF.md").read_text(encoding="utf-8")
        self.assertIn("## What to Try", content)

    def test_missing_brief_and_no_goal_returns_error(self):
        created, msg = ensure_brief(self.project)
        self.assertFalse(created)
        self.assertIn("--goal", msg)

    def test_missing_brief_generation_failure_reports(self):
        created, msg = ensure_brief(
            self.project, goal="x",
            llm_call=lambda goal: (_ for _ in ()).throw(RuntimeError("boom")))
        self.assertFalse(created)
        self.assertIn("手动编写", msg)
        self.assertFalse((self.project / "PROJECT_BRIEF.md").exists())


if __name__ == "__main__":
    unittest.main()
