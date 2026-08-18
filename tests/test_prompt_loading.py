"""Worker 提示词加载测试：agents/*.md 是单一事实源。

验证：
- 文件优先加载（不再是代码内联）
- frontmatter 被剥离（提示词不含 YAML 头）
- 文件缺失时 fallback 到内联
"""
from pathlib import Path

import core.nodes as N
from core.nodes import _WORKER_PROMPTS, _load_worker_prompt


def test_loaded_from_files():
    """正常情况：prompt 来自 agents/*.md（含新增的 launch 交接语义）。"""
    code_prompt = _WORKER_PROMPTS.get("code", "")
    assert "Launch Handoff" in code_prompt  # md 中的新指令
    assert "monitor node's job" in code_prompt
    assert "dry-run" in code_prompt.lower()


def test_frontmatter_stripped():
    """frontmatter（--- name/description/model ---）不进入提示词。"""
    for name in ("code", "idea", "writing", "review"):
        prompt = _WORKER_PROMPTS.get(name, "")
        assert "---" not in prompt.splitlines()[0] if prompt else True, \
            f"{name} prompt 未剥离 frontmatter"
        assert "name:" not in prompt[:50]


def test_fallback_on_missing_file(tmp_path: Path, monkeypatch):
    """文件缺失 → fallback 到内联。"""
    monkeypatch.setattr(N, "_AGENTS_DIR", tmp_path)  # 空目录 → 所有文件缺失
    fallback = "INLINE_FALLBACK_CONTENT"
    loaded = _load_worker_prompt("nonexistent", fallback)
    assert loaded == fallback


def test_code_prompt_has_efficiency_rules():
    """高效执行原则在文件 prompt 中。"""
    prompt = _WORKER_PROMPTS.get("code", "")
    assert "Efficiency Principles" in prompt


def test_idea_writing_prompts_loaded():
    assert "Idea agent" in _WORKER_PROMPTS.get("idea", "")
    assert "Writing agent" in _WORKER_PROMPTS.get("writing", "")
