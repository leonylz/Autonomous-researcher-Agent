"""
Brief 自动生成 — 把用户"一句话目标"展开为可执行的 PROJECT_BRIEF.md。

用户最小输入路径:
    python -m core.nodes --project <dir> --goal "把 MNIST 训练到 99%"
    → brief 缺失 + 有 --goal → 本模块用 LLM 生成四段式 brief
      (Goal / Codebase / What to Try 决策树 / Constraints + Current Status)
    → 写 PROJECT_BRIEF.md → agent 开始自主循环

与 dashboard 的 draft 逻辑同源但独立实现(本模块零 dashboard 依赖):
- What to Try 决策树是质量关键:brief 有决策树 → agent 走"决策树纪律";
  没有 → A4 硬路由强制 idea 先调研。生成时必须有决策树段。
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("autoresearcher.brief_draft")

DRAFT_SYSTEM = (
    "You are a research experiment design assistant. Expand the user's "
    "one-sentence research goal into an executable experiment plan draft. "
    "Output JSON ONLY, no other text. The JSON MUST contain these fields:\n"
    '{"goal": "research goal (one sentence, specific about dataset and metric)",\n'
    ' "success_criteria": "success criteria (with concrete numbers, e.g. test_acc >= 0.99)",\n'
    ' "constraints": "constraints (framework/resources/duration, e.g. PyTorch, CPU, max 10 epochs)",\n'
    ' "what_to_try": "decision tree: 3-5 branches by current result, one per line '
    '(e.g. \'if acc < 0.97: increase model capacity\'), thresholds must be concrete '
    'and verifiable against the goal. The LAST branch MUST be the fallback: '
    '\'if the decision-tree branches above are exhausted or gains are diminishing: '
    'research innovative methods (papers/new ideas) via the idea agent first — do '
    'not blindly keep tuning\'"}\n'
    "Decision-tree branches MUST use explicit if ... : ... conditions with concrete thresholds."
)

DRAFT_MODEL = os.environ.get("DRAFT_MODEL", "deepseek-chat")


def draft_llm_settings() -> tuple[str, str, str]:
    """解析 (base_url, model, api_key) —— 与 dashboard 的 _draft_llm_settings 同规则。"""
    provider = os.environ.get("DRAFT_PROVIDER", "deepseek")
    base_url = os.environ.get("DRAFT_BASE_URL", "")
    key_env = os.environ.get("DRAFT_KEY_ENV", "")
    model = os.environ.get("DRAFT_MODEL", DRAFT_MODEL)
    if not base_url:
        presets = {
            "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
            "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
            "kimi": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
            "glm": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPUAI_API_KEY"),
        }
        base_url, key_env = presets.get(provider, presets["deepseek"])
    api_key = os.environ.get(key_env, "")
    return base_url, model, api_key


def call_draft_llm(goal: str) -> str:
    """一次 OpenAI 兼容调用,返回 LLM 原文(失败抛异常,由调用方降级)。"""
    import openai

    base_url, model, api_key = draft_llm_settings()
    if not api_key:
        raise RuntimeError(
            f"未配置 LLM API Key(需要 {os.environ.get('DRAFT_KEY_ENV', 'DEEPSEEK_API_KEY')} "
            f"或对应环境变量),无法自动生成 brief —— 请手动编写 PROJECT_BRIEF.md"
        )
    client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=60)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": DRAFT_SYSTEM},
                  {"role": "user", "content": f"一句话目标：{goal}"}],
        temperature=0.3,
        max_tokens=2000,
    )
    return (resp.choices[0].message.content or "").strip()


def parse_draft_json(text: str) -> dict:
    """解析 LLM 返回的草案 JSON(容忍 ``` 围栏;缺字段 → ValueError)。

    what_to_try 兼容两种形态:字符串(换行/分号分隔)或数组(逐条分支)。
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("LLM returned non-JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("LLM returned non-object JSON")
    for key in ("goal", "success_criteria", "constraints"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"LLM JSON missing/invalid field: {key}")
    wt = data.get("what_to_try")
    if isinstance(wt, list):
        branches = [str(x).strip() for x in wt if str(x).strip()]
        if not branches:
            raise ValueError("LLM JSON missing/invalid field: what_to_try")
        data["what_to_try"] = "\n".join(branches)
    elif not isinstance(wt, str) or not wt.strip():
        raise ValueError("LLM JSON missing/invalid field: what_to_try")
    else:
        data["what_to_try"] = wt.strip()
    return {k: data[k] for k in
            ("goal", "success_criteria", "constraints", "what_to_try")}


def render_brief(draft: dict) -> str:
    """把解析后的草案渲染为 PROJECT_BRIEF.md(四段式 + Current Status)。"""
    what_to_try = draft["what_to_try"]
    # 兼容换行分隔与逗号分隔的决策树
    if "\n" not in what_to_try and ";" in what_to_try:
        branches = [b.strip() for b in what_to_try.split(";") if b.strip()]
    else:
        branches = [l.strip() for l in what_to_try.splitlines() if l.strip()]
    # 兜底分支(用户审查):决策树必须含"穷尽/收益递减 → 找创新方法"的出口,
    # 而不是永远调参。LLM 未给出时自动追加(幂等:已有则不再加)。
    _FALLBACK_BRANCH = (
        "if the decision-tree branches above are exhausted or gains are "
        "diminishing: research innovative methods (papers/new ideas) via the "
        "idea agent FIRST — do not blindly keep tuning")
    if branches and not any("idea" in b.lower() or "创新" in b
                            for b in branches):
        branches = branches + [_FALLBACK_BRANCH]
    branch_lines = "\n".join(f"- {b}" for b in branches[:7]) or f"- {_FALLBACK_BRANCH}"
    return (
        f"# {draft['goal']}\n\n"
        f"## Goal\n{draft['goal']}\n\n"
        f"## Codebase\n"
        f"- Training script: `train.py` (the agent creates it from `core/train_template.py`)\n"
        f"- Checkpoints: `./checkpoints/` (`best_model.pth` kept)\n\n"
        f"## What to Try\n{branch_lines}\n\n"
        f"## Constraints\n{draft['constraints']}\n\n"
        f"## Current Status\n"
        f"- No experiments yet. Start from a baseline.\n"
    )


def generate_brief(goal: str, project_dir: Path,
                   llm_call=None) -> Optional[str]:
    """一句话目标 → PROJECT_BRIEF.md 内容。

    llm_call 可注入(测试用);None 时用默认 call_draft_llm。
    失败返回 None(调用方降级提示手动编写)。
    """
    if llm_call is None:
        llm_call = call_draft_llm
    try:
        raw = llm_call(goal)
        draft = parse_draft_json(raw)
        return render_brief(draft)
    except Exception as exc:
        logger.warning("brief draft failed: %s", exc)
        return None


def ensure_brief(project_dir: Path, goal: str = "",
                 llm_call=None) -> tuple[bool, str]:
    """确保项目有 PROJECT_BRIEF.md。

    返回 (created, message):brief 已存在 → (False, "已有 brief,忽略 --goal");
    brief 缺失 + goal → 自动生成;都缺 → (False, 报错提示)。
    llm_call 可注入(测试用)。
    """
    brief_path = project_dir / "PROJECT_BRIEF.md"
    if brief_path.exists():
        return False, f"PROJECT_BRIEF.md 已存在,忽略 --goal"
    if not goal.strip():
        return False, (
            "缺少 PROJECT_BRIEF.md 且未提供 --goal。"
            "请编写 brief 或使用 --goal \"一句话目标\" 自动生成。"
        )
    content = generate_brief(goal, project_dir, llm_call=llm_call)
    if content is None:
        return False, "brief 自动生成失败(LLM 调用/解析出错)—— 请手动编写 PROJECT_BRIEF.md"
    brief_path.write_text(content, encoding="utf-8")
    return True, f"已自动生成 PROJECT_BRIEF.md(基于 --goal: {goal[:60]})"
