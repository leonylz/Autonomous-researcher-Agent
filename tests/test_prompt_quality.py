"""Prompt 质量不变量测试:每个 agent 提示词必须结构完整、工具名与注册表一致。"""
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.nodes import (  # noqa: E402
    LEADER_REFLECT_PROMPT,
    LEADER_THINK_PROMPT,
    TOOL_FUNCTIONS,
    _WORKER_PROMPTS,
    _load_worker_prompt,
)

AGENTS_DIR = PROJECT_ROOT / "agents"

# agent 定义文件(排除 README.md 等非 agent 文档);key = 文件名 stem,
# 与 frontmatter 的 name 字段一致
AGENT_FILES = {
    "code_agent": "code_agent.md",
    "idea_agent": "idea_agent.md",
    "leader": "leader.md",
    "leader_think": "leader_think_agent.md",
    "leader_reflect": "leader_reflect_agent.md",
    "writing_agent": "writing_agent.md",
    "review_agent": "review_agent.md",
}

# 每个 agent md 必须包含的结构段(角色/流程/输出契约,按实际标题)
REQUIRED_SECTIONS = {
    "code_agent": ["Role", "Tools Available", "Mandatory Workflow", "Dry-Run"],
    "idea_agent": ["Role & Mission", "Tools Available", "Workflow",
                   "Citation Traceability", "IDEA_NOTES.md format"],
    "leader": ["Your Role", "Decision Framework", "Constraints", "Output Format"],
    "leader_think": ["Role & Mission", "Decision Procedure", "Output Contract",
                     "Examples", "Constraints"],
    "leader_reflect": ["Role & Mission", "Decision Procedure", "Output Contract",
                       "Examples", "Constraints"],
    "writing_agent": ["Role & Mission", "Tools Available", "Output Format"],
    "review_agent": ["Review Checklist", "Output Format"],
}

# 禁止出现的占位符/未完成标记(TODO 除外:它是训练模板的真实术语,
# 指 agent 可编辑区域 —— 真实 few-shot 输出与模板都包含它)
FORBIDDEN_PLACEHOLDERS = ("TBD", "FIXME", "lorem ipsum", "{{placeholder")


def _agent_frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.lstrip().startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    meta = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta


def test_all_agent_files_have_valid_frontmatter():
    for stem, fname in AGENT_FILES.items():
        md = AGENTS_DIR / fname
        meta = _agent_frontmatter(md)
        assert "name" in meta, f"{fname}: frontmatter 缺 name"
        assert "description" in meta, f"{fname}: frontmatter 缺 description"
        assert meta["name"] == stem, (
            f"{fname}: frontmatter name({meta['name']}) 与文件名不一致")


def test_no_placeholders_in_prompts():
    for fname in AGENT_FILES.values():
        text = (AGENTS_DIR / fname).read_text(encoding="utf-8", errors="replace")
        for token in FORBIDDEN_PLACEHOLDERS:
            assert token not in text, f"{fname}: 含占位符 {token}"


def test_required_sections_present():
    for stem, sections in REQUIRED_SECTIONS.items():
        text = (AGENTS_DIR / AGENT_FILES[stem]).read_text(
            encoding="utf-8", errors="replace")
        for section in sections:
            # 标题可以带序号/后缀,如 "### Step 3: Dry-Run (MANDATORY)"
            assert re.search(
                rf"^#{{2,3}} [^#\n]*{re.escape(section)}", text, re.M), (
                f"{AGENT_FILES[stem]}: 缺必需段 '{section}'")


def test_tool_names_in_prompts_match_registry():
    """prompt 里提到的工具名必须存在于**该 agent 自己的** TOOL_FUNCTIONS 注册表。

    (T6 实测修复:idea_agent 的 prompt 提到 list_files/list_tree,但注册表
     没有 → 工具层返回 unknown tool,agent 只能 read_file 猜路径。)
    """
    stem_to_key = {"code_agent": "code", "idea_agent": "idea",
                   "writing_agent": "writing", "review_agent": "review"}
    for stem, fname in AGENT_FILES.items():
        if stem not in stem_to_key:
            continue  # leader 无工具注册表(系统注入)
        registered = {name for name, _ in TOOL_FUNCTIONS.get(stem_to_key[stem], [])}
        text = (AGENTS_DIR / fname).read_text(encoding="utf-8", errors="replace")
        # 提取 "Tools Available" 到下一个 ## 之间的内容
        m = re.search(r"^## Tools Available\n(.*?)(?=^## )", text, re.M | re.S)
        if not m:
            continue  # 无工具列表段的 agent 跳过
        # 只匹配列表项行首的 "- `name`"(描述里的参数名如 start_line 不算)
        mentioned = set(re.findall(r"^[-*] `([a-z_]{3,30})`", m.group(1), re.M))
        for tool in mentioned:
            assert tool in registered, (
                f"{fname}: 工具列表提到 '{tool}',但 {stem_to_key[stem]} 注册表"
                f"没有它(工具层会返回 unknown tool)")


def test_worker_prompts_load_without_fallback():
    """_WORKER_PROMPTS 必须从文件加载成功(不是内联 fallback)。"""
    for agent in ("code", "idea", "writing", "review"):
        prompt = _WORKER_PROMPTS[agent]
        assert prompt and len(prompt) > 100, f"{agent} prompt 加载异常"
        # fallback 是纯内联文本;文件加载版以 "# {Name} Agent" 标题开头
        assert prompt.startswith("#"), f"{agent}: 未从文件加载(可能走了 fallback)"


def test_leader_prompts_load_from_files():
    """leader 两阶段提示词必须从 agents/leader_think|reflect.md 加载(单一事实源)。"""
    assert LEADER_THINK_PROMPT.startswith("#"), "leader_think 未从文件加载"
    assert LEADER_REFLECT_PROMPT.startswith("#"), "leader_reflect 未从文件加载"
    assert "Tune vs. Innovate" in LEADER_THINK_PROMPT
    assert "next: idea_agent" in LEADER_REFLECT_PROMPT or "next:" in LEADER_REFLECT_PROMPT
    assert "## Examples" in LEADER_THINK_PROMPT and "## Examples" in LEADER_REFLECT_PROMPT


# ── 语言统一防回退(用户审查:提示词必须全英文,模型可见消息同)──
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def test_leader_prompts_are_english():
    """leader think/reflect 内联提示词不得含 CJK(语言统一防回退)。"""
    for name, prompt in (("think", LEADER_THINK_PROMPT),
                         ("reflect", LEADER_REFLECT_PROMPT)):
        assert not _CJK_RE.search(prompt), f"LEADER_{name.upper()}_PROMPT 含中文"


def test_agent_prompt_files_are_english():
    """agents/*.md 提示词文件不得含 CJK。"""
    for fname in AGENT_FILES.values():
        text = (AGENTS_DIR / fname).read_text(encoding="utf-8", errors="replace")
        assert not _CJK_RE.search(text), f"{fname} 含中文"


def test_tool_descriptions_are_english():
    """工具描述(模型可见)不得含 CJK。"""
    for tools in TOOL_FUNCTIONS.values():
        for _, fn in tools:
            desc = getattr(fn, "description", "") or ""
            assert not _CJK_RE.search(desc), f"{fn.__name__} description 含中文"
