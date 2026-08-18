"""
结构化 Prompt 构建器 + 定量评估。

设计原则：
1. XML 标签隔离 → 减少跨区段 attention 串扰（Lost in the Middle）
2. priority 属性  → 告诉 LLM 哪些信息不可丢失
3. Few-shot 难度排序 → 简单→复杂渐进式示例，降低学习曲线
4. 可评估          → needle recall 定量测量，不做"感觉还行"

面试价值：Prompt 不是"写一段话"，是可测试、可调试、可量化的工程组件。
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from typing import Callable, Optional

logger = logging.getLogger("autoresearcher.prompt")


# ═══════════════════════════════════════════════════════════════════
# PromptBuilder
# ═══════════════════════════════════════════════════════════════════

class PromptBuilder:
    """结构化 Prompt 构建器。

    用法:
        pb = PromptBuilder()
        pb.add_section("identity", "You are a research agent...", priority="critical")
        pb.add_section("rules", "Always dry-run before training...", priority="high")
        pb.add_few_shot(examples, sort_by="difficulty")
        pb.add_section("tools", tool_schemas_text, priority="high", collapse_when=4000)
        prompt = pb.build(max_tokens=3000)
    """

    def __init__(self):
        self._sections: list[dict] = []

    # ------------------------------------------------------------------
    # Builder API
    # ------------------------------------------------------------------

    def add_section(self, tag: str, content: str, *,
                    priority: str = "medium",
                    section_id: str = "",
                    collapse_when: Optional[int] = None,
                    ) -> "PromptBuilder":
        """添加一个 XML 包裹的语义区段。

        Parameters
        ----------
        tag : str
            XML 标签名，如 "identity", "rules", "tools", "few_shot"。
        content : str
            区段内容（纯文本或嵌套 XML）。
        priority : str
            "critical" | "high" | "medium" | "low"
            超长时低优先级区段先被折叠。
        section_id : str
            唯一标识，方便消融实验定位。默认用 tag。
        collapse_when : int or None
            当总 prompt 超过此字符数时折叠该区段。None 表示不折叠。
        """
        self._sections.append({
            "tag": tag,
            "content": content,
            "priority": priority,
            "id": section_id or tag,
            "collapse_when": collapse_when,
        })
        return self

    def add_few_shot(self, examples: list[dict], sort_by: str = "difficulty") -> "PromptBuilder":
        """插入 Few-shot 示例，按难度排列。

        Parameters
        ----------
        examples : list[dict]
            每个元素: {"input": ..., "output": ..., "difficulty": 1-10, "tag": "type_a", "why": "..."}
        sort_by : str
            "difficulty" — 简单→复杂（渐进式）
            "diversity" — 按类型轮询（覆盖更多 case）
        """
        if not examples:
            return self

        if sort_by == "difficulty":
            examples = sorted(examples, key=lambda e: e.get("difficulty", 5))
        elif sort_by == "diversity":
            by_tag = defaultdict(list)
            for e in examples:
                by_tag[e.get("tag", "general")].append(e)
            interleaved: list[dict] = []
            max_len = max(len(v) for v in by_tag.values()) if by_tag else 0
            for i in range(max_len):
                for tag in sorted(by_tag):
                    if i < len(by_tag[tag]):
                        interleaved.append(by_tag[tag][i])
            examples = interleaved

        lines = []
        for i, ex in enumerate(examples):
            lines.append(
                f'<example id="{i + 1}" difficulty="{ex.get("difficulty", "?")}"'
                f' tag="{ex.get("tag", "general")}">'
            )
            lines.append(f"<input>{ex['input']}</input>")
            lines.append(f"<output>{ex['output']}</output>")
            if "why" in ex:
                lines.append(f"<rationale>{ex['why']}</rationale>")
            lines.append("</example>")

        return self.add_section(
            "few_shot_examples",
            "\n".join(lines),
            priority="high",
            section_id="few_shot",
        )

    def build(self, max_tokens: int = 3000, collapse: bool = True) -> str:
        """组装最终 prompt。

        Parameters
        ----------
        max_tokens : int
            目标 token 上限（1 token ≈ 4 chars 粗略估计）。
        collapse : bool
            True 时超长自动折叠低优先级区段。
        """
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sections = sorted(self._sections, key=lambda s: (order.get(s["priority"], 2), s["id"]))

        parts: list[str] = []
        for sec in sections:
            block = (
                f'<{sec["tag"]}'
                f' id="{sec["id"]}"'
                f' priority="{sec["priority"]}"'
                f'>\n{sec["content"]}\n'
                f'</{sec["tag"]}>'
            )
            parts.append(block)

        prompt = "\n\n".join(parts)
        max_chars = max_tokens * 4

        if collapse and len(prompt) > max_chars:
            prompt = self._collapse_low_priority(prompt, sections, max_chars)

        return prompt

    def _collapse_low_priority(self, prompt: str, sections: list[dict], max_chars: int) -> str:
        """折叠低优先级区段直到 prompt 在限制内。"""
        # 按优先级升序折叠：先折叠 low，再 medium
        sorted_secs = sorted(
            sections,
            key=lambda s: ({"critical": 3, "high": 2, "medium": 1, "low": 0}.get(s["priority"], 0)),
        )

        for sec in sorted_secs:
            if len(prompt) <= max_chars:
                break
            if sec["priority"] in ("critical",):
                continue  # 永远不折叠关键区段

            full_block = (
                f'<{sec["tag"]}'
                f' id="{sec["id"]}"'
                f' priority="{sec["priority"]}"'
                f'>\n{sec["content"]}\n'
                f'</{sec["tag"]}>'
            )
            collapsed_block = (
                f'<{sec["tag"]}'
                f' id="{sec["id"]}"'
                f' priority="{sec["priority"]}"'
                f' collapsed="true">'
                f'\n[Section folded to stay within {max_chars} char limit. '
                f'Original content: {len(sec["content"])} chars]'
                f'\n</{sec["tag"]}>'
            )
            prompt = prompt.replace(full_block, collapsed_block, 1)

        return prompt

    # ------------------------------------------------------------------
    # 预置模板
    # ------------------------------------------------------------------

    @classmethod
    def for_supervisor(cls, task: str, tool_list: str) -> "PromptBuilder":
        """Supervisor 节点的标准 prompt 结构。"""
        return (
            cls()
            .add_section(
                "identity",
                "You are an autonomous deep learning research supervisor. "
                "Your job is to decide which worker agent to dispatch next: "
                "think (analyze & plan), execute (code & launch), or reflect (evaluate results).",
                priority="critical",
                section_id="identity",
            )
            .add_section(
                "task",
                f"<goal>{task}</goal>",
                priority="critical",
                section_id="task",
            )
            .add_section(
                "routing_rules",
                "<rule id=\"r1\">New cycle without results → THINK</rule>\n"
                "<rule id=\"r2\">Experiment launched and still running → MONITOR</rule>\n"
                "<rule id=\"r3\">Training completed with results → REFLECT</rule>\n"
                "<rule id=\"r4\">Goal reached → FINALIZE</rule>",
                priority="high",
                section_id="routing_rules",
            )
            .add_section(
                "available_agents",
                tool_list,
                priority="high",
                section_id="agents",
            )
        )

    @classmethod
    def for_worker(cls, role: str, tools: str, few_shot: Optional[list[dict]] = None) -> "PromptBuilder":
        """Worker Agent 的标准 prompt 结构。"""
        pb = (
            cls()
            .add_section(
                "identity",
                f"You are a {role} agent. Execute the assigned task using the tools below.",
                priority="critical",
                section_id="identity",
            )
            .add_section(
                "tools",
                tools,
                priority="high",
                section_id="tools",
            )
            .add_section(
                "protocol",
                "Emit <tool_call> blocks for actions. "
                "When finished, respond in plain text with NO tool_call blocks.",
                priority="high",
                section_id="protocol",
            )
        )
        if few_shot:
            pb.add_few_shot(few_shot, sort_by="difficulty")
        return pb


# ═══════════════════════════════════════════════════════════════════
# Needle-in-Haystack 定量评估
# ═══════════════════════════════════════════════════════════════════

def evaluate_needle_recall(
    llm_call: Callable[[str], str],
    test_cases: list[dict],
    needle_marker: str = "$$NEEDLE$$",
) -> dict:
    """
    针-in-草垛测试。

    在长 prompt 的指定位置插入关键信息，测量 LLM 能否准确提取。

    Parameters
    ----------
    llm_call : callable
        接受 prompt 字符串，返回 response 字符串。
    test_cases : list[dict]
        每个元素:
        {
            "context_len": 2000,       # prompt 总字符数
            "needle_position": 0.5,    # 0.0=开头, 0.5=中间, 1.0=结尾
            "needle": "关键信息文本",
            "question": "根据上文回答...",
            "expected": "期望输出",
        }
    needle_marker : str
        在 prompt 中标记 needle 的分隔符。

    Returns
    -------
    dict
        {
            "total": int,
            "correct": int,
            "accuracy": float,
            "by_position": {"pos_0.0": {"total": int, "correct": int, "recall": float}, ...},
            "by_length": {"len_2k": {...}, "len_4k": {...}, ...},
            "elapsed_s": float,
        }
    """
    results = {
        "total": 0,
        "correct": 0,
        "by_position": {},
        "by_length": {},
        "elapsed_s": 0.0,
        "per_case": [],
    }
    filler = "The quick brown fox jumps over the lazy dog. " * 80  # ~4K chars filler
    start_time = time.time()

    for i, case in enumerate(test_cases):
        context_len = case.get("context_len", 2000)
        needle_pos = case.get("needle_position", 0.5)
        needle = case.get("needle", "")
        question = case.get("question", "")
        expected = case.get("expected", "").lower()

        # 构造带 needle 的长 prompt
        filler_chars = filler[: context_len * 2]  # 够用
        insert_at = int(len(filler_chars) * needle_pos)
        context = (
            filler_chars[:insert_at]
            + f"\n{needle_marker} {needle} {needle_marker}\n"
            + filler_chars[insert_at:]
        )
        prompt = f"{context}\n\n---\n\nQuestion: {question}\nAnswer:"

        try:
            response = llm_call(prompt)
            correct = expected in response.lower()
        except Exception as exc:
            logger.warning("LLM call failed for case %d: %s", i, exc)
            response = f"ERROR: {exc}"
            correct = False

        results["total"] += 1
        if correct:
            results["correct"] += 1

        # 按位置分桶
        pos_bucket = f"pos_{int(needle_pos * 10) / 10:.1f}"
        if pos_bucket not in results["by_position"]:
            results["by_position"][pos_bucket] = {"total": 0, "correct": 0}
        results["by_position"][pos_bucket]["total"] += 1
        if correct:
            results["by_position"][pos_bucket]["correct"] += 1

        # 按长度分桶
        len_bucket = f"len_{context_len // 1000}k"
        if len_bucket not in results["by_length"]:
            results["by_length"][len_bucket] = {"total": 0, "correct": 0}
        results["by_length"][len_bucket]["total"] += 1
        if correct:
            results["by_length"][len_bucket]["correct"] += 1

        results["per_case"].append({
            "case_id": i,
            "context_len": context_len,
            "needle_position": needle_pos,
            "correct": correct,
            "expected": expected,
            "response_snippet": response[:100],
        })

    results["accuracy"] = round(results["correct"] / max(results["total"], 1), 4)
    for v in results["by_position"].values():
        v["recall"] = round(v["correct"] / max(v["total"], 1), 4)
    for v in results["by_length"].values():
        v["recall"] = round(v["correct"] / max(v["total"], 1), 4)
    results["elapsed_s"] = round(time.time() - start_time, 2)

    return results


def print_needle_report(results: dict) -> str:
    """将 needle recall 结果格式化为可读报告。"""
    lines = [
        "=" * 50,
        "Needle-in-Haystack Evaluation Report",
        "=" * 50,
        "",
        f"Total cases: {results['total']}",
        f"Correct:     {results['correct']}",
        f"Accuracy:    {results['accuracy']:.2%}",
        f"Elapsed:     {results['elapsed_s']}s",
        "",
        "--- By Position ---",
    ]
    for bucket in sorted(results["by_position"].keys()):
        v = results["by_position"][bucket]
        bar = "█" * int(v["recall"] * 20)
        lines.append(f"  {bucket}: {v['recall']:.2%} {bar} ({v['correct']}/{v['total']})")

    lines.append("")
    lines.append("--- By Context Length ---")
    for bucket in sorted(results["by_length"].keys()):
        v = results["by_length"][bucket]
        bar = "█" * int(v["recall"] * 20)
        lines.append(f"  {bucket}: {v['recall']:.2%} {bar} ({v['correct']}/{v['total']})")

    lines.append("")
    return "\n".join(lines)
