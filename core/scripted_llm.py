"""
ScriptedLLM — 确定性 LLM 替身,让完整 ResearchGraph 循环无需 API key 即可离线回归。

用途:
- eval-first 架构的基石:CI / 本地无 key 时,用 ScriptedLLM 驱动完整
  think → supervise → finish 状态流转,验证事件、账本、checkpoint、锁释放。
- 面试可讲:"整个 agent 循环可以在零成本、零网络下做确定性回归,
  真实 LLM 只负责质量,不负责正确性。"

设计:
- 实现 LangChain 模型的最小接口(invoke / model_name / response_metadata),
  不依赖 langchain 类型,保持零依赖。
- scenario 决定 think 阶段的返回,其他阶段返回降级 JSON(不会被用到)。
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Optional


class ScriptedLLM:
    """按 scenario 返回预设 JSON 的确定性 LLM。"""

    def __init__(self, scenario: str = "wait", model_name: str = "scripted"):
        self.scenario = scenario
        self.model_name = model_name
        self.calls: list[dict] = []  # 调用记录,测试可断言

    # ── LangChain 最小接口 ──

    def invoke(self, messages, **kwargs):
        """返回带 .content 的响应对象(SimpleNamespace 即可,不需要真实 AIMessage)。"""
        self.calls.append({
            "messages": list(messages),
            "kwargs": kwargs,
        })
        content = self._respond(messages)
        return SimpleNamespace(
            content=content,
            response_metadata={},  # token_usage 缺失 → _safe_llm_call 拿到空 dict
        )

    # ── scenario 逻辑 ──

    def _respond(self, messages) -> str:
        # 通过 system prompt 内容区分调用阶段(think 的 system 含 "Leader")
        joined = " ".join(str(getattr(m, "content", "")) for m in messages)[:500]

        if self.scenario == "wait":
            # 第一轮 think 即返回 wait → supervisor 规则路由走 finish,循环干净终止
            return json.dumps({
                "action": "wait",
                "reason": "scripted eval: 确定性回归,不发起真实实验",
            }, ensure_ascii=False)

        if self.scenario == "think_experiment":
            # 返回 experiment,但 worker 阶段会得到"未启动"的结果 → 验证
            # 账本记录 no_experiment + 反卡死计数路径
            if "Leader" in joined:
                return json.dumps({
                    "action": "experiment",
                    "agent": "code",
                    "next_stage": "execute",
                    "task": "scripted task: 只创建说明文件,不训练",
                    "hypothesis": "scripted hypothesis",
                    "success_criteria": "scripted: 文件存在",
                }, ensure_ascii=False)
            # worker:最终答案(无工具调用)
            return "scripted worker: 任务完成,无需工具调用。"

        if self.scenario == "malformed":
            # 连续返回非 JSON → 验证解析失败 → retry → 熔断 finish 路径
            return "这不是 JSON,只是一段散文。"

        return json.dumps({
            "action": "wait",
            "reason": f"scripted({self.scenario})",
        }, ensure_ascii=False)
