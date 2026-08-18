"""
轻量组件注册表 — 加工具/agent = 注册一个 manifest,零改核心。

对应 DSH 的"一切皆插件"理念的最小实现:
- tools:register_tool(agent_type, name, fn) → 自动出现在 TOOL_FUNCTIONS
  (agent 内嵌 bind_tools 与 MCP tools/list 都从注册表读取)
- 面试叙事:"加一个工具 = 一个 @tool + 一行注册,核心零改动;
  内嵌与 MCP 两侧自动可见。"
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger("autoresearcher.registry")


class ToolRegistry:
    """工具注册表:包装 nodes.TOOL_FUNCTIONS 的注册/查询(保持兼容)。"""

    def __init__(self, tool_functions: Optional[dict] = None):
        if tool_functions is None:
            from . import nodes
            tool_functions = nodes.TOOL_FUNCTIONS
        self._tf = tool_functions

    def register(self, agent_type: str, name: str, fn) -> None:
        """注册一个 @tool 对象到指定 agent(重复注册覆盖)。"""
        if agent_type not in self._tf:
            self._tf[agent_type] = []
        for i, (existing_name, _) in enumerate(self._tf[agent_type]):
            if existing_name == name:
                self._tf[agent_type][i] = (name, fn)
                logger.info("registry: tool %s.%s replaced", agent_type, name)
                return
        self._tf[agent_type].append((name, fn))
        logger.info("registry: tool %s.%s registered", agent_type, name)

    def tools_for(self, agent_type: str) -> list:
        return list(self._tf.get(agent_type, []))

    def all_names(self) -> set:
        return {name for tools in self._tf.values() for name, _ in tools}
