"""Provider 配置解析回归测试 — LangGraph 引擎 (core/nodes.py)。

从旧引擎 (core/agents.py.AgentDispatcher) 迁移。配置解析被提取为纯函数
resolve_provider_config(无 IO,可独立测试),客户端构造 (ChatOpenAI) 单独测。
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from core import nodes
from core.nodes import ResearchGraph, resolve_provider_config


class ProviderPresetTests(unittest.TestCase):
    def test_deepseek_preset_fills_base_url_and_key_env(self):
        r = resolve_provider_config(provider="deepseek", model="deepseek-chat")
        self.assertEqual(r["base_url"], "https://api.deepseek.com/v1")
        self.assertEqual(r["api_key_env"], "DEEPSEEK_API_KEY")
        self.assertEqual(r["provider_label"], "deepseek")
        self.assertEqual(r["model"], "deepseek-chat")  # model passed through verbatim

    def test_preset_aliases_resolve(self):
        for name, host in [
            ("qwen", "dashscope.aliyuncs.com"),
            ("kimi", "api.moonshot.cn"),
            ("glm", "open.bigmodel.cn"),
        ]:
            r = resolve_provider_config(provider=name, model="m")
            self.assertIn(host, r["base_url"])

    def test_explicit_base_url_and_key_env_override_preset(self):
        r = resolve_provider_config(
            provider="deepseek", model="deepseek-chat",
            base_url="https://proxy.internal/v1", api_key_env="MY_KEY",
        )
        self.assertEqual(r["base_url"], "https://proxy.internal/v1")
        self.assertEqual(r["api_key_env"], "MY_KEY")

    def test_base_providers_pass_through_untouched(self):
        r = resolve_provider_config(provider="openai", model="gpt-5",
                                    base_url="https://api.openai.com/v1")
        self.assertEqual(r["base_url"], "https://api.openai.com/v1")
        self.assertEqual(r["api_key_env"], "")

    def test_unknown_provider_raises_with_preset_names(self):
        with self.assertRaisesRegex(ValueError, "deepseek"):
            resolve_provider_config(provider="not-a-provider", model="m")

    def test_auth_token_env_passthrough(self):
        r = resolve_provider_config(
            provider="anthropic", model="m", auth_token_env="MINIMAX_AUTH_TOKEN"
        )
        self.assertEqual(r["auth_token_env"], "MINIMAX_AUTH_TOKEN")


class MakeLlmTests(unittest.TestCase):
    def setUp(self):
        self.graph = object.__new__(ResearchGraph)

    def test_make_llm_passes_resolved_config_to_chat_openai(self):
        ctor = MagicMock()
        with patch.object(nodes, "ChatOpenAI", ctor):
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "secret-key"}, clear=False):
                llm = self.graph._make_llm(
                    "qwen-plus", "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "DASHSCOPE_API_KEY",
                )
        ctor.assert_called_once_with(
            model="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="secret-key",
            temperature=0,
        )
        self.assertIsNotNone(llm)

    def test_make_llm_empty_model_returns_none(self):
        with patch.object(nodes, "ChatOpenAI", MagicMock()):
            self.assertIsNone(self.graph._make_llm("", "https://x", "KEY"))


if __name__ == "__main__":
    unittest.main()
