"""Dashboard「生成草案」（Draft & Confirm）纯函数 + 端点契约测试（无网络调用）。

覆盖：
  - _parse_draft_json：合法 / ``` 围栏 / 坏 JSON / 缺 key / 空值错型
  - _draft_llm_settings：默认 deepseek / 国内 preset+env 被采用 /
    anthropic 配置被忽略回退 / 全无 key 返回空
  - _draft_prompt 组装、_render_brief 直通（草案字段 → 真实 brief）
  - /api/draft/brief 端点契约：200 / 400 / 503 / 502

隔离策略：
  - monkeypatch core.dashboard._load_config（真实仓库 config.yaml 永不进测试）
  - monkeypatch _call_draft_llm（端点测试永不发起真实网络请求）
  - monkeypatch.delenv 清空所有 preset 环境变量
"""
import asyncio
import json

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException

from core import dashboard as dash
from core.dashboard import (
    DraftBriefBody,
    _draft_prompt,
    _parse_draft_json,
    _render_brief,
)

_PRESET_ENVS = ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY",
                "MOONSHOT_API_KEY", "ZHIPUAI_API_KEY")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """隔离本机环境变量与真实 config.yaml，避免外部状态干扰断言。"""
    for env in _PRESET_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(dash, "_load_config", lambda project_dir: {})


# ── _parse_draft_json ──

def test_parse_draft_json_valid():
    data = _parse_draft_json(
        '{"goal": " 训练一个ViT ", "success_criteria": "准确率>85%", "constraints": "PyTorch"}')
    assert data == {"goal": "训练一个ViT", "success_criteria": "准确率>85%",
                    "constraints": "PyTorch"}


def test_parse_draft_json_strips_code_fence():
    text = '```json\n{"goal": "g", "success_criteria": "s", "constraints": "c"}\n```'
    assert _parse_draft_json(text) == {"goal": "g", "success_criteria": "s",
                                       "constraints": "c"}


def test_parse_draft_json_invalid_json():
    with pytest.raises(ValueError):
        _parse_draft_json("not json at all")


def test_parse_draft_json_missing_key():
    with pytest.raises(ValueError, match="constraints"):
        _parse_draft_json('{"goal": "g", "success_criteria": "s"}')


def test_parse_draft_json_empty_or_wrong_type():
    with pytest.raises(ValueError):
        _parse_draft_json('{"goal": "", "success_criteria": 5, "constraints": "c"}')


# ── _draft_llm_settings ──

def test_draft_llm_settings_defaults_to_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    url, model, key = dash._draft_llm_settings()
    assert (url, model, key) == ("https://api.deepseek.com/v1", "deepseek-chat", "k")


def test_draft_llm_settings_honors_preset_with_env(monkeypatch):
    monkeypatch.setattr(dash, "_load_config", lambda project_dir: {
        "agent": {"provider": "qwen", "model": "qwen-plus"}})
    monkeypatch.setenv("DASHSCOPE_API_KEY", "qk")
    url, model, key = dash._draft_llm_settings()
    assert url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert model == "qwen-plus"
    assert key == "qk"


def test_draft_llm_settings_ignores_anthropic_config(monkeypatch):
    """当前仓库真实配置（provider: anthropic）必须回退 deepseek，不能拿错 key。"""
    monkeypatch.setattr(dash, "_load_config", lambda project_dir: {
        "agent": {"provider": "anthropic", "model": "claude-sonnet-4-6"}})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    url, model, key = dash._draft_llm_settings()
    assert (url, model, key) == ("https://api.deepseek.com/v1", "deepseek-chat", "k")


def test_draft_llm_settings_no_key_returns_empty():
    url, model, key = dash._draft_llm_settings()
    assert url and model
    assert key == ""


def test_draft_llm_settings_preset_without_env_falls_back(monkeypatch):
    """config 指定了 preset 但 env 没设置 → 回退 deepseek 默认。"""
    monkeypatch.setattr(dash, "_load_config", lambda project_dir: {
        "agent": {"provider": "glm", "model": "glm-4"}})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    url, model, key = dash._draft_llm_settings()
    assert (url, model, key) == ("https://api.deepseek.com/v1", "deepseek-chat", "k")


# ── _draft_prompt / _render_brief ──

def test_draft_prompt_contains_goal_and_json_schema():
    prompt = _draft_prompt("在CIFAR-100上训练ViT", name="cifar-vit")
    assert "在CIFAR-100上训练ViT" in prompt
    assert "cifar-vit" in prompt
    system = dash._DRAFT_SYSTEM
    assert "success_criteria" in system and "constraints" in system
    assert '{"goal": ' in system and "dry-run" in system


def test_render_brief_integration_with_draft_fields():
    """草案三字段直接喂给真实 _render_brief，产出三节（草案 → brief 直通）。"""
    parsed = _parse_draft_json(
        '{"goal": "训练CIFAR-100 ViT", "success_criteria": "测试准确率>85%", '
        '"constraints": "PyTorch, max 50 epochs"}')
    brief = _render_brief("proj", parsed["goal"], parsed["success_criteria"],
                          parsed["constraints"])
    assert "## Goal" in brief and "训练CIFAR-100 ViT" in brief
    assert "## Success Criteria" in brief and "测试准确率>85%" in brief
    assert "## Constraints" in brief and "max 50 epochs" in brief
    assert "## MUST-DO" in brief  # 硬性要求段由 _render_brief 自带


# ── /api/draft/brief 端点契约 ──

def _run(coro):
    return asyncio.run(coro)


def test_api_draft_brief_endpoint_contract(monkeypatch):
    fake_fields = {"goal": "训练ViT", "success_criteria": "准确率>85%",
                   "constraints": "PyTorch, 50 epochs, 先 dry-run"}
    monkeypatch.setattr(dash, "_draft_llm_settings",
                        lambda: ("http://fake", "m", "k"))
    monkeypatch.setattr(dash, "_call_draft_llm",
                        lambda *a, **k: json.dumps(fake_fields, ensure_ascii=False))
    resp = _run(dash.api_draft_brief(DraftBriefBody(goal="在CIFAR-100上训练ViT",
                                                    name="cifar-vit")))
    assert resp == {"ok": True, **fake_fields}


def test_api_draft_brief_empty_goal_400():
    with pytest.raises(HTTPException) as exc:
        _run(dash.api_draft_brief(DraftBriefBody(goal="   ")))
    assert exc.value.status_code == 400


def test_api_draft_brief_no_key_503(monkeypatch):
    monkeypatch.setattr(dash, "_draft_llm_settings", lambda: ("", "", ""))
    with pytest.raises(HTTPException) as exc:
        _run(dash.api_draft_brief(DraftBriefBody(goal="训练ViT")))
    assert exc.value.status_code == 503


def test_api_draft_brief_bad_llm_json_502(monkeypatch):
    monkeypatch.setattr(dash, "_draft_llm_settings",
                        lambda: ("http://fake", "m", "k"))
    monkeypatch.setattr(dash, "_call_draft_llm", lambda *a, **k: "garbage")
    with pytest.raises(HTTPException) as exc:
        _run(dash.api_draft_brief(DraftBriefBody(goal="训练ViT")))
    assert exc.value.status_code == 502


def test_api_draft_brief_llm_error_502(monkeypatch):
    monkeypatch.setattr(dash, "_draft_llm_settings",
                        lambda: ("http://fake", "m", "k"))

    def boom(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(dash, "_call_draft_llm", boom)
    with pytest.raises(HTTPException) as exc:
        _run(dash.api_draft_brief(DraftBriefBody(goal="训练ViT")))
    assert exc.value.status_code == 502
