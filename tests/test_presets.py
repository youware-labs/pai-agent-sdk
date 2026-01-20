"""Tests for subagents.presets module."""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pai_agent_sdk.presets import (
    ANTHROPIC_1M_DEFAULT,
    ANTHROPIC_1M_HIGH,
    ANTHROPIC_1M_LOW,
    ANTHROPIC_1M_MEDIUM,
    ANTHROPIC_1M_OFF,
    ANTHROPIC_DEFAULT,
    ANTHROPIC_HIGH,
    ANTHROPIC_LOW,
    ANTHROPIC_MEDIUM,
    ANTHROPIC_OFF,
    INHERIT,
    OPENAI_DEFAULT,
    OPENAI_HIGH,
    OPENAI_LOW,
    OPENAI_MEDIUM,
    OPENAI_RESPONSES_DEFAULT,
    OPENAI_RESPONSES_HIGH,
    OPENAI_RESPONSES_LOW,
    OPENAI_RESPONSES_MEDIUM,
    ModelConfigPreset,
    ModelSettingsPreset,
    get_model_cfg,
    get_model_settings,
    list_model_cfg_presets,
    list_presets,
    resolve_model_cfg,
    resolve_model_settings,
)


def test_anthropic_presets_structure() -> None:
    """Test that Anthropic standard presets have expected structure (no beta headers)."""
    # Standard presets should NOT have beta headers but should have caching
    for preset in [ANTHROPIC_DEFAULT, ANTHROPIC_HIGH, ANTHROPIC_MEDIUM, ANTHROPIC_LOW]:
        assert "extra_headers" not in preset
        assert preset["anthropic_cache_instructions"] is True
        assert preset["anthropic_cache_messages"] is True

    # Thinking presets should have thinking config
    for preset in [ANTHROPIC_HIGH, ANTHROPIC_MEDIUM, ANTHROPIC_LOW]:
        assert "anthropic_thinking" in preset
        assert preset["anthropic_thinking"]["type"] == "enabled"
        assert "budget_tokens" in preset["anthropic_thinking"]

    # OFF should have thinking disabled
    assert ANTHROPIC_OFF["anthropic_thinking"]["type"] == "disabled"
    assert "extra_headers" not in ANTHROPIC_OFF


def test_anthropic_1m_presets_structure() -> None:
    """Test that Anthropic 1M presets have beta headers and caching."""
    # All 1M presets should have beta headers and caching
    for preset in [
        ANTHROPIC_1M_DEFAULT,
        ANTHROPIC_1M_HIGH,
        ANTHROPIC_1M_MEDIUM,
        ANTHROPIC_1M_LOW,
    ]:
        assert "extra_headers" in preset
        assert "anthropic-beta" in preset["extra_headers"]
        assert preset["anthropic_cache_instructions"] is True
        assert preset["anthropic_cache_messages"] is True

    # Thinking presets should have thinking config
    for preset in [ANTHROPIC_1M_HIGH, ANTHROPIC_1M_MEDIUM, ANTHROPIC_1M_LOW]:
        assert "anthropic_thinking" in preset
        assert preset["anthropic_thinking"]["type"] == "enabled"
        assert "budget_tokens" in preset["anthropic_thinking"]

    # 1M OFF should have thinking disabled but still have beta headers
    assert ANTHROPIC_1M_OFF["anthropic_thinking"]["type"] == "disabled"
    assert "extra_headers" in ANTHROPIC_1M_OFF
    assert "anthropic-beta" in ANTHROPIC_1M_OFF["extra_headers"]


def test_anthropic_thinking_budgets() -> None:
    """Test that Anthropic thinking budgets are ordered correctly."""
    # Standard presets
    high_budget = ANTHROPIC_HIGH["anthropic_thinking"]["budget_tokens"]
    medium_budget = ANTHROPIC_MEDIUM["anthropic_thinking"]["budget_tokens"]
    low_budget = ANTHROPIC_LOW["anthropic_thinking"]["budget_tokens"]
    assert high_budget > medium_budget > low_budget

    # 1M presets (should have same budget values)
    high_budget_1m = ANTHROPIC_1M_HIGH["anthropic_thinking"]["budget_tokens"]
    medium_budget_1m = ANTHROPIC_1M_MEDIUM["anthropic_thinking"]["budget_tokens"]
    low_budget_1m = ANTHROPIC_1M_LOW["anthropic_thinking"]["budget_tokens"]
    assert high_budget_1m > medium_budget_1m > low_budget_1m

    # Verify same values between standard and 1M
    assert high_budget == high_budget_1m
    assert medium_budget == medium_budget_1m
    assert low_budget == low_budget_1m


def test_openai_chat_presets_structure() -> None:
    """Test that OpenAI Chat presets have expected structure."""
    for preset in [OPENAI_DEFAULT, OPENAI_HIGH, OPENAI_MEDIUM, OPENAI_LOW]:
        assert "openai_reasoning_effort" in preset
        assert "max_tokens" in preset


def test_openai_responses_presets_structure() -> None:
    """Test that OpenAI Responses presets have expected structure."""
    for preset in [
        OPENAI_RESPONSES_DEFAULT,
        OPENAI_RESPONSES_HIGH,
        OPENAI_RESPONSES_MEDIUM,
        OPENAI_RESPONSES_LOW,
    ]:
        assert "openai_reasoning_effort" in preset
        assert "openai_reasoning_summary" in preset


def test_get_model_settings_by_enum() -> None:
    """Test getting model settings by enum."""
    settings = get_model_settings(ModelSettingsPreset.ANTHROPIC_HIGH)
    assert settings == ANTHROPIC_HIGH

    # Test 1M preset
    settings_1m = get_model_settings(ModelSettingsPreset.ANTHROPIC_1M_HIGH)
    assert settings_1m == ANTHROPIC_1M_HIGH


def test_get_model_settings_by_string() -> None:
    """Test getting model settings by string name."""
    settings = get_model_settings("anthropic_high")
    assert settings == ANTHROPIC_HIGH

    # Test 1M preset
    settings_1m = get_model_settings("anthropic_1m_high")
    assert settings_1m == ANTHROPIC_1M_HIGH


def test_get_model_settings_by_alias() -> None:
    """Test getting model settings by alias."""
    settings = get_model_settings("anthropic")
    assert settings == ANTHROPIC_DEFAULT

    # Test 1M alias
    settings_1m = get_model_settings("anthropic_1m")
    assert settings_1m == ANTHROPIC_1M_DEFAULT

    settings = get_model_settings("openai")
    assert settings == OPENAI_DEFAULT


def test_get_model_settings_invalid() -> None:
    """Test that invalid preset name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown preset"):
        get_model_settings("invalid_preset_name")


def test_resolve_model_settings_none() -> None:
    """Test that None returns None."""
    result = resolve_model_settings(None)
    assert result is None


def test_resolve_model_settings_dict() -> None:
    """Test that dict is returned as-is."""
    custom = {"temperature": 0.5, "max_tokens": 1000}
    result = resolve_model_settings(custom)
    assert result == custom


def test_resolve_model_settings_string() -> None:
    """Test that string is resolved to preset."""
    result = resolve_model_settings("anthropic_medium")
    assert result == ANTHROPIC_MEDIUM

    # Test 1M preset
    result_1m = resolve_model_settings("anthropic_1m_medium")
    assert result_1m == ANTHROPIC_1M_MEDIUM


def test_list_presets() -> None:
    """Test list_presets returns all available presets."""
    presets = list_presets()

    assert presets == snapshot([
        "anthropic",
        "anthropic_1m",
        "anthropic_1m_default",
        "anthropic_1m_high",
        "anthropic_1m_low",
        "anthropic_1m_medium",
        "anthropic_1m_off",
        "anthropic_default",
        "anthropic_high",
        "anthropic_low",
        "anthropic_medium",
        "anthropic_off",
        "gemini",
        "gemini_2.5",
        "gemini_3",
        "gemini_thinking_budget_default",
        "gemini_thinking_budget_high",
        "gemini_thinking_budget_low",
        "gemini_thinking_budget_medium",
        "gemini_thinking_level_default",
        "gemini_thinking_level_high",
        "gemini_thinking_level_low",
        "gemini_thinking_level_medium",
        "gemini_thinking_level_minimal",
        "high",
        "low",
        "medium",
        "openai",
        "openai_default",
        "openai_high",
        "openai_low",
        "openai_medium",
        "openai_responses",
        "openai_responses_default",
        "openai_responses_high",
        "openai_responses_low",
        "openai_responses_medium",
    ])


# =============================================================================
# ModelConfigPreset Tests
# =============================================================================


def test_model_cfg_presets_structure() -> None:
    """Test that ModelConfig presets have expected structure."""
    cfg = get_model_cfg("claude_200k")
    assert cfg["context_window"] == 200_000
    assert cfg["max_videos"] == 0  # Claude doesn't support video
    assert "max_images" in cfg
    assert "capabilities" in cfg

    cfg_1m = get_model_cfg("claude_1m")
    assert cfg_1m["context_window"] == 1_000_000
    assert cfg_1m["max_videos"] == 0  # Claude doesn't support video


def test_model_cfg_capabilities() -> None:
    """Test that ModelConfig presets have correct capabilities."""
    from pai_agent_sdk.context import ModelCapability

    # Claude: vision + document, no video
    cfg_claude = get_model_cfg("claude_200k")
    assert ModelCapability.vision in cfg_claude["capabilities"]
    assert ModelCapability.document_understanding in cfg_claude["capabilities"]
    assert ModelCapability.video_understanding not in cfg_claude["capabilities"]

    # GPT-5: vision only
    cfg_gpt = get_model_cfg("gpt5_270k")
    assert ModelCapability.vision in cfg_gpt["capabilities"]
    assert ModelCapability.video_understanding not in cfg_gpt["capabilities"]

    # Gemini: vision + video + document
    cfg_gemini = get_model_cfg("gemini_1m")
    assert ModelCapability.vision in cfg_gemini["capabilities"]
    assert ModelCapability.video_understanding in cfg_gemini["capabilities"]
    assert ModelCapability.document_understanding in cfg_gemini["capabilities"]


def test_get_model_cfg_by_enum() -> None:
    """Test getting model config by enum."""
    cfg = get_model_cfg(ModelConfigPreset.CLAUDE_200K)
    assert cfg["context_window"] == 200_000

    cfg_gemini = get_model_cfg(ModelConfigPreset.GEMINI_1M)
    assert cfg_gemini["context_window"] == 1_000_000
    assert cfg_gemini["max_videos"] == 1  # Gemini supports video


def test_get_model_cfg_by_string() -> None:
    """Test getting model config by string name."""
    cfg = get_model_cfg("claude_200k")
    assert cfg["context_window"] == 200_000

    cfg_gpt = get_model_cfg("gpt5_270k")
    assert cfg_gpt["context_window"] == 270_000
    assert cfg_gpt["max_videos"] == 0  # GPT doesn't support video


def test_get_model_cfg_by_alias() -> None:
    """Test getting model config by alias."""
    cfg = get_model_cfg("claude")
    assert cfg["context_window"] == 200_000

    cfg = get_model_cfg("anthropic")
    assert cfg["context_window"] == 200_000

    cfg = get_model_cfg("openai")
    assert cfg["context_window"] == 270_000  # GPT-5 series

    cfg = get_model_cfg("gemini")
    assert cfg["context_window"] == 200_000  # Default to 200K (cheaper)


def test_get_model_cfg_invalid() -> None:
    """Test that invalid preset name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown ModelConfig preset"):
        get_model_cfg("invalid_preset_name")


def test_resolve_model_cfg_none() -> None:
    """Test that None returns None (inherit)."""
    result = resolve_model_cfg(None)
    assert result is None


def test_resolve_model_cfg_inherit() -> None:
    """Test that 'inherit' returns None."""
    result = resolve_model_cfg(INHERIT)
    assert result is None
    result = resolve_model_cfg("inherit")
    assert result is None


def test_resolve_model_cfg_dict() -> None:
    """Test that dict is returned as-is."""
    custom = {"context_window": 100000, "max_images": 10}
    result = resolve_model_cfg(custom)
    assert result == custom


def test_resolve_model_cfg_string() -> None:
    """Test that string is resolved to preset."""
    result = resolve_model_cfg("claude_200k")
    assert result is not None
    assert result["context_window"] == 200_000


def test_list_model_cfg_presets() -> None:
    """Test list_model_cfg_presets returns all available presets."""
    presets = list_model_cfg_presets()

    assert presets == snapshot([
        "anthropic",
        "claude",
        "claude_1m",
        "claude_200k",
        "gemini",
        "gemini_1m",
        "gemini_200k",
        "gpt5",
        "gpt5_270k",
        "openai",
    ])
