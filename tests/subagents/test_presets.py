"""Tests for subagents.presets module."""

from __future__ import annotations

import pytest

from pai_agent_sdk.subagents.presets import (
    ANTHROPIC_DEFAULT,
    ANTHROPIC_HIGH,
    ANTHROPIC_LOW,
    ANTHROPIC_MEDIUM,
    ANTHROPIC_OFF,
    GEMINI_DEFAULT,
    GEMINI_HIGH,
    GEMINI_LOW,
    GEMINI_MEDIUM,
    GEMINI_MINIMAL,
    OPENAI_DEFAULT,
    OPENAI_HIGH,
    OPENAI_LOW,
    OPENAI_MEDIUM,
    OPENAI_RESPONSES_DEFAULT,
    OPENAI_RESPONSES_HIGH,
    OPENAI_RESPONSES_LOW,
    OPENAI_RESPONSES_MEDIUM,
    ModelSettingsPreset,
    get_model_settings,
    list_presets,
    resolve_model_settings,
)


def test_anthropic_presets_structure() -> None:
    """Test that Anthropic presets have expected structure."""
    # All should have beta headers and caching
    for preset in [ANTHROPIC_DEFAULT, ANTHROPIC_HIGH, ANTHROPIC_MEDIUM, ANTHROPIC_LOW]:
        assert "extra_headers" in preset
        assert "anthropic-beta" in preset["extra_headers"]
        assert preset["anthropic_cache_instructions"] is True
        assert preset["anthropic_cache_messages"] is True

    # Thinking presets should have thinking config
    for preset in [ANTHROPIC_HIGH, ANTHROPIC_MEDIUM, ANTHROPIC_LOW]:
        assert "anthropic_thinking" in preset
        assert preset["anthropic_thinking"]["type"] == "enabled"
        assert "budget_tokens" in preset["anthropic_thinking"]

    # OFF should have thinking disabled
    assert ANTHROPIC_OFF["anthropic_thinking"]["type"] == "disabled"


def test_anthropic_thinking_budgets() -> None:
    """Test that Anthropic thinking budgets are ordered correctly."""
    high_budget = ANTHROPIC_HIGH["anthropic_thinking"]["budget_tokens"]
    medium_budget = ANTHROPIC_MEDIUM["anthropic_thinking"]["budget_tokens"]
    low_budget = ANTHROPIC_LOW["anthropic_thinking"]["budget_tokens"]

    assert high_budget > medium_budget > low_budget


def test_openai_chat_presets_structure() -> None:
    """Test that OpenAI Chat presets have expected structure."""
    for preset in [OPENAI_DEFAULT, OPENAI_HIGH, OPENAI_MEDIUM, OPENAI_LOW]:
        assert "openai_reasoning_effort" in preset
        assert "max_tokens" in preset


def test_openai_responses_presets_structure() -> None:
    """Test that OpenAI Responses presets have expected structure."""
    for preset in [OPENAI_RESPONSES_DEFAULT, OPENAI_RESPONSES_HIGH, OPENAI_RESPONSES_MEDIUM, OPENAI_RESPONSES_LOW]:
        assert "openai_reasoning_effort" in preset
        assert "openai_reasoning_summary" in preset


def test_gemini_presets_structure() -> None:
    """Test that Gemini presets have expected structure."""
    for preset in [GEMINI_DEFAULT, GEMINI_HIGH, GEMINI_MEDIUM, GEMINI_LOW]:
        assert "google_thinking_config" in preset
        assert "max_tokens" in preset

    # MINIMAL should have MINIMAL thinking level
    assert GEMINI_MINIMAL["google_thinking_config"]["thinking_level"] == "MINIMAL"


def test_get_model_settings_by_enum() -> None:
    """Test getting model settings by enum."""
    settings = get_model_settings(ModelSettingsPreset.ANTHROPIC_HIGH)
    assert settings == ANTHROPIC_HIGH


def test_get_model_settings_by_string() -> None:
    """Test getting model settings by string name."""
    settings = get_model_settings("anthropic_high")
    assert settings == ANTHROPIC_HIGH


def test_get_model_settings_by_alias() -> None:
    """Test getting model settings by alias."""
    settings = get_model_settings("anthropic")
    assert settings == ANTHROPIC_DEFAULT

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


def test_list_presets() -> None:
    """Test list_presets returns all available presets."""
    presets = list_presets()

    # Should include main presets
    assert "anthropic_high" in presets
    assert "openai_medium" in presets
    assert "gemini_low" in presets

    # Should include aliases
    assert "anthropic" in presets
    assert "openai" in presets
    assert "gemini" in presets
