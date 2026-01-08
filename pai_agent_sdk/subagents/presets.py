"""ModelSettings presets for different providers and thinking levels.

This module provides pre-configured ModelSettings for common use cases across
different model providers (Anthropic, OpenAI, Gemini). Each provider has presets
for different "thinking levels" (reasoning intensity).

Naming Convention:
- `{provider}_{level}` - e.g., `anthropic_high`, `openai_medium`
- `{provider}_{api}_{level}` - for providers with multiple APIs, e.g., `openai_responses_high`

Thinking Levels:
- `high`: Maximum reasoning depth, higher latency
- `medium`: Balanced reasoning (default)
- `low`: Minimal reasoning, lower latency

Usage::

    from pai_agent_sdk.subagents.presets import get_model_settings, ModelSettingsPreset

    # Get preset by name
    settings = get_model_settings("anthropic_high")

    # Or use enum
    settings = get_model_settings(ModelSettingsPreset.ANTHROPIC_HIGH)

    # Use with Agent
    agent = Agent(model="anthropic:claude-sonnet-4", model_settings=settings)
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    pass

# =============================================================================
# Constants
# =============================================================================

K_TOKENS = 1024

# Anthropic beta headers for extended context
ANTHROPIC_BETAS = [
    "context-1m-2025-08-07",
]

ANTHROPIC_BETA_HEADERS = {
    "anthropic-beta": ",".join(ANTHROPIC_BETAS),
}


# =============================================================================
# Preset Enum
# =============================================================================


class ModelSettingsPreset(str, Enum):
    """Available ModelSettings presets."""

    # Anthropic presets (Claude with thinking)
    ANTHROPIC_DEFAULT = "anthropic_default"
    ANTHROPIC_HIGH = "anthropic_high"
    ANTHROPIC_MEDIUM = "anthropic_medium"
    ANTHROPIC_LOW = "anthropic_low"
    ANTHROPIC_OFF = "anthropic_off"

    # OpenAI Chat Completions presets (GPT-4, etc.)
    OPENAI_DEFAULT = "openai_default"
    OPENAI_HIGH = "openai_high"
    OPENAI_MEDIUM = "openai_medium"
    OPENAI_LOW = "openai_low"

    # OpenAI Responses API presets (o1, o3 reasoning models)
    OPENAI_RESPONSES_DEFAULT = "openai_responses_default"
    OPENAI_RESPONSES_HIGH = "openai_responses_high"
    OPENAI_RESPONSES_MEDIUM = "openai_responses_medium"
    OPENAI_RESPONSES_LOW = "openai_responses_low"

    # Gemini presets (with thinking config)
    GEMINI_DEFAULT = "gemini_default"
    GEMINI_HIGH = "gemini_high"
    GEMINI_MEDIUM = "gemini_medium"
    GEMINI_LOW = "gemini_low"
    GEMINI_MINIMAL = "gemini_minimal"


# =============================================================================
# Anthropic Presets
# =============================================================================


def _anthropic_settings(
    thinking_budget: int,
    max_tokens: int = 21 * K_TOKENS,
) -> dict[str, Any]:
    """Create Anthropic model settings with thinking enabled.

    Args:
        thinking_budget: Token budget for thinking (higher = more reasoning).
        max_tokens: Maximum output tokens.

    Returns:
        Dict suitable for AnthropicModelSettings.
    """
    return {
        "extra_headers": ANTHROPIC_BETA_HEADERS,
        "max_tokens": max_tokens,
        "anthropic_thinking": {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        },
        "anthropic_cache_instructions": True,
        "anthropic_cache_response": True,
        "anthropic_cache_messages": True,
    }


ANTHROPIC_DEFAULT: dict[str, Any] = _anthropic_settings(
    thinking_budget=16 * K_TOKENS,
    max_tokens=16 * K_TOKENS,
)
"""Anthropic default: Same as medium, 16K thinking budget."""

ANTHROPIC_HIGH: dict[str, Any] = _anthropic_settings(
    thinking_budget=32 * K_TOKENS,
    max_tokens=21 * K_TOKENS,
)
"""Anthropic high thinking: 32K thinking budget, max reasoning depth."""

ANTHROPIC_MEDIUM: dict[str, Any] = _anthropic_settings(
    thinking_budget=16 * K_TOKENS,
    max_tokens=16 * K_TOKENS,
)
"""Anthropic medium thinking: 16K thinking budget, balanced reasoning."""

ANTHROPIC_LOW: dict[str, Any] = _anthropic_settings(
    thinking_budget=4 * K_TOKENS,
    max_tokens=8 * K_TOKENS,
)
"""Anthropic low thinking: 4K thinking budget, minimal reasoning overhead."""

ANTHROPIC_OFF: dict[str, Any] = {
    "extra_headers": ANTHROPIC_BETA_HEADERS,
    "anthropic_thinking": {
        "type": "disabled",
    },
    "anthropic_cache_instructions": True,
    "anthropic_cache_response": True,
    "anthropic_cache_messages": True,
}
"""Anthropic off: Thinking disabled, with 1M context beta and caching enabled."""


# =============================================================================
# OpenAI Chat Completions Presets
# =============================================================================


def _openai_chat_settings(
    reasoning_effort: Literal["low", "medium", "high"] | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Create OpenAI Chat Completions settings.

    Note: reasoning_effort is supported for o1/o3 models via Chat Completions API.
    For non-reasoning models (GPT-4, etc.), reasoning_effort is ignored.

    Args:
        reasoning_effort: Reasoning intensity for o1/o3 models ('low', 'medium', 'high').
        max_tokens: Maximum output tokens (None for model default).

    Returns:
        Dict suitable for OpenAIChatModelSettings.
    """
    settings: dict[str, Any] = {}
    if reasoning_effort is not None:
        settings["openai_reasoning_effort"] = reasoning_effort
    if max_tokens is not None:
        settings["max_tokens"] = max_tokens
    return settings


OPENAI_DEFAULT: dict[str, Any] = _openai_chat_settings(
    reasoning_effort="medium",
    max_tokens=8 * K_TOKENS,
)
"""OpenAI Chat default: Same as medium, balanced reasoning and max_tokens."""

OPENAI_HIGH: dict[str, Any] = _openai_chat_settings(
    reasoning_effort="high",
    max_tokens=16 * K_TOKENS,
)
"""OpenAI Chat high: Maximum reasoning effort, higher max_tokens."""

OPENAI_MEDIUM: dict[str, Any] = _openai_chat_settings(
    reasoning_effort="medium",
    max_tokens=8 * K_TOKENS,
)
"""OpenAI Chat medium: Balanced reasoning effort and max_tokens."""

OPENAI_LOW: dict[str, Any] = _openai_chat_settings(
    reasoning_effort="low",
    max_tokens=4 * K_TOKENS,
)
"""OpenAI Chat low: Minimal reasoning, lower max_tokens for faster responses."""


# =============================================================================
# OpenAI Responses API Presets (o1, o3 reasoning models)
# =============================================================================


def _openai_responses_settings(
    reasoning_effort: Literal["low", "medium", "high"],
    reasoning_summary: Literal["detailed", "concise", "auto"] = "auto",
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Create OpenAI Responses API settings for reasoning models.

    Args:
        reasoning_effort: Reasoning intensity ('low', 'medium', 'high').
        reasoning_summary: Summary level of reasoning process.
        max_tokens: Maximum output tokens (None for model default).

    Returns:
        Dict suitable for OpenAIResponsesModelSettings.
    """
    settings: dict[str, Any] = {
        "openai_reasoning_effort": reasoning_effort,
        "openai_reasoning_summary": reasoning_summary,
    }
    if max_tokens is not None:
        settings["max_tokens"] = max_tokens
    return settings


OPENAI_RESPONSES_DEFAULT: dict[str, Any] = _openai_responses_settings(
    reasoning_effort="medium",
    reasoning_summary="auto",
    max_tokens=16 * K_TOKENS,
)
"""OpenAI Responses default: Same as medium, balanced reasoning effort."""

OPENAI_RESPONSES_HIGH: dict[str, Any] = _openai_responses_settings(
    reasoning_effort="high",
    reasoning_summary="detailed",
    max_tokens=32 * K_TOKENS,
)
"""OpenAI Responses high: Maximum reasoning effort with detailed summary."""

OPENAI_RESPONSES_MEDIUM: dict[str, Any] = _openai_responses_settings(
    reasoning_effort="medium",
    reasoning_summary="auto",
    max_tokens=16 * K_TOKENS,
)
"""OpenAI Responses medium: Balanced reasoning effort."""

OPENAI_RESPONSES_LOW: dict[str, Any] = _openai_responses_settings(
    reasoning_effort="low",
    reasoning_summary="concise",
    max_tokens=8 * K_TOKENS,
)
"""OpenAI Responses low: Minimal reasoning, faster responses."""


# =============================================================================
# Gemini Presets
# =============================================================================


def _gemini_settings(
    thinking_level: Literal["HIGH", "MEDIUM", "LOW"] | None = None,
    thinking_budget: int | None = None,
    max_tokens: int | None = None,
    include_thoughts: bool = False,
) -> dict[str, Any]:
    """Create Gemini model settings with thinking config.

    Note: Gemini 3 uses thinking_level, Gemini 2.5 uses thinking_budget.
    You can specify both for compatibility.

    Args:
        thinking_level: For Gemini 3+ ('HIGH', 'MEDIUM', 'LOW').
        thinking_budget: For Gemini 2.5 (token budget).
        max_tokens: Maximum output tokens.
        include_thoughts: Whether to include thinking in response.

    Returns:
        Dict suitable for GoogleModelSettings.
    """
    thinking_config: dict[str, Any] = {
        "include_thoughts": include_thoughts,
    }
    if thinking_level is not None:
        thinking_config["thinking_level"] = thinking_level
    if thinking_budget is not None:
        thinking_config["thinking_budget"] = thinking_budget

    settings: dict[str, Any] = {
        "google_thinking_config": thinking_config,
    }
    if max_tokens is not None:
        settings["max_tokens"] = max_tokens
    return settings


GEMINI_DEFAULT: dict[str, Any] = _gemini_settings(
    thinking_level="MEDIUM",
    thinking_budget=16 * K_TOKENS,
    max_tokens=16 * K_TOKENS,
    include_thoughts=False,
)
"""Gemini default: Same as medium, balanced reasoning."""

GEMINI_HIGH: dict[str, Any] = _gemini_settings(
    thinking_level="HIGH",
    thinking_budget=32 * K_TOKENS,
    max_tokens=21 * K_TOKENS,
    include_thoughts=False,
)
"""Gemini high: Maximum reasoning depth (default for Gemini 3)."""

GEMINI_MEDIUM: dict[str, Any] = _gemini_settings(
    thinking_level="MEDIUM",
    thinking_budget=16 * K_TOKENS,
    max_tokens=16 * K_TOKENS,
    include_thoughts=False,
)
"""Gemini medium: Balanced reasoning (Gemini 3 Flash only)."""

GEMINI_LOW: dict[str, Any] = _gemini_settings(
    thinking_level="LOW",
    thinking_budget=4 * K_TOKENS,
    max_tokens=8 * K_TOKENS,
    include_thoughts=False,
)
"""Gemini low: Minimal reasoning, lower latency."""

GEMINI_MINIMAL: dict[str, Any] = {
    "google_thinking_config": {
        "thinking_level": "MINIMAL",
        "include_thoughts": False,
    },
    "max_tokens": 4 * K_TOKENS,
}
"""Gemini minimal: Near-zero thinking (Gemini 3 Flash only, may still think for complex tasks)."""


# =============================================================================
# Preset Registry
# =============================================================================

_PRESET_REGISTRY: dict[str, dict[str, Any]] = {
    # Anthropic
    ModelSettingsPreset.ANTHROPIC_DEFAULT.value: ANTHROPIC_DEFAULT,
    ModelSettingsPreset.ANTHROPIC_HIGH.value: ANTHROPIC_HIGH,
    ModelSettingsPreset.ANTHROPIC_MEDIUM.value: ANTHROPIC_MEDIUM,
    ModelSettingsPreset.ANTHROPIC_LOW.value: ANTHROPIC_LOW,
    ModelSettingsPreset.ANTHROPIC_OFF.value: ANTHROPIC_OFF,
    # OpenAI Chat
    ModelSettingsPreset.OPENAI_DEFAULT.value: OPENAI_DEFAULT,
    ModelSettingsPreset.OPENAI_HIGH.value: OPENAI_HIGH,
    ModelSettingsPreset.OPENAI_MEDIUM.value: OPENAI_MEDIUM,
    ModelSettingsPreset.OPENAI_LOW.value: OPENAI_LOW,
    # OpenAI Responses
    ModelSettingsPreset.OPENAI_RESPONSES_DEFAULT.value: OPENAI_RESPONSES_DEFAULT,
    ModelSettingsPreset.OPENAI_RESPONSES_HIGH.value: OPENAI_RESPONSES_HIGH,
    ModelSettingsPreset.OPENAI_RESPONSES_MEDIUM.value: OPENAI_RESPONSES_MEDIUM,
    ModelSettingsPreset.OPENAI_RESPONSES_LOW.value: OPENAI_RESPONSES_LOW,
    # Gemini
    ModelSettingsPreset.GEMINI_DEFAULT.value: GEMINI_DEFAULT,
    ModelSettingsPreset.GEMINI_HIGH.value: GEMINI_HIGH,
    ModelSettingsPreset.GEMINI_MEDIUM.value: GEMINI_MEDIUM,
    ModelSettingsPreset.GEMINI_LOW.value: GEMINI_LOW,
    ModelSettingsPreset.GEMINI_MINIMAL.value: GEMINI_MINIMAL,
}

# Short aliases for convenience
_PRESET_ALIASES: dict[str, str] = {
    # Provider defaults (default preset)
    "anthropic": ModelSettingsPreset.ANTHROPIC_DEFAULT.value,
    "openai": ModelSettingsPreset.OPENAI_DEFAULT.value,
    "openai_responses": ModelSettingsPreset.OPENAI_RESPONSES_DEFAULT.value,
    "gemini": ModelSettingsPreset.GEMINI_DEFAULT.value,
    # Generic level aliases (default to anthropic)
    "high": ModelSettingsPreset.ANTHROPIC_HIGH.value,
    "medium": ModelSettingsPreset.ANTHROPIC_MEDIUM.value,
    "low": ModelSettingsPreset.ANTHROPIC_LOW.value,
}


# =============================================================================
# Public API
# =============================================================================


def get_model_settings(preset: str | ModelSettingsPreset) -> dict[str, Any]:
    """Get ModelSettings by preset name.

    Args:
        preset: Preset name (string) or ModelSettingsPreset enum.

    Returns:
        ModelSettings dict for the specified preset.

    Raises:
        ValueError: If preset name is not found.

    Example::

        # By string name
        settings = get_model_settings("anthropic_high")

        # By enum
        settings = get_model_settings(ModelSettingsPreset.GEMINI_MEDIUM)

        # By alias
        settings = get_model_settings("anthropic")  # -> anthropic_medium
    """
    name = preset.value if isinstance(preset, ModelSettingsPreset) else preset

    # Check aliases first
    if name in _PRESET_ALIASES:
        name = _PRESET_ALIASES[name]

    if name not in _PRESET_REGISTRY:
        available = list(_PRESET_REGISTRY.keys()) + list(_PRESET_ALIASES.keys())
        msg = f"Unknown preset: {preset!r}. Available: {sorted(available)}"
        raise ValueError(msg)

    return _PRESET_REGISTRY[name]


def resolve_model_settings(
    preset_or_dict: str | dict[str, Any] | ModelSettingsPreset | None,
) -> dict[str, Any] | None:
    """Resolve a preset name or dict to ModelSettings.

    This is the main entry point for resolving model settings from various formats:
    - None -> None (use model defaults)
    - str -> lookup preset by name
    - ModelSettingsPreset -> lookup preset by enum
    - dict -> return as-is (assumed to be valid ModelSettings)

    Args:
        preset_or_dict: Preset name, enum, dict, or None.

    Returns:
        ModelSettings dict or None.

    Example::

        # From YAML config
        config_value = "anthropic_high"
        settings = resolve_model_settings(config_value)

        # From dict in YAML
        config_value = {"temperature": 0.5, "max_tokens": 4096}
        settings = resolve_model_settings(config_value)
    """
    if preset_or_dict is None:
        return None
    if isinstance(preset_or_dict, dict):
        return preset_or_dict
    return get_model_settings(preset_or_dict)


def list_presets() -> list[str]:
    """List all available preset names.

    Returns:
        List of preset names (including aliases).
    """
    return sorted(set(_PRESET_REGISTRY.keys()) | set(_PRESET_ALIASES.keys()))
