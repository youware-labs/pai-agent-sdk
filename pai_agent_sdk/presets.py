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

from pai_agent_sdk.context import ModelCapability

if TYPE_CHECKING:
    from pydantic_ai import ModelSettings

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

    # Anthropic standard presets (no beta headers)
    ANTHROPIC_DEFAULT = "anthropic_default"
    ANTHROPIC_HIGH = "anthropic_high"
    ANTHROPIC_MEDIUM = "anthropic_medium"
    ANTHROPIC_LOW = "anthropic_low"
    ANTHROPIC_OFF = "anthropic_off"

    # Anthropic 1M context presets (with beta headers for extended context)
    ANTHROPIC_1M_DEFAULT = "anthropic_1m_default"
    ANTHROPIC_1M_HIGH = "anthropic_1m_high"
    ANTHROPIC_1M_MEDIUM = "anthropic_1m_medium"
    ANTHROPIC_1M_LOW = "anthropic_1m_low"
    ANTHROPIC_1M_OFF = "anthropic_1m_off"

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

    # Gemini thinking_budget presets (for Gemini 2.5)
    GEMINI_THINKING_BUDGET_DEFAULT = "gemini_thinking_budget_default"
    GEMINI_THINKING_BUDGET_HIGH = "gemini_thinking_budget_high"
    GEMINI_THINKING_BUDGET_MEDIUM = "gemini_thinking_budget_medium"
    GEMINI_THINKING_BUDGET_LOW = "gemini_thinking_budget_low"

    # Gemini thinking_level presets (for Gemini 3)
    GEMINI_THINKING_LEVEL_DEFAULT = "gemini_thinking_level_default"
    GEMINI_THINKING_LEVEL_HIGH = "gemini_thinking_level_high"
    GEMINI_THINKING_LEVEL_MEDIUM = "gemini_thinking_level_medium"
    GEMINI_THINKING_LEVEL_LOW = "gemini_thinking_level_low"
    GEMINI_THINKING_LEVEL_MINIMAL = "gemini_thinking_level_minimal"


# =============================================================================
# Anthropic Presets
# =============================================================================


def _anthropic_settings(
    thinking_budget: int,
    max_tokens: int = 21 * K_TOKENS,
    *,
    use_1m_context: bool = False,
) -> dict[str, Any]:
    """Create Anthropic model settings with thinking enabled.

    Args:
        thinking_budget: Token budget for thinking (higher = more reasoning).
        max_tokens: Maximum output tokens.
        use_1m_context: Whether to include 1M context beta headers.

    Returns:
        Dict suitable for AnthropicModelSettings.
    """
    settings: dict[str, Any] = {
        "max_tokens": max_tokens,
        "anthropic_thinking": {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        },
        "anthropic_cache_instructions": True,
        "anthropic_cache_response": True,
        "anthropic_cache_messages": True,
    }
    if use_1m_context:
        settings["extra_headers"] = ANTHROPIC_BETA_HEADERS
    return settings


def _anthropic_off_settings(*, use_1m_context: bool = False) -> dict[str, Any]:
    """Create Anthropic model settings with thinking disabled.

    Args:
        use_1m_context: Whether to include 1M context beta headers.

    Returns:
        Dict suitable for AnthropicModelSettings.
    """
    settings: dict[str, Any] = {
        "anthropic_thinking": {
            "type": "disabled",
        },
        "anthropic_cache_instructions": True,
        "anthropic_cache_response": True,
        "anthropic_cache_messages": True,
    }
    if use_1m_context:
        settings["extra_headers"] = ANTHROPIC_BETA_HEADERS
    return settings


# -----------------------------------------------------------------------------
# Standard Anthropic presets (no beta headers)
# -----------------------------------------------------------------------------

ANTHROPIC_DEFAULT: dict[str, Any] = _anthropic_settings(
    thinking_budget=16 * K_TOKENS,
    max_tokens=21 * K_TOKENS,
)
"""Anthropic default: Same as medium, 16K thinking budget."""

ANTHROPIC_HIGH: dict[str, Any] = _anthropic_settings(
    thinking_budget=32 * K_TOKENS,
    max_tokens=21 * K_TOKENS,
)
"""Anthropic high thinking: 32K thinking budget, max reasoning depth."""

ANTHROPIC_MEDIUM: dict[str, Any] = _anthropic_settings(
    thinking_budget=16 * K_TOKENS,
    max_tokens=21 * K_TOKENS,
)
"""Anthropic medium thinking: 16K thinking budget, balanced reasoning."""

ANTHROPIC_LOW: dict[str, Any] = _anthropic_settings(
    thinking_budget=4 * K_TOKENS,
    max_tokens=8 * K_TOKENS,
)
"""Anthropic low thinking: 4K thinking budget, minimal reasoning overhead."""

ANTHROPIC_OFF: dict[str, Any] = _anthropic_off_settings()
"""Anthropic off: Thinking disabled, caching enabled."""

# -----------------------------------------------------------------------------
# Anthropic 1M context presets (with beta headers for extended context)
# -----------------------------------------------------------------------------

ANTHROPIC_1M_DEFAULT: dict[str, Any] = _anthropic_settings(
    thinking_budget=16 * K_TOKENS,
    max_tokens=16 * K_TOKENS,
    use_1m_context=True,
)
"""Anthropic 1M default: Same as medium, 16K thinking budget, with 1M context beta."""

ANTHROPIC_1M_HIGH: dict[str, Any] = _anthropic_settings(
    thinking_budget=32 * K_TOKENS,
    max_tokens=21 * K_TOKENS,
    use_1m_context=True,
)
"""Anthropic 1M high thinking: 32K thinking budget, max reasoning depth, with 1M context beta."""

ANTHROPIC_1M_MEDIUM: dict[str, Any] = _anthropic_settings(
    thinking_budget=16 * K_TOKENS,
    max_tokens=16 * K_TOKENS,
    use_1m_context=True,
)
"""Anthropic 1M medium thinking: 16K thinking budget, balanced reasoning, with 1M context beta."""

ANTHROPIC_1M_LOW: dict[str, Any] = _anthropic_settings(
    thinking_budget=4 * K_TOKENS,
    max_tokens=8 * K_TOKENS,
    use_1m_context=True,
)
"""Anthropic 1M low thinking: 4K thinking budget, minimal reasoning overhead, with 1M context beta."""

ANTHROPIC_1M_OFF: dict[str, Any] = _anthropic_off_settings(use_1m_context=True)
"""Anthropic 1M off: Thinking disabled, with 1M context beta and caching enabled."""


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
# Gemini thinking_budget Presets (for Gemini 2.5)
# =============================================================================


def _gemini_thinking_budget_settings(
    thinking_budget: int,
    max_tokens: int | None = None,
    include_thoughts: bool = False,
) -> dict[str, Any]:
    """Create Gemini model settings with thinking_budget only (for Gemini 2.5).

    Args:
        thinking_budget: Token budget for thinking.
        max_tokens: Maximum output tokens.
        include_thoughts: Whether to include thinking in response.

    Returns:
        Dict suitable for GoogleModelSettings.
    """
    settings: dict[str, Any] = {
        "google_thinking_config": {
            "thinking_budget": thinking_budget,
            "include_thoughts": include_thoughts,
        },
    }
    if max_tokens is not None:
        settings["max_tokens"] = max_tokens
    return settings


GEMINI_THINKING_BUDGET_DEFAULT: dict[str, Any] = _gemini_thinking_budget_settings(
    thinking_budget=16 * K_TOKENS,
    max_tokens=16 * K_TOKENS,
    include_thoughts=False,
)
"""Gemini 2.5 default: 16K thinking budget, balanced reasoning."""

GEMINI_THINKING_BUDGET_HIGH: dict[str, Any] = _gemini_thinking_budget_settings(
    thinking_budget=32 * K_TOKENS,
    max_tokens=21 * K_TOKENS,
    include_thoughts=False,
)
"""Gemini 2.5 high: 32K thinking budget, maximum reasoning depth."""

GEMINI_THINKING_BUDGET_MEDIUM: dict[str, Any] = _gemini_thinking_budget_settings(
    thinking_budget=16 * K_TOKENS,
    max_tokens=16 * K_TOKENS,
    include_thoughts=False,
)
"""Gemini 2.5 medium: 16K thinking budget, balanced reasoning."""

GEMINI_THINKING_BUDGET_LOW: dict[str, Any] = _gemini_thinking_budget_settings(
    thinking_budget=4 * K_TOKENS,
    max_tokens=8 * K_TOKENS,
    include_thoughts=False,
)
"""Gemini 2.5 low: 4K thinking budget, minimal reasoning overhead."""


# =============================================================================
# Gemini thinking_level Presets (for Gemini 3)
# =============================================================================


def _gemini_thinking_level_settings(
    thinking_level: Literal["HIGH", "MEDIUM", "LOW", "MINIMAL"],
    max_tokens: int | None = None,
    include_thoughts: bool = False,
) -> dict[str, Any]:
    """Create Gemini model settings with thinking_level only (for Gemini 3).

    Args:
        thinking_level: Thinking level ('HIGH', 'MEDIUM', 'LOW', 'MINIMAL').
        max_tokens: Maximum output tokens.
        include_thoughts: Whether to include thinking in response.

    Returns:
        Dict suitable for GoogleModelSettings.
    """
    settings: dict[str, Any] = {
        "google_thinking_config": {
            "thinking_level": thinking_level,
            "include_thoughts": include_thoughts,
        },
    }
    if max_tokens is not None:
        settings["max_tokens"] = max_tokens
    return settings


GEMINI_THINKING_LEVEL_DEFAULT: dict[str, Any] = _gemini_thinking_level_settings(
    thinking_level="LOW",
    max_tokens=16 * K_TOKENS,
    include_thoughts=False,
)
"""Gemini 3 default: MEDIUM thinking level, balanced reasoning."""

GEMINI_THINKING_LEVEL_HIGH: dict[str, Any] = _gemini_thinking_level_settings(
    thinking_level="HIGH",
    max_tokens=21 * K_TOKENS,
    include_thoughts=False,
)
"""Gemini 3 high: HIGH thinking level, maximum reasoning depth."""

GEMINI_THINKING_LEVEL_MEDIUM: dict[str, Any] = _gemini_thinking_level_settings(
    thinking_level="MEDIUM",
    max_tokens=16 * K_TOKENS,
    include_thoughts=False,
)
"""Gemini 3 medium: MEDIUM thinking level, balanced reasoning."""

GEMINI_THINKING_LEVEL_LOW: dict[str, Any] = _gemini_thinking_level_settings(
    thinking_level="LOW",
    max_tokens=8 * K_TOKENS,
    include_thoughts=False,
)
"""Gemini 3 low: LOW thinking level, minimal reasoning overhead."""

GEMINI_THINKING_LEVEL_MINIMAL: dict[str, Any] = _gemini_thinking_level_settings(
    thinking_level="MINIMAL",
    max_tokens=4 * K_TOKENS,
    include_thoughts=False,
)
"""Gemini 3 minimal: MINIMAL thinking level (Flash only, may still think for complex tasks)."""


# =============================================================================
# Preset Registry
# =============================================================================

_PRESET_REGISTRY: dict[str, dict[str, Any]] = {
    # Anthropic standard (no beta headers)
    ModelSettingsPreset.ANTHROPIC_DEFAULT.value: ANTHROPIC_DEFAULT,
    ModelSettingsPreset.ANTHROPIC_HIGH.value: ANTHROPIC_HIGH,
    ModelSettingsPreset.ANTHROPIC_MEDIUM.value: ANTHROPIC_MEDIUM,
    ModelSettingsPreset.ANTHROPIC_LOW.value: ANTHROPIC_LOW,
    ModelSettingsPreset.ANTHROPIC_OFF.value: ANTHROPIC_OFF,
    # Anthropic 1M context (with beta headers)
    ModelSettingsPreset.ANTHROPIC_1M_DEFAULT.value: ANTHROPIC_1M_DEFAULT,
    ModelSettingsPreset.ANTHROPIC_1M_HIGH.value: ANTHROPIC_1M_HIGH,
    ModelSettingsPreset.ANTHROPIC_1M_MEDIUM.value: ANTHROPIC_1M_MEDIUM,
    ModelSettingsPreset.ANTHROPIC_1M_LOW.value: ANTHROPIC_1M_LOW,
    ModelSettingsPreset.ANTHROPIC_1M_OFF.value: ANTHROPIC_1M_OFF,
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
    # Gemini thinking_budget (for Gemini 2.5)
    ModelSettingsPreset.GEMINI_THINKING_BUDGET_DEFAULT.value: GEMINI_THINKING_BUDGET_DEFAULT,
    ModelSettingsPreset.GEMINI_THINKING_BUDGET_HIGH.value: GEMINI_THINKING_BUDGET_HIGH,
    ModelSettingsPreset.GEMINI_THINKING_BUDGET_MEDIUM.value: GEMINI_THINKING_BUDGET_MEDIUM,
    ModelSettingsPreset.GEMINI_THINKING_BUDGET_LOW.value: GEMINI_THINKING_BUDGET_LOW,
    # Gemini thinking_level (for Gemini 3)
    ModelSettingsPreset.GEMINI_THINKING_LEVEL_DEFAULT.value: GEMINI_THINKING_LEVEL_DEFAULT,
    ModelSettingsPreset.GEMINI_THINKING_LEVEL_HIGH.value: GEMINI_THINKING_LEVEL_HIGH,
    ModelSettingsPreset.GEMINI_THINKING_LEVEL_MEDIUM.value: GEMINI_THINKING_LEVEL_MEDIUM,
    ModelSettingsPreset.GEMINI_THINKING_LEVEL_LOW.value: GEMINI_THINKING_LEVEL_LOW,
    ModelSettingsPreset.GEMINI_THINKING_LEVEL_MINIMAL.value: GEMINI_THINKING_LEVEL_MINIMAL,
}

# Short aliases for convenience
_PRESET_ALIASES: dict[str, str] = {
    # Provider defaults (default preset)
    "anthropic": ModelSettingsPreset.ANTHROPIC_DEFAULT.value,
    "anthropic_1m": ModelSettingsPreset.ANTHROPIC_1M_DEFAULT.value,
    "openai": ModelSettingsPreset.OPENAI_DEFAULT.value,
    "openai_responses": ModelSettingsPreset.OPENAI_RESPONSES_DEFAULT.value,
    "gemini_2.5": ModelSettingsPreset.GEMINI_THINKING_BUDGET_DEFAULT.value,
    "gemini_3": ModelSettingsPreset.GEMINI_THINKING_LEVEL_DEFAULT.value,
    "gemini": ModelSettingsPreset.GEMINI_THINKING_LEVEL_DEFAULT.value,  # Default to Gemini 3
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
    preset_or_dict: ModelSettings | str | dict[str, Any] | ModelSettingsPreset | None,
) -> dict[str, Any] | None:
    """Resolve a preset name or dict to ModelSettings.

    This is the main entry point for resolving model settings from various formats:
    - None -> None (use model defaults)
    - str -> lookup preset by name
    - ModelSettingsPreset -> lookup preset by enum
    - dict -> return as-is (assumed to be valid ModelSettings)
    - ModelSettings -> convert to dict using model_dump()

    Args:
        preset_or_dict: Preset name, enum, dict, ModelSettings, or None.

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
    if isinstance(preset_or_dict, str):
        return get_model_settings(preset_or_dict)
    if isinstance(preset_or_dict, ModelSettingsPreset):
        return get_model_settings(preset_or_dict)
    # ModelSettings is a TypedDict (subclass of dict), return as-is
    return dict(preset_or_dict)


def list_presets() -> list[str]:
    """List all available preset names.

    Returns:
        List of preset names (including aliases).
    """
    return sorted(set(_PRESET_REGISTRY.keys()) | set(_PRESET_ALIASES.keys()))


# =============================================================================
# ModelConfig Presets
# =============================================================================

# Special value indicating inheritance from parent
INHERIT = "inherit"


class ModelConfigPreset(str, Enum):
    """Available ModelConfig presets for context management."""

    # Anthropic models
    CLAUDE_200K = "claude_200k"
    CLAUDE_1M = "claude_1m"

    # OpenAI models (GPT-5 series with 270k context)
    GPT5_270K = "gpt5_270k"

    # Gemini models
    GEMINI_200K = "gemini_200k"
    GEMINI_1M = "gemini_1m"


# ModelConfig preset registry
_MODEL_CFG_REGISTRY: dict[str, dict[str, Any]] = {
    # Anthropic Claude models (vision, no video support)
    ModelConfigPreset.CLAUDE_200K.value: {
        "context_window": 200_000,
        "max_images": 20,
        "max_videos": 0,  # Claude doesn't support video
        "support_gif": True,
        "capabilities": {ModelCapability.vision, ModelCapability.document_understanding},
    },
    ModelConfigPreset.CLAUDE_1M.value: {
        "context_window": 1_000_000,
        "max_images": 20,
        "max_videos": 0,  # Claude doesn't support video
        "support_gif": True,
        "capabilities": {ModelCapability.vision, ModelCapability.document_understanding},
    },
    # OpenAI GPT-5 series (vision, no video support)
    ModelConfigPreset.GPT5_270K.value: {
        "context_window": 270_000,
        "max_images": 20,
        "max_videos": 0,  # GPT doesn't support video
        "support_gif": False,
        "capabilities": {ModelCapability.vision},
    },
    # Gemini models (vision + video support)
    ModelConfigPreset.GEMINI_200K.value: {
        "context_window": 200_000,
        "max_images": 20,
        "max_videos": 1,  # Gemini supports video
        "support_gif": True,
        "capabilities": {
            ModelCapability.vision,
            ModelCapability.video_understanding,
            ModelCapability.document_understanding,
        },
    },
    ModelConfigPreset.GEMINI_1M.value: {
        "context_window": 1_000_000,
        "max_images": 20,
        "max_videos": 1,  # Gemini supports video
        "support_gif": True,
        "capabilities": {
            ModelCapability.vision,
            ModelCapability.video_understanding,
            ModelCapability.document_understanding,
        },
    },
}

# ModelConfig aliases
_MODEL_CFG_ALIASES: dict[str, str] = {
    "claude": ModelConfigPreset.CLAUDE_200K.value,
    "anthropic": ModelConfigPreset.CLAUDE_200K.value,
    "gpt5": ModelConfigPreset.GPT5_270K.value,
    "openai": ModelConfigPreset.GPT5_270K.value,
    "gemini": ModelConfigPreset.GEMINI_200K.value,
}


def get_model_cfg(preset: str | ModelConfigPreset) -> dict[str, Any]:
    """Get ModelConfig by preset name.

    Args:
        preset: Preset name (string) or ModelConfigPreset enum.

    Returns:
        Dict suitable for ModelConfig constructor.

    Raises:
        ValueError: If preset name is not found.

    Example::

        # By string name
        cfg = get_model_cfg("claude_200k")

        # By enum
        cfg = get_model_cfg(ModelConfigPreset.GEMINI_1M)

        # By alias
        cfg = get_model_cfg("claude")  # -> claude_200k
    """
    name = preset.value if isinstance(preset, ModelConfigPreset) else preset

    # Check aliases first
    if name in _MODEL_CFG_ALIASES:
        name = _MODEL_CFG_ALIASES[name]

    if name not in _MODEL_CFG_REGISTRY:
        available = list(_MODEL_CFG_REGISTRY.keys()) + list(_MODEL_CFG_ALIASES.keys())
        msg = f"Unknown ModelConfig preset: {preset!r}. Available: {sorted(available)}"
        raise ValueError(msg)

    return _MODEL_CFG_REGISTRY[name].copy()


def resolve_model_cfg(
    preset_or_dict: str | dict[str, Any] | ModelConfigPreset | None,
) -> dict[str, Any] | None:
    """Resolve a preset name or dict to ModelConfig dict.

    This is the main entry point for resolving ModelConfig from various formats:
    - None -> None (inherit from parent)
    - "inherit" -> None (explicit inherit)
    - str -> lookup preset by name
    - ModelConfigPreset -> lookup preset by enum
    - dict -> return as-is (assumed to be valid ModelConfig kwargs)

    Args:
        preset_or_dict: Preset name, enum, dict, or None.

    Returns:
        Dict suitable for ModelConfig constructor, or None for inherit.

    Example::

        # Inherit from parent (default)
        cfg = resolve_model_cfg(None)  # -> None
        cfg = resolve_model_cfg("inherit")  # -> None

        # From preset name
        cfg = resolve_model_cfg("claude_200k")

        # From dict
        cfg = resolve_model_cfg({"context_window": 100000, "max_images": 10})
    """
    if preset_or_dict is None:
        return None
    if isinstance(preset_or_dict, str):
        if preset_or_dict == INHERIT:
            return None
        return get_model_cfg(preset_or_dict)
    if isinstance(preset_or_dict, ModelConfigPreset):
        return get_model_cfg(preset_or_dict)
    # dict - return as-is
    return dict(preset_or_dict)


def list_model_cfg_presets() -> list[str]:
    """List all available ModelConfig preset names.

    Returns:
        List of preset names (including aliases).
    """
    return sorted(set(_MODEL_CFG_REGISTRY.keys()) | set(_MODEL_CFG_ALIASES.keys()))
