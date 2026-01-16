"""Toolsets for pai-agent-sdk.

This module provides:
- BaseTool: Abstract base class for individual tools
- BaseToolset: Abstract base class for toolsets with instruction support
- Toolset: Container for tools with hooks and HITL support
- InstructableToolset: Protocol for toolsets that provide instructions
- BrowserUseToolset: Browser automation via Chrome DevTools Protocol
"""

from pai_agent_sdk.toolsets.base import (
    BaseTool,
    BaseToolset,
    InstructableToolset,
    UserInputPreprocessResult,
    resolve_instructions,
)
from pai_agent_sdk.toolsets.browser_use import BrowserUseSettings, BrowserUseToolset
from pai_agent_sdk.toolsets.core.base import (
    CallMetadata,
    GlobalHooks,
    GlobalPostHookFunc,
    GlobalPreHookFunc,
    HookableToolsetTool,
    PostHookFunc,
    PreHookFunc,
    Toolset,
    UserInteraction,
)

__all__ = [
    "BaseTool",
    "BaseToolset",
    "BrowserUseSettings",
    "BrowserUseToolset",
    "CallMetadata",
    "GlobalHooks",
    "GlobalPostHookFunc",
    "GlobalPreHookFunc",
    "HookableToolsetTool",
    "InstructableToolset",
    "PostHookFunc",
    "PreHookFunc",
    "Toolset",
    "UserInputPreprocessResult",
    "UserInteraction",
    "resolve_instructions",
]
