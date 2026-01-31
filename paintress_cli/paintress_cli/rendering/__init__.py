"""Rendering module for paintress-cli.

This module provides all rendering components for the TUI:
- Types and enums (ToolCallState, RenderDirective, ToolCallInfo)
- ToolCallTracker for tracking tool execution lifecycle
- RichRenderer for Rich-to-ANSI conversion
- ToolMessage for tool result formatting
- EventRenderer for agent event rendering

Example:
    from paintress_cli.rendering import EventRenderer, ToolMessage

    renderer = EventRenderer(width=120)

    # Render tool start
    output = renderer.render_tool_call_start("grep", "call-1")

    # Render tool completion
    msg = ToolMessage(tool_call_id="call-1", name="grep", content="Found 5 matches")
    output = renderer.render_tool_call_complete(msg, duration=0.5)
"""

from __future__ import annotations

from paintress_cli.rendering.event_renderer import EventRenderer
from paintress_cli.rendering.renderer import CachedRichRenderer, RichRenderer
from paintress_cli.rendering.tool_message import ToolMessage
from paintress_cli.rendering.tracker import ToolCallTracker
from paintress_cli.rendering.types import RenderDirective, ToolCallInfo, ToolCallState

__all__ = [
    "CachedRichRenderer",
    "EventRenderer",
    "RenderDirective",
    "RichRenderer",
    "ToolCallInfo",
    "ToolCallState",
    "ToolCallTracker",
    "ToolMessage",
]
