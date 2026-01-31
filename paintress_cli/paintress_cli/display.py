"""Display components for TUI rendering.

This module re-exports all rendering components from the refactored
paintress_cli.rendering package for backward compatibility.

The actual implementations are now in:
- paintress_cli.rendering.types - Enums and data types
- paintress_cli.rendering.tracker - ToolCallTracker
- paintress_cli.rendering.renderer - RichRenderer, CachedRichRenderer
- paintress_cli.rendering.tool_message - ToolMessage
- paintress_cli.rendering.event_renderer - EventRenderer
- paintress_cli.rendering.tool_panels - Special tool panel rendering

Example:
    # Both of these work:
    from paintress_cli.display import EventRenderer, ToolMessage
    from paintress_cli.rendering import EventRenderer, ToolMessage
"""

from __future__ import annotations

# Re-export everything from the rendering module for backward compatibility
from paintress_cli.rendering import (
    CachedRichRenderer,
    EventRenderer,
    RenderDirective,
    RichRenderer,
    ToolCallInfo,
    ToolCallState,
    ToolCallTracker,
    ToolMessage,
)

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
