"""Streaming module for paintress-cli.

This module provides streaming utilities for the TUI:
- TextStreamer: Manages streaming text accumulation and rendering
- ThinkingStreamer: Manages streaming thinking content
- SubagentTracker: Tracks subagent execution progress
- StreamEventHandler: Dispatches stream events to handlers

Example:
    from paintress_cli.streaming import TextStreamer, SubagentTracker

    # Text streaming
    streamer = TextStreamer(renderer, code_theme="monokai")
    streamer.start("Initial text", line_index=5)
    rendered = streamer.update(" more text")
    final = streamer.finalize()

    # Subagent tracking
    tracker = SubagentTracker(renderer)
    line = tracker.start("explorer", "Explorer Agent", line_index=10)
    tracker.add_tool("explorer", "grep")
    summary, idx = tracker.complete("explorer", success=True, duration_seconds=2.5)
"""

from __future__ import annotations

from paintress_cli.streaming.event_handler import (
    EVENT_TYPES,
    StreamEventHandler,
    is_text_delta,
    is_text_start,
    is_thinking_delta,
    is_thinking_start,
)
from paintress_cli.streaming.subagent_tracker import SubagentState, SubagentTracker
from paintress_cli.streaming.text_streamer import TextStreamer, ThinkingStreamer

__all__ = [
    "EVENT_TYPES",
    # Event handling
    "StreamEventHandler",
    # Subagent tracking
    "SubagentState",
    "SubagentTracker",
    # Text streaming
    "TextStreamer",
    "ThinkingStreamer",
    "is_text_delta",
    "is_text_start",
    "is_thinking_delta",
    "is_thinking_start",
]
