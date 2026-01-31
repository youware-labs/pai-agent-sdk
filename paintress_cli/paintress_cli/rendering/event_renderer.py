"""Event rendering for agent stream events.

Provides EventRenderer for rendering agent events to display-ready output.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.text import Text

from paintress_cli.rendering.renderer import RichRenderer
from paintress_cli.rendering.tool_message import ToolMessage
from paintress_cli.rendering.tracker import ToolCallTracker


class EventRenderer:
    """Render agent stream events to display-ready output.

    Tracks text streaming and tool calls, producing Rich-rendered output.
    """

    def __init__(
        self,
        width: int | None = None,
        code_theme: str = "monokai",
        max_tool_result_lines: int = 2,
        max_arg_length: int = 50,
    ) -> None:
        self._renderer = RichRenderer(width=width)
        self._code_theme = code_theme
        self._max_tool_result_lines = max_tool_result_lines
        self._max_arg_length = max_arg_length
        self._tracker = ToolCallTracker()
        self._tool_messages: dict[str, ToolMessage] = {}

        # Current streaming text
        self._current_text: str = ""
        self._current_thinking: str = ""

    @property
    def tracker(self) -> ToolCallTracker:
        """Get the tool call tracker."""
        return self._tracker

    def clear(self) -> None:
        """Clear all state for new conversation turn."""
        self._tracker.clear()
        self._tool_messages.clear()
        self._current_text = ""
        self._current_thinking = ""

    def get_current_text(self) -> str:
        """Get current accumulated text content."""
        return self._current_text

    def get_current_thinking(self) -> str:
        """Get current accumulated thinking content."""
        return self._current_thinking

    def update_thinking(self, delta: str) -> None:
        """Update current thinking content with delta."""
        self._current_thinking += delta

    def start_thinking(self, content: str = "") -> None:
        """Start a new thinking block."""
        self._current_thinking = content

    def render_thinking(self, content: str | None = None, width: int | None = None) -> str:
        """Render thinking content as a styled blockquote.

        Uses dim style with '>' prefix for each line to visually distinguish
        model's internal reasoning from regular output.

        Args:
            content: Optional content to render. If None, uses current thinking.
            width: Optional render width.

        Returns:
            Rendered ANSI string.
        """
        thinking_content = content if content is not None else self._current_thinking
        if not thinking_content:
            return ""

        # Format as blockquote with dim style
        lines = thinking_content.split("\n")
        text = Text()
        for i, line in enumerate(lines):
            if i > 0:
                text.append("\n")
            text.append("> ", style="dim magenta")
            text.append(line, style="dim italic")

        return self._renderer.render(text, width=width)

    def render_tool_call_start(self, name: str, tool_call_id: str) -> str:
        """Render tool call start indicator."""
        text = Text()
        text.append("Calling: ", style="dim")
        text.append(name, style="bold cyan")
        return self._renderer.render(text)

    def render_tool_call_complete(
        self,
        tool_message: ToolMessage,
        duration: float = 0.0,
        width: int | None = None,
    ) -> str:
        """Render completed tool call.

        Special tools (edit, thinking, to_do) use Panel format.
        Normal tools use inline Text format for cleaner display.
        """
        render_width = width or 120
        if tool_message.name in {
            "edit",
            "thinking",
            "to_do_read",
            "to_do_write",
            "multi_edit",
            "task_create",
            "task_get",
            "task_update",
            "task_list",
        }:
            panel = tool_message.to_special_panel(code_theme=self._code_theme)
            return self._renderer.render(panel, width=render_width)
        else:
            # Use inline text format for normal tools
            # Calculate available width for args/output (reserve space for labels)
            max_line_len = max(50, render_width - 20)
            text = tool_message.to_inline_text(
                duration=duration,
                max_arg_length=min(self._max_arg_length, max_line_len),
                max_result_lines=self._max_tool_result_lines,
                max_line_length=max_line_len,
            )
            return self._renderer.render(text, width=render_width)

    def render_markdown(self, text: str) -> str:
        """Render markdown text."""
        return self._renderer.render_markdown(text, code_theme=self._code_theme)

    def render_text(self, text: str, style: str | None = None) -> str:
        """Render styled text."""
        return self._renderer.render_text(text, style=style)

    # =========================================================================
    # Event Panel Rendering
    # =========================================================================

    def render_compact_start(self, message_count: int) -> str:
        """Render compact start notification (single line)."""
        text = Text()
        text.append("> ", style="cyan")
        text.append(f"Context compacting {message_count} messages...", style="dim")
        return self._renderer.render(text)

    def render_compact_complete(
        self,
        original_count: int,
        compacted_count: int,
        summary: str = "",
    ) -> str:
        """Render compact complete panel."""
        reduction = int((1 - compacted_count / original_count) * 100) if original_count > 0 else 0
        content = Text()
        content.append(f"{original_count} -> {compacted_count} messages ", style="bold")
        content.append(f"({reduction}% reduction)", style="dim")
        if summary:
            content.append(summary, style="dim italic")
        panel = Panel(content, border_style="cyan", title="[cyan]Context Compacted[/cyan]", title_align="left")
        return self._renderer.render(panel)

    def render_compact_failed(self, error: str) -> str:
        """Render compact failed notification (single line)."""
        text = Text()
        text.append("x ", style="red")
        text.append("Compact failed: ", style="bold red")
        text.append(error[:100], style="dim")
        return self._renderer.render(text)

    def render_handoff_start(self, message_count: int) -> str:
        """Render handoff start notification (single line)."""
        text = Text()
        text.append("> ", style="magenta")
        text.append(f"Preparing context handoff from {message_count} messages...", style="dim")
        return self._renderer.render(text)

    def render_handoff_complete(self, content: str) -> str:
        """Render handoff complete panel."""
        panel_content = Text()
        panel_content.append("Context reset with preserved state\n", style="bold green")
        if content:
            panel_content.append(content, style="dim")
        panel = Panel(
            panel_content, border_style="magenta", title="[magenta]Handoff Complete[/magenta]", title_align="left"
        )
        return self._renderer.render(panel)

    def render_handoff_failed(self, error: str) -> str:
        """Render handoff failed notification (single line)."""
        text = Text()
        text.append("x ", style="red")
        text.append("Handoff failed: ", style="bold red")
        text.append(error[:100], style="dim")
        return self._renderer.render(text)

    def render_steering_injected(self, messages: list[str], max_line_len: int = 100) -> str:
        """Render steering message injected panel."""
        panel_content = Text()
        panel_content.append(f"Guidance injected ({len(messages)} message(s))\n", style="bold")
        for msg in messages:
            line = msg.replace("\n", " ")
            preview = line[:max_line_len] + "..." if len(line) > max_line_len else line
            panel_content.append(f'"{preview}"\n', style="dim italic")
        panel = Panel(panel_content, border_style="yellow", title="[yellow]Steering[/yellow]", title_align="left")
        return self._renderer.render(panel)
