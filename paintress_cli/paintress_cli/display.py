"""Display components for TUI rendering.

This module provides Rich-based rendering for the TUI output pane:
- RichRenderer: Convert Rich renderables to ANSI strings
- EventRenderer: Render agent stream events to display strings
- ToolCallTracker: Track tool call states for display

Example:
    renderer = RichRenderer(width=120)
    text = renderer.render(Markdown("# Hello"))
    print(text)  # ANSI-formatted output
"""

from __future__ import annotations

import enum
import json
import time
from io import StringIO
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    pass


# =============================================================================
# Enums
# =============================================================================


class ToolCallState(str, enum.Enum):
    """Tool call execution state."""

    CALLING = "calling"
    COMPLETE = "complete"
    RENDERED = "rendered"


class RenderDirective(str, enum.Enum):
    """Render directive for display updates."""

    CALLING = "calling"
    COMPLETE = "complete"
    TEXT = "text"
    RESULT = "result"


# =============================================================================
# Tool Call Tracking
# =============================================================================


class ToolCallInfo(BaseModel):
    """Information about a single tool call."""

    tool_call_id: str
    name: str
    args: str | dict[str, Any] | None = None
    state: ToolCallState
    start_time: float
    end_time: float | None = None
    result: Any | None = None

    def duration(self) -> float:
        """Calculate execution duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def is_special_tool(self) -> bool:
        """Check if this is a special tool requiring detailed panel."""
        return self.name in {"edit", "thinking", "to_do_read", "to_do_write", "multi_edit"}


class ToolCallTracker:
    """Track tool call states and manage rendering lifecycle."""

    def __init__(self) -> None:
        self.tool_calls: dict[str, ToolCallInfo] = {}
        self.call_order: list[str] = []

    def start_call(self, tool_call_id: str, name: str, args: str | dict[str, Any] | None = None) -> None:
        """Register a new tool call."""
        self.tool_calls[tool_call_id] = ToolCallInfo(
            tool_call_id=tool_call_id,
            name=name,
            args=args,
            state=ToolCallState.CALLING,
            start_time=time.time(),
        )
        self.call_order.append(tool_call_id)

    def complete_call(self, tool_call_id: str, result: Any = None) -> None:
        """Mark tool call as complete."""
        if tool_call_id in self.tool_calls:
            info = self.tool_calls[tool_call_id]
            info.state = ToolCallState.COMPLETE
            info.end_time = time.time()
            info.result = result

    def mark_rendered(self, tool_call_id: str) -> None:
        """Mark tool call as rendered."""
        if tool_call_id in self.tool_calls:
            self.tool_calls[tool_call_id].state = ToolCallState.RENDERED

    def get_calling_tools(self) -> list[ToolCallInfo]:
        """Get tools in CALLING state."""
        return [
            self.tool_calls[tid]
            for tid in self.call_order
            if tid in self.tool_calls and self.tool_calls[tid].state == ToolCallState.CALLING
        ]

    def get_completed_tools(self) -> list[ToolCallInfo]:
        """Get tools in COMPLETE state (ready to render)."""
        return [
            self.tool_calls[tid]
            for tid in self.call_order
            if tid in self.tool_calls and self.tool_calls[tid].state == ToolCallState.COMPLETE
        ]

    def has_active_calls(self) -> bool:
        """Check if there are any active tool calls."""
        return len(self.get_calling_tools()) > 0 or len(self.get_completed_tools()) > 0

    def clear(self) -> None:
        """Clear all tracked tool calls."""
        self.tool_calls.clear()
        self.call_order.clear()


# =============================================================================
# Rich Renderer
# =============================================================================


class RichRenderer:
    """Convert Rich renderables to ANSI strings for output buffer."""

    def __init__(self, width: int | None = None) -> None:
        self._width = width or 120

    def render(self, renderable: Any, width: int | None = None) -> str:
        """Render Rich object to ANSI string.

        Args:
            renderable: Rich renderable object
            width: Optional width override.
        """
        render_width = width or self._width
        string_io = StringIO()
        console = Console(
            file=string_io,
            force_terminal=True,
            width=render_width,
            no_color=False,
        )
        console.print(renderable)
        return string_io.getvalue()

    def render_markdown(self, text: str, code_theme: str = "monokai") -> str:
        """Render markdown text to ANSI string."""
        return self.render(Markdown(text, code_theme=code_theme))

    def render_text(self, text: str, style: str | None = None) -> str:
        """Render styled text to ANSI string."""
        return self.render(Text(text, style=style or ""))

    def render_panel(
        self,
        content: str | Any,
        title: str | None = None,
        border_style: str = "blue",
    ) -> str:
        """Render a panel to ANSI string."""
        return self.render(Panel(content, title=title, border_style=border_style))


# =============================================================================
# Tool Message Display
# =============================================================================


class ToolMessage(BaseModel):
    """Tool message for display formatting."""

    tool_call_id: str
    name: str
    args: str | dict[str, Any] | None = None
    content: str | None = None

    def _format_args_text(self, max_arg_length: int) -> str:
        """Format tool arguments for display."""
        if not self.args:
            return ""
        try:
            if isinstance(self.args, dict):
                args_lines = []
                for key, value in self.args.items():
                    str_value = str(value)
                    if len(str_value) > max_arg_length:
                        str_value = str_value[:max_arg_length] + "..."
                    args_lines.append(f"{key}: {str_value}")
                return "\n".join(args_lines[:3]) + "\n..." if len(args_lines) > 3 else "\n".join(args_lines)
            str_args = str(self.args)
            if len(str_args) > max_arg_length:
                str_args = str_args[:max_arg_length] + "..."
            return f"args: {str_args}"
        except Exception:
            return "args: <complex>"

    def _format_result_text(self, max_result_lines: int, max_arg_length: int) -> str:
        """Format tool result for display."""
        if not self.content:
            return ""
        try:
            content_lines = self.content.split("\n")
            if len(content_lines) > max_result_lines:
                result_text = "\n".join(content_lines[:max_result_lines]) + "\n[...]"
            else:
                result_text = self.content
        except Exception:
            result_text = "<result available>"
        return result_text[:max_arg_length]

    def to_panel(
        self,
        max_result_lines: int = 2,
        max_arg_length: int = 120,
    ) -> Panel:
        """Convert the tool message to a Rich Panel object."""
        args_text = self._format_args_text(max_arg_length)
        result_text = self._format_result_text(max_result_lines, max_arg_length)

        # Combine args and result
        panel_content = args_text
        if result_text:
            if panel_content:
                panel_content += "\n\n"
            panel_content += result_text

        if not panel_content:
            panel_content = "No details available"

        return Panel(panel_content, title=f"[TOOL] {self.name}", title_align="left")

    def to_special_panel(self, code_theme: str = "monokai") -> Panel:
        """Create special panel for to_do and thinking tools."""
        if self.name in ["to_do_read", "to_do_write"]:
            return self._create_to_do_panel()
        elif self.name == "thinking":
            return self._create_thinking_panel(code_theme)
        elif self.name in ["edit", "multi_edit"]:
            return self._create_edit_panel(code_theme)
        else:
            return self.to_panel()

    def _create_to_do_panel(self) -> Panel:
        """Create a special panel for to_do tools."""
        panel_content = ""
        try:
            if self.content:
                if self.content.startswith("[") and self.content.endswith("]"):
                    to_dos = json.loads(self.content)
                    if isinstance(to_dos, list) and to_dos:
                        panel_content = self._format_to_do_list(to_dos)
                    else:
                        panel_content = "No to_dos found"
                elif isinstance(self.content, str):
                    panel_content = self.content
                else:
                    panel_content = str(self.content)
            else:
                panel_content = "No to_do data available"
        except json.JSONDecodeError:
            panel_content = f"Raw content:\n{self.content}"
        except Exception:
            panel_content = "Error processing to_do data"

        return Panel(
            panel_content,
            title=f"[TOOL] {self.name}",
            title_align="left",
            border_style="blue",
        )

    def _create_thinking_panel(self, code_theme: str = "monokai") -> Panel:
        """Create a special panel for thinking tools."""
        panel_content: Any = ""
        try:
            if self.content:
                if self.content.startswith("{") and self.content.endswith("}"):
                    thinking_data = json.loads(self.content)
                    if isinstance(thinking_data, dict) and "thought" in thinking_data:
                        thought = thinking_data["thought"]
                        panel_content = Markdown(thought, code_theme=code_theme)
                    else:
                        panel_content = Markdown(f"```json\n{self.content}\n```", code_theme=code_theme)
                else:
                    panel_content = Markdown(self.content, code_theme=code_theme)
            else:
                panel_content = "No thinking content available"
        except json.JSONDecodeError:
            panel_content = Markdown(self.content or "", code_theme=code_theme)
        except Exception:
            panel_content = "Error processing thinking data"

        return Panel(
            panel_content,
            title=f"[TOOL] {self.name}",
            title_align="left",
            border_style="magenta",
        )

    def _create_edit_panel(self, code_theme: str = "monokai") -> Panel:
        """Create a special panel for edit tools."""
        try:
            args_dict = self._parse_edit_args()
            if args_dict:
                file_path = args_dict.get("file_path", "unknown")
                content_parts = [Text(f"Editing: {file_path}", style="bold cyan")]

                # Show result if available
                if self.content:
                    content_parts.append(Text(""))
                    result_text = Text(f"Result: {self.content}", style="bold green")
                    content_parts.append(result_text)

                rendered_content = Group(*content_parts)
            else:
                rendered_content = Text("No edit parameters available")
        except Exception as e:
            rendered_content = Text(f"Error processing edit data: {e!s}")

        return Panel(
            rendered_content,
            title=f"[TOOL] {self.name}",
            title_align="left",
            border_style="green",
        )

    def _parse_edit_args(self) -> dict:
        """Parse the edit tool arguments into a dictionary."""
        args_dict = {}
        if self.args:
            if isinstance(self.args, dict):
                args_dict = self.args
            elif isinstance(self.args, str):
                try:
                    args_dict = json.loads(self.args)
                except json.JSONDecodeError:
                    args_dict = {"raw_args": self.args}
        return args_dict

    def _format_to_do_list(self, to_dos: list) -> str:
        """Format a list of to_dos for display."""
        if not to_dos:
            return "No to_dos found"

        priority_groups: dict[str, list[dict]] = {"high": [], "medium": [], "low": []}
        for to_do in to_dos:
            if isinstance(to_do, dict):
                priority = to_do.get("priority", "medium")
                if priority in priority_groups:
                    priority_groups[priority].append(to_do)

        lines = []
        for priority in ["high", "medium", "low"]:
            items = priority_groups[priority]
            if items:
                lines.append(f"{priority.title()} Priority ({len(items)} items)")
                for item in items:
                    status = item.get("status", "pending")
                    content = item.get("content", "No content")
                    item_id = item.get("id", "")
                    status_text = {
                        "pending": "[ ]",
                        "in_progress": "[*]",
                        "completed": "[v]",
                    }.get(status, f"[{status}]")
                    if item_id:
                        lines.append(f"  {status_text} {item_id}: {content}")
                    else:
                        lines.append(f"  {status_text} {content}")
                lines.append("")

        # Progress summary
        total = len(to_dos)
        completed = sum(1 for to_do in to_dos if to_do.get("status") == "completed")
        in_progress = sum(1 for to_do in to_dos if to_do.get("status") == "in_progress")
        pending = sum(1 for to_do in to_dos if to_do.get("status") == "pending")

        if total > 0:
            completion_rate = int((completed / total) * 100)
            lines.append(f"Progress: {completed}/{total} completed ({completion_rate}%)")
            lines.append(f"Status: {in_progress} in progress, {pending} pending")

        return "\n".join(lines)


# =============================================================================
# Event Renderer
# =============================================================================


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

    def render_tool_call_start(self, name: str, tool_call_id: str) -> str:
        """Render tool call start indicator."""
        return f"[Tool] {name}..."

    def render_tool_call_complete(self, tool_message: ToolMessage) -> str:
        """Render completed tool call as panel."""
        if tool_message.name in {"edit", "thinking", "to_do_read", "to_do_write", "multi_edit"}:
            panel = tool_message.to_special_panel(code_theme=self._code_theme)
        else:
            panel = tool_message.to_panel(
                max_result_lines=self._max_tool_result_lines,
                max_arg_length=self._max_arg_length,
            )
        return self._renderer.render(panel)

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
            preview = summary[:150] + "..." if len(summary) > 150 else summary
            content.append(f"\n{preview}", style="dim italic")
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
            preview = content[:250] + "..." if len(content) > 250 else content
            panel_content.append(preview, style="dim")
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

    def render_steering_injected(self, message_count: int, content: str) -> str:
        """Render steering message injected panel."""
        panel_content = Text()
        panel_content.append(f"Guidance injected ({message_count} message(s))\n", style="bold")
        if content:
            preview = content[:150] + "..." if len(content) > 150 else content
            panel_content.append(f'"{preview}"', style="dim italic")
        panel = Panel(panel_content, border_style="yellow", title="[yellow]Steering[/yellow]", title_align="left")
        return self._renderer.render(panel)
