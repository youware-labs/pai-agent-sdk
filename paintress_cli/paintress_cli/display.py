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

import difflib
import enum
import json
import time
from io import StringIO
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
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

    def render_markdown(self, text: str, code_theme: str = "monokai", width: int | None = None) -> str:
        """Render markdown text to ANSI string."""
        return self.render(Markdown(text, code_theme=code_theme), width=width)

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

    def to_inline_text(
        self,
        duration: float = 0.0,
        max_arg_length: int = 80,
        max_result_lines: int = 8,
        max_line_length: int = 100,
    ) -> Text:
        """Convert the tool message to inline Rich Text format (no panel border).

        Format:
            Complete: tool_name (0.1s) | Args: {...} | Output: {...}
        """
        # Check if tool execution failed
        is_error = self.content and self.content.startswith("Tool execution error")

        content = Text()
        if is_error:
            content.append("x Error: ", style="dim")
            content.append(self.name, style="bold red")
        else:
            content.append("Complete: ", style="dim")
            content.append(self.name, style="bold green")
        content.append(f" ({duration:.1f}s)", style="dim")

        # Add args preview (inline, no newline)
        args_preview, args_truncated = self._format_inline_args_with_info(max_arg_length)
        if args_preview:
            content.append(" | ", style="dim")
            content.append("Args: ", style="dim cyan")
            content.append(args_preview, style="dim")
            if args_truncated > 0:
                content.append(f" (+{args_truncated} chars)", style="dim italic")

        # Add output preview (inline, no newline)
        output_preview, output_truncated = self._format_inline_output_with_info(max_result_lines, max_line_length)
        if output_preview and output_preview != "[no output]":
            content.append(" | ", style="dim")
            if is_error:
                content.append("Error: ", style="dim red")
            else:
                content.append("Output: ", style="dim yellow")
            content.append(output_preview, style="dim")
            if output_truncated > 0:
                content.append(f" (+{output_truncated} chars)", style="dim italic")

        return content

    def _format_inline_args(self, max_length: int = 80) -> str:
        """Format tool arguments for inline display."""
        result, _ = self._format_inline_args_with_info(max_length)
        return result

    def _format_inline_args_with_info(self, max_length: int = 80) -> tuple[str, int]:
        """Format tool arguments for inline display with truncation info.

        Returns:
            Tuple of (formatted_args, truncated_chars)
        """
        if not self.args:
            return "", 0
        try:
            if isinstance(self.args, dict):
                args_str = json.dumps(self.args, ensure_ascii=False)
            else:
                args_str = str(self.args)
            if len(args_str) > max_length:
                return args_str[: max_length - 3] + "...", len(args_str) - max_length + 3
            return args_str, 0
        except Exception:
            s = str(self.args)
            if len(s) > max_length:
                return s[:max_length], len(s) - max_length
            return s, 0

    def _format_inline_output(self, max_lines: int = 8, max_line_length: int = 100) -> str:
        """Format tool output for inline display."""
        result, _ = self._format_inline_output_with_info(max_lines, max_line_length)
        return result

    def _format_inline_output_with_info(self, max_lines: int = 8, max_line_length: int = 100) -> tuple[str, int]:
        """Format tool output for inline display with truncation info.

        Returns:
            Tuple of (formatted_output, truncated_chars)
        """
        if not self.content:
            return "[no output]", 0

        original_len = len(self.content)
        try:
            # Try to parse as JSON for prettier formatting
            try:
                parsed = json.loads(self.content)
                if isinstance(parsed, dict):
                    lines = []
                    for key, value in list(parsed.items())[:3]:  # Limit to 3 fields
                        str_value = str(value)
                        if len(str_value) > max_line_length:
                            str_value = str_value[: max_line_length - 3] + "..."
                        lines.append(f"{key}: {str_value}")
                    result = "{" + ", ".join(lines) + "}"
                    truncated = original_len - len(result) if len(parsed) > 3 else 0
                    return result, max(0, truncated)
            except (json.JSONDecodeError, TypeError):
                pass

            # Plain text handling - single line preview
            first_line = self.content.split("\n")[0]
            if len(first_line) > max_line_length:
                result = first_line[: max_line_length - 3] + "..."
                truncated = original_len - max_line_length + 3
            else:
                result = first_line
                truncated = original_len - len(first_line) if "\n" in self.content else 0
            return result, max(0, truncated)
        except Exception:
            s = str(self.content)[:max_line_length]
            return s, max(0, original_len - len(s))

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
        """Create a special panel for to_do tools.

        Both to_do_read and to_do_write return JSON string in content.
        """
        panel_content = ""
        try:
            if self.content and self.content.startswith("["):
                to_dos = json.loads(self.content)
                if isinstance(to_dos, list) and to_dos:
                    panel_content = self._format_to_do_list(to_dos)
                else:
                    panel_content = "No to_dos found"
            elif self.content:
                panel_content = str(self.content)
            else:
                panel_content = "No to_do data available"
        except json.JSONDecodeError:
            panel_content = str(self.content) if self.content else "Invalid to_do data"

        return Panel(
            panel_content,
            title=f"[TOOL] {self.name}",
            title_align="left",
            border_style="blue",
        )

    def _create_thinking_panel(self, code_theme: str = "monokai") -> Panel:
        """Create a special panel for thinking tools.

        The thought content is in args['thought'], not in the result content.
        """
        panel_content: Any = ""
        try:
            # Extract thought from args (where thinking tool stores the content)
            thought = None
            if isinstance(self.args, dict) and "thought" in self.args:
                thought = self.args["thought"]
            elif isinstance(self.args, str):
                # Try parsing args as JSON string
                try:
                    args_data = json.loads(self.args)
                    if isinstance(args_data, dict) and "thought" in args_data:
                        thought = args_data["thought"]
                except json.JSONDecodeError:
                    pass

            if thought:
                panel_content = Markdown(thought, code_theme=code_theme)
            else:
                panel_content = "No thinking content available"
        except Exception:
            panel_content = "Error processing thinking data"

        return Panel(
            panel_content,
            title=f"[TOOL] {self.name}",
            title_align="left",
            border_style="magenta",
        )

    def _create_edit_panel(self, code_theme: str = "monokai") -> Panel:
        """Create a special panel for edit tools with diff display."""
        try:
            args_dict = self._parse_edit_args()

            if args_dict:
                file_path = args_dict.get("file_path", "unknown")
                edits_list = args_dict.get("edits", [])

                content_parts = []

                # Add file header
                file_info = Text(f"Editing file: {file_path}", style="bold cyan")
                content_parts.append(file_info)

                if edits_list:
                    # Add edit summary for multi_edit
                    summary = self._create_edit_summary(edits_list)
                    content_parts.append(summary)
                    content_parts.append(Text(""))

                    # Format and add edit sequence
                    edit_sequence = self._format_edit_sequence(edits_list, code_theme)
                    content_parts.extend(edit_sequence)
                else:
                    # Handle single edit format
                    old_string = args_dict.get("old_string", "")
                    new_string = args_dict.get("new_string", "")

                    if old_string or new_string:
                        if not old_string:
                            # New file creation
                            if new_string:
                                lines = new_string.split("\n")
                                content_preview = (
                                    "\n".join(lines[:20]) + "\n...(truncated)" if len(lines) > 20 else new_string
                                )
                                syntax_content = Syntax(
                                    content_preview,
                                    lexer="text",
                                    theme=code_theme,
                                    line_numbers=False,
                                    background_color="default",
                                )
                                content_parts.append(syntax_content)
                            else:
                                content_parts.append(Text("Empty file"))
                        else:
                            # Content modification - show diff
                            diff_content = self._generate_clean_diff_content(old_string, new_string)
                            if diff_content.strip():
                                syntax_diff = Syntax(
                                    diff_content,
                                    lexer="diff",
                                    theme=code_theme,
                                    line_numbers=False,
                                    background_color="default",
                                )
                                content_parts.append(syntax_diff)
                            else:
                                content_parts.append(Text("No changes detected"))
                    else:
                        content_parts.append(Text("No edit operations found", style="dim"))

                # Add operation result if available
                if self.content:
                    content_parts.append(Text(""))
                    result_text = Text(f"Result: {self.content}", style="bold green")
                    content_parts.append(result_text)

                rendered_content = Group(*content_parts)
            else:
                # Fallback content
                fallback_parts = [Text("No edit parameters available")]
                if self.args and "raw_args" in self._parse_edit_args():
                    fallback_parts.append(Text(f"Raw args: {self._parse_edit_args()['raw_args']}"))
                if self.content:
                    fallback_parts.append(Text(f"Result: {self.content}"))
                rendered_content = Group(*fallback_parts)

        except Exception as e:
            error_parts = [Text(f"Error processing edit data: {e!s}")]
            if self.content:
                error_parts.append(Text(f"Raw result: {self.content}"))
            rendered_content = Group(*error_parts)

        return Panel(
            rendered_content,
            title=f"[TOOL] {self.name}",
            title_align="left",
            border_style="green",
        )

    def _format_edit_sequence(self, edits_list: list[dict], code_theme: str = "monokai") -> list:
        """Format a sequence of edit operations for display."""
        content_parts = []

        for i, edit_item in enumerate(edits_list, 1):
            old_string = edit_item.get("old_string", "")
            new_string = edit_item.get("new_string", "")
            replace_all = edit_item.get("replace_all", False)

            # Add edit operation header
            operation_type = "New file creation" if not old_string else "Content modification"
            replace_info = " (replace all)" if replace_all else ""
            edit_header = Text(f"Edit #{i}: {operation_type}{replace_info}", style="bold blue")
            content_parts.append(edit_header)

            # Handle new file creation (empty old_string)
            if not old_string:
                if new_string:
                    lines = new_string.split("\n")
                    content_preview = "\n".join(lines[:15]) + "\n...(truncated)" if len(lines) > 15 else new_string
                    syntax_content = Syntax(
                        content_preview, lexer="text", theme=code_theme, line_numbers=False, background_color="default"
                    )
                    content_parts.append(syntax_content)
                else:
                    content_parts.append(Text("Empty file", style="dim"))
            else:
                # Generate diff content for modification
                diff_content = self._generate_clean_diff_content(old_string, new_string)
                if diff_content.strip():
                    syntax_diff = Syntax(
                        diff_content, lexer="diff", theme=code_theme, line_numbers=False, background_color="default"
                    )
                    content_parts.append(syntax_diff)
                else:
                    content_parts.append(Text("No changes detected", style="dim"))

            # Add spacing between edits
            if i < len(edits_list):
                content_parts.append(Text(""))

        return content_parts

    def _create_edit_summary(self, edits_list: list[dict]) -> Text:
        """Create a summary of edit operations."""
        total_edits = len(edits_list)
        new_files = sum(1 for edit in edits_list if not edit.get("old_string", ""))
        modifications = total_edits - new_files
        replace_all_count = sum(1 for edit in edits_list if edit.get("replace_all", False))

        summary_parts = []
        if new_files > 0:
            summary_parts.append(f"{new_files} new file{'s' if new_files > 1 else ''}")
        if modifications > 0:
            summary_parts.append(f"{modifications} modification{'s' if modifications > 1 else ''}")
        if replace_all_count > 0:
            summary_parts.append(f"{replace_all_count} replace-all operation{'s' if replace_all_count > 1 else ''}")

        summary_text = ", ".join(summary_parts) if summary_parts else "No operations"
        return Text(f"Summary: {total_edits} edit{'s' if total_edits > 1 else ''} ({summary_text})", style="bold green")

    def _generate_clean_diff_content(self, old_string: str, new_string: str) -> str:
        """Generate a clean diff between old and new content."""
        # Use splitlines() without keepends to normalize line handling
        old_lines = old_string.splitlines()
        new_lines = new_string.splitlines()

        diff = difflib.unified_diff(old_lines, new_lines, fromfile="before", tofile="after", lineterm="")

        diff_lines = list(diff)
        if diff_lines:
            # Skip the header lines (first 2 lines with --- and +++)
            content_lines = diff_lines[2:] if len(diff_lines) > 2 else diff_lines

            # Remove excessive consecutive blank lines
            cleaned_lines = []
            prev_empty = False
            for line in content_lines:
                is_empty = line.strip() == ""
                if not (is_empty and prev_empty):
                    cleaned_lines.append(line)
                prev_empty = is_empty

            diff_content = "\n".join(cleaned_lines).rstrip()
            return diff_content if diff_content.strip() else "No changes detected"

        return "No changes detected"

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
        if tool_message.name in {"edit", "thinking", "to_do_read", "to_do_write", "multi_edit"}:
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
