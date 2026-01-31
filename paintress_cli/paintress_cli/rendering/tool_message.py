"""Tool message display formatting.

Provides ToolMessage class for formatting tool call results.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from rich.panel import Panel
from rich.text import Text

from paintress_cli.rendering.tool_panels import (
    create_default_panel,
    create_edit_panel,
    create_task_panel,
    create_thinking_panel,
    create_todo_panel,
    format_args_for_display,
    format_output_for_display,
    generate_unified_diff,
)


class ToolMessage(BaseModel):
    """Tool message for display formatting."""

    tool_call_id: str
    name: str
    args: str | dict[str, Any] | None = None
    content: str | None = None

    def to_panel(
        self,
        max_result_lines: int = 2,
        max_arg_length: int = 120,
    ) -> Panel:
        """Convert the tool message to a Rich Panel object."""
        return create_default_panel(
            name=self.name,
            args=self.args,
            content=self.content,
            max_result_lines=max_result_lines,
            max_arg_length=max_arg_length,
        )

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

    def _format_inline_args_with_info(self, max_length: int = 80) -> tuple[str, int]:
        """Format tool arguments for inline display with truncation info."""
        return format_args_for_display(self.args, max_length)

    def _format_inline_output_with_info(self, max_lines: int = 8, max_line_length: int = 100) -> tuple[str, int]:
        """Format tool output for inline display with truncation info."""
        return format_output_for_display(self.content, max_lines, max_line_length)

    def to_special_panel(self, code_theme: str = "monokai") -> Panel:
        """Create special panel for to_do, task, thinking, and edit tools."""
        if self.name in ["to_do_read", "to_do_write"]:
            return create_todo_panel(self.name, self.content)
        elif self.name == "thinking":
            return create_thinking_panel(self.args, code_theme)
        elif self.name in ["edit", "multi_edit"]:
            return create_edit_panel(self.name, self.args, self.content, code_theme)
        elif self.name in ["task_create", "task_get", "task_update", "task_list"]:
            return create_task_panel(self.name, self.content)
        else:
            return self.to_panel()

    def _generate_clean_diff_content(self, old_string: str, new_string: str) -> str:
        """Generate a clean diff between old and new content."""
        return generate_unified_diff(old_string, new_string)
