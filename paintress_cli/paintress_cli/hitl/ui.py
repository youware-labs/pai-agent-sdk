"""Human-in-the-Loop UI components.

Provides UI rendering for tool approval panels.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from paintress_cli.rendering import RichRenderer

if TYPE_CHECKING:
    from pydantic_ai.messages import ToolCallPart


class ApprovalUI:
    """Renders approval UI components for HITL workflow."""

    def __init__(
        self,
        renderer: RichRenderer,
        code_theme: str = "monokai",
    ) -> None:
        """Initialize ApprovalUI.

        Args:
            renderer: RichRenderer for rendering.
            code_theme: Code highlighting theme.
        """
        self._renderer = renderer
        self._code_theme = code_theme

    def render_approval_panel(
        self,
        tool_call: ToolCallPart,
        index: int,
        total: int,
        width: int | None = None,
    ) -> str:
        """Render an approval panel for a tool call.

        Args:
            tool_call: The tool call needing approval.
            index: Current index (1-based).
            total: Total number of tools.
            width: Optional render width.

        Returns:
            Rendered ANSI string.
        """
        content_parts: list[Any] = [
            Text(f"Tool {index} of {total}", style="bold cyan"),
            Text(""),
            Text(f"Tool: {tool_call.tool_name}", style="bold yellow"),
        ]

        if tool_call.args:
            content_parts.append(Text(""))
            content_parts.append(Text("Arguments:", style="bold cyan"))
            formatted_args = self._format_args(tool_call.args)
            # Determine if it looks like JSON for syntax highlighting
            is_json_like = formatted_args.strip().startswith(("{", "["))
            syntax = Syntax(
                formatted_args,
                "json" if is_json_like else "text",
                theme=self._code_theme,
            )
            content_parts.append(syntax)

        panel = Panel(
            Group(*content_parts),
            title="[yellow]Tool Approval Required[/yellow]",
            subtitle="[dim]Enter/Y: Approve | Any text: Reject with reason | Ctrl+C: Cancel[/dim]",
            border_style="yellow",
            padding=(1, 2),
        )

        return self._renderer.render(panel, width=width).rstrip()

    def render_approval_result(
        self,
        tool_name: str,
        approved: bool,
        reason: str | None = None,
    ) -> str:
        """Render an approval result message.

        Args:
            tool_name: Name of the tool.
            approved: Whether the tool was approved.
            reason: Rejection reason (if not approved).

        Returns:
            Rendered ANSI string.
        """
        text = Text()
        text.append("  ")
        if approved:
            text.append("[Approved: ", style="dim")
            text.append(tool_name, style="green")
            text.append("]", style="dim")
        else:
            text.append("[Rejected: ", style="dim")
            text.append(tool_name, style="red")
            if reason:
                text.append(f" - {reason}", style="dim")
            text.append("]", style="dim")

        return self._renderer.render(text).rstrip()

    def render_approval_header(self, count: int) -> str:
        """Render approval header message.

        Args:
            count: Number of tools needing approval.

        Returns:
            Rendered ANSI string.
        """
        text = Text()
        text.append(f"[Tool approval required: {count} tool(s)]", style="bold yellow")
        return self._renderer.render(text).rstrip()

    def _format_args(
        self,
        args: Any,
        max_str_len: int = 500,
        max_lines: int = 30,
    ) -> str:
        """Format tool arguments for display with smart truncation.

        Args:
            args: Tool arguments (can be dict, JSON string, or any object)
            max_str_len: Maximum length for string values before truncation
            max_lines: Maximum number of lines in output

        Returns:
            Formatted JSON string or fallback representation
        """

        def truncate_strings(obj: Any, max_len: int) -> Any:
            """Recursively truncate long strings in nested structures."""
            if isinstance(obj, str):
                if len(obj) > max_len:
                    return obj[:max_len] + f"... ({len(obj) - max_len} more chars)"
                return obj
            elif isinstance(obj, dict):
                return {k: truncate_strings(v, max_len) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [truncate_strings(item, max_len) for item in obj]
            return obj

        try:
            # If args is a string, try to parse it as JSON first
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                    args = parsed
                except json.JSONDecodeError:
                    # Not valid JSON, treat as plain string
                    if len(args) > max_str_len:
                        return args[:max_str_len] + f"\n... ({len(args) - max_str_len} more chars)"
                    return args

            # Truncate long strings in the structure
            truncated = truncate_strings(args, max_str_len)

            # Format as pretty JSON
            formatted = json.dumps(truncated, indent=2, ensure_ascii=False)

            # Limit total lines
            lines = formatted.split("\n")
            if len(lines) > max_lines:
                formatted = "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"

            return formatted

        except Exception:
            # Ultimate fallback: convert to string
            result = str(args)
            if len(result) > max_str_len:
                result = result[:max_str_len] + f"... ({len(result) - max_str_len} more chars)"
            return result
