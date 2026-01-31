"""Edit tool panel rendering."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from paintress_cli.rendering.tool_panels.base import generate_unified_diff


def create_edit_panel(
    name: str,
    args: Any,
    content: str | None,
    code_theme: str = "monokai",
) -> Panel:
    """Create a special panel for edit/multi_edit tools with diff display."""
    try:
        args_dict = _parse_edit_args(args)

        if args_dict:
            file_path = args_dict.get("file_path", "unknown")
            edits_list = args_dict.get("edits", [])

            content_parts = []

            # Add file header
            file_info = Text(f"Editing file: {file_path}", style="bold cyan")
            content_parts.append(file_info)

            if edits_list:
                # Add edit summary for multi_edit
                summary = _create_edit_summary(edits_list)
                content_parts.append(summary)
                content_parts.append(Text(""))

                # Format and add edit sequence
                edit_sequence = _format_edit_sequence(edits_list, code_theme)
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
                        diff_content = generate_unified_diff(old_string, new_string)
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
            if content:
                content_parts.append(Text(""))
                result_text = Text(f"Result: {content}", style="bold green")
                content_parts.append(result_text)

            rendered_content = Group(*content_parts)
        else:
            # Fallback content
            fallback_parts = [Text("No edit parameters available")]
            if content:
                fallback_parts.append(Text(f"Result: {content}"))
            rendered_content = Group(*fallback_parts)

    except Exception as e:
        error_parts = [Text(f"Error processing edit data: {e!s}")]
        if content:
            error_parts.append(Text(f"Raw result: {content}"))
        rendered_content = Group(*error_parts)

    return Panel(
        rendered_content,
        title=f"[TOOL] {name}",
        title_align="left",
        border_style="green",
    )


def _parse_edit_args(args: Any) -> dict:
    """Parse the edit tool arguments into a dictionary."""
    args_dict = {}
    if args:
        if isinstance(args, dict):
            args_dict = args
        elif isinstance(args, str):
            try:
                args_dict = json.loads(args)
            except json.JSONDecodeError:
                args_dict = {"raw_args": args}
    return args_dict


def _create_edit_summary(edits_list: list[dict]) -> Text:
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


def _format_edit_sequence(edits_list: list[dict], code_theme: str = "monokai") -> list:
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
            diff_content = generate_unified_diff(old_string, new_string)
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
