"""Base utilities for tool panel rendering."""

from __future__ import annotations

import difflib
import json
from typing import Any

from rich.panel import Panel


def format_args_for_display(args: Any, max_length: int = 80) -> tuple[str, int]:
    """Format tool arguments for inline display with truncation info.

    Returns:
        Tuple of (formatted_args, truncated_chars)
    """
    if not args:
        return "", 0
    try:
        if isinstance(args, dict):
            args_str = json.dumps(args, ensure_ascii=False)
        else:
            args_str = str(args)
        if len(args_str) > max_length:
            return args_str[: max_length - 3] + "...", len(args_str) - max_length + 3
        return args_str, 0
    except Exception:
        s = str(args)
        if len(s) > max_length:
            return s[:max_length], len(s) - max_length
        return s, 0


def format_output_for_display(content: str | None, max_lines: int = 8, max_line_length: int = 100) -> tuple[str, int]:
    """Format tool output for inline display with truncation info.

    Returns:
        Tuple of (formatted_output, truncated_chars)
    """
    if not content:
        return "[no output]", 0

    original_len = len(content)
    try:
        # Try to parse as JSON for prettier formatting
        try:
            parsed = json.loads(content)
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
        first_line = content.split("\n")[0]
        if len(first_line) > max_line_length:
            result = first_line[: max_line_length - 3] + "..."
            truncated = original_len - max_line_length + 3
        else:
            result = first_line
            truncated = original_len - len(first_line) if "\n" in content else 0
        return result, max(0, truncated)
    except Exception:
        s = str(content)[:max_line_length]
        return s, max(0, original_len - len(s))


def generate_unified_diff(old_string: str, new_string: str) -> str:
    """Generate a clean unified diff between old and new content."""
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


def create_default_panel(
    name: str,
    args: Any = None,
    content: str | None = None,
    max_result_lines: int = 2,
    max_arg_length: int = 120,
) -> Panel:
    """Create a default tool panel."""
    args_text = ""
    if args:
        try:
            if isinstance(args, dict):
                args_lines = []
                for key, value in args.items():
                    str_value = str(value)
                    if len(str_value) > max_arg_length:
                        str_value = str_value[:max_arg_length] + "..."
                    args_lines.append(f"{key}: {str_value}")
                args_text = "\n".join(args_lines[:3]) + "\n..." if len(args_lines) > 3 else "\n".join(args_lines)
            else:
                str_args = str(args)
                if len(str_args) > max_arg_length:
                    str_args = str_args[:max_arg_length] + "..."
                args_text = f"args: {str_args}"
        except Exception:
            args_text = "args: <complex>"

    result_text = ""
    if content:
        try:
            content_lines = content.split("\n")
            if len(content_lines) > max_result_lines:
                result_text = "\n".join(content_lines[:max_result_lines]) + "\n[...]"
            else:
                result_text = content
        except Exception:
            result_text = "<result available>"

    # Combine args and result
    panel_content = args_text
    if result_text:
        if panel_content:
            panel_content += "\n\n"
        panel_content += result_text[:max_arg_length]

    if not panel_content:
        panel_content = "No details available"

    return Panel(panel_content, title=f"[TOOL] {name}", title_align="left")
