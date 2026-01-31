"""To-do tool panel rendering."""

from __future__ import annotations

import json

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text


def create_todo_panel(name: str, content: str | None) -> Panel:
    """Create a special panel for to_do tools.

    Both to_do_read and to_do_write return JSON string in content.
    """
    panel_content: RenderableType = ""
    try:
        if content and content.startswith("["):
            to_dos = json.loads(content)
            if isinstance(to_dos, list) and to_dos:
                panel_content = _format_todo_list(to_dos)
            else:
                panel_content = "No to_dos found"
        elif content:
            panel_content = str(content)
        else:
            panel_content = "No to_do data available"
    except json.JSONDecodeError:
        panel_content = str(content) if content else "Invalid to_do data"

    return Panel(
        panel_content,
        title=f"[TOOL] {name}",
        title_align="left",
        border_style="blue",
    )


def _format_todo_list(to_dos: list) -> RenderableType:
    """Format a list of to_dos for display with rich styling."""
    if not to_dos:
        return "No to_dos found"

    priority_groups: dict[str, list[dict]] = {"high": [], "medium": [], "low": []}
    for to_do in to_dos:
        if isinstance(to_do, dict):
            priority = to_do.get("priority", "medium")
            if priority in priority_groups:
                priority_groups[priority].append(to_do)

    parts: list[RenderableType] = []

    priority_styles = {
        "high": ("bold magenta", "High"),
        "medium": ("bold", "Medium"),
        "low": ("dim", "Low"),
    }

    for priority in ["high", "medium", "low"]:
        items = priority_groups[priority]
        if items:
            style, label = priority_styles[priority]
            header = Text(f"{label} Priority ({len(items)} items)", style=style)
            parts.append(header)

            for item in items:
                status = item.get("status", "pending")
                item_content = item.get("content", "No content")
                item_id = item.get("id", "")

                line = Text()
                line.append("  ")

                # Status indicator with styling
                if status == "completed":
                    line.append("[x]", style="bold green")
                    line.append(" ")
                    if item_id:
                        line.append(f"{item_id}: ", style="strike dim")
                    line.append(item_content, style="strike dim")
                elif status == "in_progress":
                    line.append("[~]", style="bold cyan")
                    line.append(" ")
                    if item_id:
                        line.append(f"{item_id}: ", style="cyan")
                    line.append(item_content, style="cyan")
                else:  # pending
                    line.append("[ ]", style="dim")
                    line.append(" ")
                    if item_id:
                        line.append(f"{item_id}: ")
                    line.append(item_content)

                parts.append(line)
            parts.append(Text(""))

    # Progress summary
    total = len(to_dos)
    completed = sum(1 for to_do in to_dos if to_do.get("status") == "completed")
    in_progress = sum(1 for to_do in to_dos if to_do.get("status") == "in_progress")
    pending = sum(1 for to_do in to_dos if to_do.get("status") == "pending")

    if total > 0:
        completion_rate = int((completed / total) * 100)
        progress_line = Text()
        progress_line.append("Progress: ")
        progress_line.append(f"{completed}/{total}", style="bold green" if completed == total else "bold")
        progress_line.append(f" ({completion_rate}%)")
        parts.append(progress_line)

        status_line = Text()
        status_line.append("Status: ")
        if in_progress > 0:
            status_line.append(f"{in_progress} in progress", style="cyan")
            if pending > 0:
                status_line.append(", ")
        if pending > 0:
            status_line.append(f"{pending} pending", style="dim")
        parts.append(status_line)

    return Group(*parts)
