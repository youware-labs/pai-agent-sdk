"""Task tool panel rendering."""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text


def create_task_panel(name: str, content: str | None) -> Panel:
    """Create a special panel for task tools.

    Handles task_create, task_get, task_update, task_list tools.
    """
    panel_content: RenderableType = ""

    if name == "task_list":
        panel_content = _format_task_list(content)
    elif name == "task_create":
        # Show created task info from result
        panel_content = Text(content or "Task created", style="green")
    elif name == "task_get":
        # Show task details from result
        if content:
            panel_content = _format_task_details(content)
        else:
            panel_content = Text("Task not found", style="red")
    elif name == "task_update":
        # Show update confirmation
        panel_content = Text(content or "Task updated", style="cyan")
    else:
        panel_content = str(content) if content else "No task data"

    return Panel(
        panel_content,
        title=f"[TOOL] {name}",
        title_align="left",
        border_style="cyan",
    )


def _format_task_list(content: str | None) -> RenderableType:
    """Format task list output for display."""
    if not content:
        return Text("No tasks found", style="dim")

    lines = content.split("\n")
    parts: list[RenderableType] = []

    for line in lines:
        if not line.strip():
            continue

        text = Text()
        # Parse the line format: #1 [status] Subject [blocked by #X]
        if "[completed]" in line:
            text.append(line, style="strike dim green")
        elif "[in_progress" in line:
            text.append(line, style="bold cyan")
        elif "[blocked by" in line:
            # Split at blocked indicator
            idx = line.find("[blocked by")
            text.append(line[:idx], style="dim")
            text.append(line[idx:], style="dim red")
        else:
            text.append(line)
        parts.append(text)

    if not parts:
        return Text("No tasks found", style="dim")

    # Add summary
    total = len(lines)
    completed = sum(1 for line in lines if "[completed]" in line)
    in_progress = sum(1 for line in lines if "[in_progress" in line)

    parts.append(Text(""))
    progress_line = Text()
    progress_line.append("Progress: ")
    progress_line.append(f"{completed}/{total}", style="bold green" if completed == total else "bold")
    if in_progress > 0:
        progress_line.append(f" ({in_progress} in progress)", style="cyan")
    parts.append(progress_line)

    return Group(*parts)


def _format_task_details(content: str) -> RenderableType:
    """Format single task details for display."""
    lines = content.split("\n")
    parts: list[RenderableType] = []

    for line in lines:
        text = Text()
        if line.startswith("Task #"):
            text.append(line, style="bold")
        elif line.startswith("Status:"):
            status_val = line.split(":", 1)[1].strip() if ":" in line else ""
            text.append("Status: ")
            if status_val == "completed":
                text.append(status_val, style="green")
            elif status_val == "in_progress":
                text.append(status_val, style="cyan")
            else:
                text.append(status_val, style="dim")
        elif line.startswith("Blocked By:"):
            text.append("Blocked By: ", style="dim")
            text.append(line.split(":", 1)[1].strip() if ":" in line else "", style="red")
        else:
            text.append(line)
        parts.append(text)

    return Group(*parts)
