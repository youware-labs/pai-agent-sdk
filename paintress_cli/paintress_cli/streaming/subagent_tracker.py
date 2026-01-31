"""Subagent state tracking for streaming display.

Tracks subagent execution progress for inline status updates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from rich.text import Text

from paintress_cli.rendering import RichRenderer


@dataclass
class SubagentState:
    """State for a single subagent execution."""

    agent_id: str
    agent_name: str
    line_index: int
    tool_names: list[str] = field(default_factory=list)


class SubagentTracker:
    """Tracks subagent execution states for display updates.

    Manages progress lines for running subagents, updating them
    as tools are called and completing them with summaries.
    """

    def __init__(
        self,
        renderer: RichRenderer,
        get_width: Callable[[], int] | None = None,
    ) -> None:
        """Initialize SubagentTracker.

        Args:
            renderer: RichRenderer for text rendering.
            get_width: Callback to get current terminal width.
        """
        self._renderer = renderer
        self._get_width = get_width or (lambda: 120)
        self._states: dict[str, SubagentState] = {}

    def has_state(self, agent_id: str) -> bool:
        """Check if we have state for an agent."""
        return agent_id in self._states

    def get_state(self, agent_id: str) -> SubagentState | None:
        """Get state for an agent."""
        return self._states.get(agent_id)

    def start(self, agent_id: str, agent_name: str, line_index: int) -> str:
        """Start tracking a new subagent.

        Args:
            agent_id: Agent identifier.
            agent_name: Human-readable agent name.
            line_index: Line index in output buffer.

        Returns:
            Rendered progress line.
        """
        self._states[agent_id] = SubagentState(
            agent_id=agent_id,
            agent_name=agent_name,
            line_index=line_index,
        )
        return self._render_progress(agent_id)

    def add_tool(self, agent_id: str, tool_name: str) -> str | None:
        """Add a tool call to subagent tracking.

        Args:
            agent_id: Agent identifier.
            tool_name: Name of the tool being called.

        Returns:
            Updated progress line, or None if agent not tracked.
        """
        if agent_id not in self._states:
            return None

        self._states[agent_id].tool_names.append(tool_name)
        return self._render_progress(agent_id)

    def complete(
        self,
        agent_id: str,
        success: bool,
        duration_seconds: float,
        request_count: int = 0,
        result_preview: str | None = None,
        error: str | None = None,
    ) -> tuple[str, int | None]:
        """Complete subagent tracking and return summary line.

        Args:
            agent_id: Agent identifier.
            success: Whether execution succeeded.
            duration_seconds: Execution duration.
            request_count: Number of model requests.
            result_preview: Preview of result (for success).
            error: Error message (for failure).

        Returns:
            Tuple of (rendered summary line, line_index or None).
        """
        state = self._states.pop(agent_id, None)
        line_index = state.line_index if state else None

        text = Text()
        if success:
            text.append(f"[{agent_id}] ", style="cyan")
            text.append("Done ", style="bold green")
            text.append(f"({duration_seconds:.1f}s)", style="dim")
            if request_count > 0:
                text.append(f" | {request_count} reqs", style="dim")
            if result_preview:
                # Truncate result preview
                preview = result_preview.replace("\n", " ")[:60]
                if len(result_preview) > 60:
                    preview += "..."
                text.append(f' | "{preview}"', style="dim italic")
        else:
            text.append(f"[{agent_id}] ", style="cyan")
            text.append("Failed ", style="bold red")
            text.append(f"({duration_seconds:.1f}s)", style="dim")
            if error:
                error_preview = error[:50]
                text.append(f" | {error_preview}", style="dim red")

        rendered = self._renderer.render(text, width=self._get_width()).rstrip()
        return rendered, line_index

    def clear(self) -> None:
        """Clear all tracked states."""
        self._states.clear()

    def _render_progress(self, agent_id: str) -> str:
        """Render progress line for an agent."""
        state = self._states.get(agent_id)
        if not state:
            return ""

        text = Text()
        text.append(f"[{agent_id}] ", style="cyan")
        text.append("Running... ", style="dim")

        if state.tool_names:
            # Show last few tools
            recent_tools = state.tool_names[-3:]
            tools_str = ", ".join(recent_tools)
            if len(state.tool_names) > 3:
                tools_str = f"...{tools_str}"
            text.append(tools_str, style="dim yellow")
            text.append(f" ({len(state.tool_names)} tools)", style="dim")

        return self._renderer.render(text, width=self._get_width()).rstrip()
