"""Session-level usage tracking for paintress-cli.

This module provides usage tracking across multiple agent runs in a CLI session.
It aggregates token usage from:
- Main agent runs (via stream.run.usage())
- Extra usage from subagents, image/video understanding, compact filter, etc.
  (via ctx.extra_usages)

Uses pydantic-ai's RunUsage directly for accurate tracking including details field.

Example:
    session_usage = SessionUsage()

    # After each run
    session_usage.add("main", run.usage())
    for record in ctx.extra_usages:
        session_usage.add(record.agent, record.usage)

    # Show summary
    print(session_usage.format_summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai.usage import RunUsage


@dataclass
class SessionUsage:
    """Session-level usage tracking, aggregated by agent/source.

    Tracks token usage across all agent runs in a CLI session.
    Usage is grouped by agent name (main, subagent names, image_understanding, etc.)
    Uses pydantic-ai's RunUsage for accurate tracking including details field.

    Attributes:
        agent_usages: Dict mapping agent name to its RunUsage.
    """

    agent_usages: dict[str, RunUsage] = field(default_factory=dict)

    def add(self, agent: str, usage: RunUsage) -> None:
        """Add usage for a specific agent.

        Args:
            agent: Agent name (e.g., "main", "search_agent", "image_understanding").
            usage: The RunUsage to accumulate.
        """
        if agent not in self.agent_usages:
            self.agent_usages[agent] = RunUsage()
        self.agent_usages[agent].incr(usage)

    def clear(self) -> None:
        """Clear all accumulated usage."""
        self.agent_usages.clear()

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all agents."""
        return sum(u.input_tokens or 0 for u in self.agent_usages.values())

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all agents."""
        return sum(u.output_tokens or 0 for u in self.agent_usages.values())

    @property
    def total_tokens(self) -> int:
        """Total tokens across all agents."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_requests(self) -> int:
        """Total LLM requests across all agents."""
        return sum(u.requests or 0 for u in self.agent_usages.values())

    def is_empty(self) -> bool:
        """Check if no usage has been recorded."""
        return len(self.agent_usages) == 0

    def format_summary(self) -> str:
        """Format usage summary as a string.

        Returns:
            Formatted string with usage breakdown by agent.
        """
        if self.is_empty():
            return "No usage data available."

        lines = ["Token Usage Summary:", ""]

        # Per-agent breakdown
        for agent, usage in sorted(self.agent_usages.items()):
            lines.append(f"  {agent}:")
            lines.append(f"    Input:  {usage.input_tokens or 0:,} tokens")
            lines.append(f"    Output: {usage.output_tokens or 0:,} tokens")
            if usage.cache_read_tokens:
                lines.append(f"    Cache Read:  {usage.cache_read_tokens:,} tokens")
            if usage.cache_write_tokens:
                lines.append(f"    Cache Write: {usage.cache_write_tokens:,} tokens")
            if usage.requests:
                lines.append(f"    Requests: {usage.requests}")
            if usage.details:
                lines.append(f"    Details: {usage.details}")
            lines.append("")

        # Totals
        lines.append("  Total:")
        lines.append(f"    Input:  {self.total_input_tokens:,} tokens")
        lines.append(f"    Output: {self.total_output_tokens:,} tokens")
        lines.append(f"    Total:  {self.total_tokens:,} tokens")
        lines.append(f"    Requests: {self.total_requests}")

        return "\n".join(lines)
