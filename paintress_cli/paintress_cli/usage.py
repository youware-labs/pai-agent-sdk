"""Session-level usage tracking for paintress-cli.

This module provides usage tracking across multiple agent runs in a CLI session.
It aggregates token usage from:
- Main agent runs (via stream.run.usage())
- Extra usage from subagents, image/video understanding, compact filter, etc.
  (via ctx.extra_usages)

Uses pydantic-ai's RunUsage directly for accurate tracking including details field.

Example:
    session_usage = SessionUsage()

    # After each run - main agent
    session_usage.add("main", "openai:gpt-4o", run.usage())

    # Extra usages from subagents, etc.
    for record in ctx.extra_usages:
        session_usage.add(record.agent, record.model_id, record.usage)

    # Show summary
    print(session_usage.format_summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai.usage import RunUsage


@dataclass
class SessionUsage:
    """Session-level usage tracking, aggregated by both agent and model.

    Tracks token usage across all agent runs in a CLI session.
    Usage is grouped by:
    - Agent name (main, subagent names, image_understanding, etc.)
    - Model ID (openai:gpt-4o, anthropic:claude-sonnet-4, etc.)

    Uses pydantic-ai's RunUsage for accurate tracking including details field.

    Attributes:
        agent_usages: Dict mapping agent name to its RunUsage.
        model_usages: Dict mapping model_id to its RunUsage.
    """

    agent_usages: dict[str, RunUsage] = field(default_factory=dict)
    model_usages: dict[str, RunUsage] = field(default_factory=dict)

    def add(self, agent: str, model_id: str, usage: RunUsage) -> None:
        """Add usage for a specific agent and model.

        Args:
            agent: Agent name (e.g., "main", "explorer", "image_understanding").
            model_id: Model identifier (e.g., "openai:gpt-4o", "anthropic:claude-sonnet-4").
            usage: The RunUsage to accumulate.
        """
        # Accumulate by agent
        if agent not in self.agent_usages:
            self.agent_usages[agent] = RunUsage()
        self.agent_usages[agent].incr(usage)

        # Accumulate by model
        if model_id not in self.model_usages:
            self.model_usages[model_id] = RunUsage()
        self.model_usages[model_id].incr(usage)

    def clear(self) -> None:
        """Clear all accumulated usage."""
        self.agent_usages.clear()
        self.model_usages.clear()

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all models."""
        return sum(u.input_tokens or 0 for u in self.model_usages.values())

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all models."""
        return sum(u.output_tokens or 0 for u in self.model_usages.values())

    @property
    def total_tokens(self) -> int:
        """Total tokens across all models."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_requests(self) -> int:
        """Total LLM requests across all models."""
        return sum(u.requests or 0 for u in self.model_usages.values())

    def is_empty(self) -> bool:
        """Check if no usage has been recorded."""
        return len(self.model_usages) == 0

    def _format_usage_entry(self, name: str, usage: RunUsage) -> list[str]:
        """Format a single usage entry."""
        lines = [f"  {name}:"]
        lines.append(f"    Input:  {usage.input_tokens or 0:,} tokens")
        lines.append(f"    Output: {usage.output_tokens or 0:,} tokens")
        if usage.cache_read_tokens:
            lines.append(f"    Cache Read:  {usage.cache_read_tokens:,} tokens")
        if usage.cache_write_tokens:
            lines.append(f"    Cache Write: {usage.cache_write_tokens:,} tokens")
        if usage.requests:
            lines.append(f"    Requests: {usage.requests}")
        return lines

    def format_summary(self) -> str:
        """Format usage summary as a string.

        Returns:
            Formatted string with usage breakdown by model and agent.
        """
        if self.is_empty():
            return "No usage data available."

        lines = ["Token Usage Summary:", ""]

        # By Model breakdown
        lines.append("By Model:")
        for model_id, usage in sorted(self.model_usages.items()):
            lines.extend(self._format_usage_entry(model_id, usage))
            lines.append("")

        # By Agent breakdown
        lines.append("By Agent:")
        for agent, usage in sorted(self.agent_usages.items()):
            lines.extend(self._format_usage_entry(agent, usage))
            lines.append("")

        # Totals
        lines.append("Total:")
        lines.append(f"  Input:  {self.total_input_tokens:,} tokens")
        lines.append(f"  Output: {self.total_output_tokens:,} tokens")
        lines.append(f"  Total:  {self.total_tokens:,} tokens")
        lines.append(f"  Requests: {self.total_requests}")

        return "\n".join(lines)
