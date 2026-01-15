"""Tests for usage tracking module."""

from __future__ import annotations

from paintress_cli.usage import SessionUsage
from pydantic_ai.usage import RunUsage


class TestSessionUsage:
    """Tests for SessionUsage dataclass."""

    def test_is_empty(self) -> None:
        """Test is_empty on fresh instance."""
        session = SessionUsage()
        assert session.is_empty()

    def test_add_creates_agent_entry(self) -> None:
        """Test adding usage creates agent entry."""
        session = SessionUsage()
        run_usage = RunUsage(input_tokens=100, output_tokens=50, requests=1)

        session.add("main", run_usage)

        assert not session.is_empty()
        assert "main" in session.agent_usages
        assert session.agent_usages["main"].input_tokens == 100

    def test_add_multiple_agents(self) -> None:
        """Test tracking multiple agents separately."""
        session = SessionUsage()

        session.add("main", RunUsage(input_tokens=100, output_tokens=50, requests=1))
        session.add("search_agent", RunUsage(input_tokens=200, output_tokens=100, requests=1))

        assert len(session.agent_usages) == 2
        assert session.agent_usages["main"].input_tokens == 100
        assert session.agent_usages["search_agent"].input_tokens == 200

    def test_add_same_agent_accumulates(self) -> None:
        """Test adding to same agent accumulates."""
        session = SessionUsage()

        session.add("main", RunUsage(input_tokens=100, output_tokens=50, requests=1))
        session.add("main", RunUsage(input_tokens=200, output_tokens=100, requests=1))

        assert session.agent_usages["main"].input_tokens == 300
        assert session.agent_usages["main"].requests == 2

    def test_totals(self) -> None:
        """Test total calculations across all agents."""
        session = SessionUsage()

        session.add("main", RunUsage(input_tokens=100, output_tokens=50, requests=1))
        session.add("subagent", RunUsage(input_tokens=200, output_tokens=100, requests=2))

        assert session.total_input_tokens == 300
        assert session.total_output_tokens == 150
        assert session.total_tokens == 450
        assert session.total_requests == 3

    def test_clear(self) -> None:
        """Test clearing session usage."""
        session = SessionUsage()
        session.add("main", RunUsage(input_tokens=100, output_tokens=50, requests=1))

        session.clear()

        assert session.is_empty()
        assert session.total_tokens == 0

    def test_format_summary_empty(self) -> None:
        """Test format_summary on empty session."""
        session = SessionUsage()
        summary = session.format_summary()
        assert "No usage data" in summary

    def test_format_summary_with_data(self) -> None:
        """Test format_summary with data."""
        session = SessionUsage()
        session.add("main", RunUsage(input_tokens=1000, output_tokens=500, requests=2))
        session.add("search_agent", RunUsage(input_tokens=200, output_tokens=100, requests=1))

        summary = session.format_summary()

        assert "Token Usage Summary" in summary
        assert "main:" in summary
        assert "search_agent:" in summary
        assert "1,000" in summary  # Comma formatting
        assert "Total:" in summary

    def test_preserves_details(self) -> None:
        """Test that details field is accumulated."""
        session = SessionUsage()
        # Details values must be numeric for accumulation
        session.add("main", RunUsage(input_tokens=100, details={"cached_tokens": 50}))
        session.add("main", RunUsage(input_tokens=100, details={"cached_tokens": 30}))

        assert session.agent_usages["main"].details == {"cached_tokens": 80}

    def test_cache_tokens(self) -> None:
        """Test cache token tracking."""
        session = SessionUsage()
        session.add(
            "main",
            RunUsage(
                input_tokens=100,
                output_tokens=50,
                cache_read_tokens=20,
                cache_write_tokens=10,
            ),
        )

        usage = session.agent_usages["main"]
        assert usage.cache_read_tokens == 20
        assert usage.cache_write_tokens == 10
