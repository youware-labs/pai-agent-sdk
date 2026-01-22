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

    def test_add_creates_entries(self) -> None:
        """Test adding usage creates agent and model entries."""
        session = SessionUsage()
        run_usage = RunUsage(input_tokens=100, output_tokens=50, requests=1)

        session.add("main", "openai:gpt-4o", run_usage)

        assert not session.is_empty()
        assert "main" in session.agent_usages
        assert "openai:gpt-4o" in session.model_usages
        assert session.agent_usages["main"].input_tokens == 100
        assert session.model_usages["openai:gpt-4o"].input_tokens == 100

    def test_add_multiple_agents_same_model(self) -> None:
        """Test multiple agents using the same model."""
        session = SessionUsage()

        session.add("main", "openai:gpt-4o", RunUsage(input_tokens=100, output_tokens=50, requests=1))
        session.add("explorer", "openai:gpt-4o", RunUsage(input_tokens=200, output_tokens=100, requests=1))

        # Agent usages are separate
        assert len(session.agent_usages) == 2
        assert session.agent_usages["main"].input_tokens == 100
        assert session.agent_usages["explorer"].input_tokens == 200

        # Model usage is accumulated
        assert len(session.model_usages) == 1
        assert session.model_usages["openai:gpt-4o"].input_tokens == 300

    def test_add_same_agent_different_models(self) -> None:
        """Test same agent using different models (e.g., image_understanding)."""
        session = SessionUsage()

        session.add("image_understanding", "openai:gpt-4o", RunUsage(input_tokens=100, output_tokens=50, requests=1))
        session.add(
            "image_understanding",
            "anthropic:claude-sonnet-4",
            RunUsage(input_tokens=200, output_tokens=100, requests=1),
        )

        # Agent usage is accumulated
        assert len(session.agent_usages) == 1
        assert session.agent_usages["image_understanding"].input_tokens == 300

        # Model usages are separate
        assert len(session.model_usages) == 2
        assert session.model_usages["openai:gpt-4o"].input_tokens == 100
        assert session.model_usages["anthropic:claude-sonnet-4"].input_tokens == 200

    def test_add_same_agent_accumulates(self) -> None:
        """Test adding to same agent accumulates."""
        session = SessionUsage()

        session.add("main", "openai:gpt-4o", RunUsage(input_tokens=100, output_tokens=50, requests=1))
        session.add("main", "openai:gpt-4o", RunUsage(input_tokens=200, output_tokens=100, requests=1))

        assert session.agent_usages["main"].input_tokens == 300
        assert session.agent_usages["main"].requests == 2
        assert session.model_usages["openai:gpt-4o"].input_tokens == 300

    def test_totals(self) -> None:
        """Test total calculations across all models."""
        session = SessionUsage()

        session.add("main", "openai:gpt-4o", RunUsage(input_tokens=100, output_tokens=50, requests=1))
        session.add("explorer", "anthropic:claude-sonnet-4", RunUsage(input_tokens=200, output_tokens=100, requests=2))

        assert session.total_input_tokens == 300
        assert session.total_output_tokens == 150
        assert session.total_tokens == 450
        assert session.total_requests == 3

    def test_clear(self) -> None:
        """Test clearing session usage."""
        session = SessionUsage()
        session.add("main", "openai:gpt-4o", RunUsage(input_tokens=100, output_tokens=50, requests=1))

        session.clear()

        assert session.is_empty()
        assert session.total_tokens == 0
        assert len(session.agent_usages) == 0
        assert len(session.model_usages) == 0

    def test_format_summary_empty(self) -> None:
        """Test format_summary on empty session."""
        session = SessionUsage()
        summary = session.format_summary()
        assert "No usage data" in summary

    def test_format_summary_with_data(self) -> None:
        """Test format_summary with data."""
        session = SessionUsage()
        session.add("main", "openai:gpt-4o", RunUsage(input_tokens=1000, output_tokens=500, requests=2))
        session.add("explorer", "anthropic:claude-sonnet-4", RunUsage(input_tokens=200, output_tokens=100, requests=1))

        summary = session.format_summary()

        assert "Token Usage Summary" in summary
        # By Model section
        assert "By Model:" in summary
        assert "openai:gpt-4o:" in summary
        assert "anthropic:claude-sonnet-4:" in summary
        # By Agent section
        assert "By Agent:" in summary
        assert "main:" in summary
        assert "explorer:" in summary
        # Formatting
        assert "1,000" in summary  # Comma formatting
        assert "Total:" in summary

    def test_preserves_details(self) -> None:
        """Test that details field is accumulated."""
        session = SessionUsage()
        # Details values must be numeric for accumulation
        session.add("main", "openai:gpt-4o", RunUsage(input_tokens=100, details={"cached_tokens": 50}))
        session.add("main", "openai:gpt-4o", RunUsage(input_tokens=100, details={"cached_tokens": 30}))

        assert session.agent_usages["main"].details == {"cached_tokens": 80}
        assert session.model_usages["openai:gpt-4o"].details == {"cached_tokens": 80}

    def test_cache_tokens(self) -> None:
        """Test cache token tracking."""
        session = SessionUsage()
        session.add(
            "main",
            "openai:gpt-4o",
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

        model_usage = session.model_usages["openai:gpt-4o"]
        assert model_usage.cache_read_tokens == 20
        assert model_usage.cache_write_tokens == 10
