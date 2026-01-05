"""Tests for pai_agent_sdk.context module."""

from datetime import datetime, timedelta

import pytest

from pai_agent_sdk.context import AgentContext


def test_agent_context_default_run_id() -> None:
    """Should generate a unique run_id by default."""
    ctx1 = AgentContext()
    ctx2 = AgentContext()
    assert ctx1.run_id != ctx2.run_id
    assert len(ctx1.run_id) == 32  # uuid4().hex length


def test_agent_context_no_parent_by_default() -> None:
    """Should have no parent by default."""
    ctx = AgentContext()
    assert ctx.parent_run_id is None


def test_agent_context_elapsed_time_before_start() -> None:
    """Should return None before context is started."""
    ctx = AgentContext()
    assert ctx.elapsed_time is None


def test_agent_context_elapsed_time_after_start() -> None:
    """Should return elapsed time after start."""
    ctx = AgentContext()
    ctx.start_at = datetime.now()
    elapsed = ctx.elapsed_time
    assert elapsed is not None
    assert isinstance(elapsed, timedelta)
    assert elapsed.total_seconds() >= 0


def test_agent_context_elapsed_time_after_end() -> None:
    """Should return final duration after end."""
    ctx = AgentContext()
    start = datetime.now()
    ctx.start_at = start
    ctx.end_at = start + timedelta(seconds=5)
    elapsed = ctx.elapsed_time
    assert elapsed is not None
    assert elapsed.total_seconds() == 5


def test_agent_context_enter_subagent() -> None:
    """Should create child context with proper inheritance."""
    parent = AgentContext()
    parent.start_at = datetime.now()

    with parent.enter_subagent("search") as child:
        assert child.parent_run_id == parent.run_id
        assert child.run_id != parent.run_id
        assert child._agent_name == "search"
        assert child.start_at is not None
        assert child.end_at is None

    # After exiting, end_at should be set
    assert child.end_at is not None


def test_agent_context_enter_subagent_with_override() -> None:
    """Should allow field overrides in subagent context."""
    parent = AgentContext()

    with parent.enter_subagent("reasoning", deferred_tool_metadata={"key": {}}) as child:
        assert child.deferred_tool_metadata == {"key": {}}


@pytest.mark.asyncio
async def test_agent_context_async_context_manager() -> None:
    """Should set start/end times in async context."""
    ctx = AgentContext()
    assert ctx.start_at is None
    assert ctx.end_at is None

    async with ctx:
        assert ctx.start_at is not None
        assert ctx.end_at is None

    assert ctx.end_at is not None
    assert ctx.end_at >= ctx.start_at


def test_agent_context_deferred_tool_metadata_default() -> None:
    """Should have empty metadata by default."""
    ctx = AgentContext()
    assert ctx.deferred_tool_metadata == {}


def test_agent_context_deferred_tool_metadata_storage() -> None:
    """Should store metadata by tool_call_id."""
    ctx = AgentContext()
    ctx.deferred_tool_metadata["call-1"] = {"user_choice": "option_a"}
    assert ctx.deferred_tool_metadata["call-1"]["user_choice"] == "option_a"
