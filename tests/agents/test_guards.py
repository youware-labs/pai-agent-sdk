"""Tests for agent output guards."""

from unittest.mock import MagicMock

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry

from pai_agent_sdk.agents.guards import attach_message_bus_guard, message_bus_guard
from pai_agent_sdk.bus import MessageBus
from pai_agent_sdk.context import AgentContext


def create_mock_ctx(agent_id: str = "main", message_bus: MessageBus | None = None) -> RunContext[AgentContext]:
    """Create a mock RunContext with AgentContext."""
    ctx = AgentContext()
    ctx._agent_id = agent_id
    if message_bus:
        ctx.message_bus = message_bus

    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = ctx
    return mock_run_ctx


async def test_message_bus_guard_no_pending() -> None:
    """Test guard passes when no pending messages."""
    bus = MessageBus()
    bus.subscribe("main")
    ctx = create_mock_ctx(message_bus=bus)

    result = await message_bus_guard(ctx, "output text")

    assert result == "output text"


async def test_message_bus_guard_with_pending() -> None:
    """Test guard raises ModelRetry when messages pending."""
    bus = MessageBus()
    bus.subscribe("main")
    bus.send("Please focus", source="user", target="main")
    ctx = create_mock_ctx(message_bus=bus)

    with pytest.raises(ModelRetry) as exc_info:
        await message_bus_guard(ctx, "output text")

    assert "pending messages" in str(exc_info.value).lower()


async def test_message_bus_guard_different_target() -> None:
    """Test guard passes when messages for different agent."""
    bus = MessageBus()
    bus.subscribe("main")
    bus.subscribe("other-agent")
    bus.send("For other", source="user", target="other-agent")
    ctx = create_mock_ctx(agent_id="main", message_bus=bus)

    result = await message_bus_guard(ctx, "output text")

    assert result == "output text"


async def test_message_bus_guard_broadcast() -> None:
    """Test guard triggers on broadcast messages."""
    bus = MessageBus()
    bus.subscribe("main")
    bus.send("Broadcast", source="user")  # No target = broadcast
    ctx = create_mock_ctx(message_bus=bus)

    with pytest.raises(ModelRetry):
        await message_bus_guard(ctx, "output text")


async def test_message_bus_guard_preserves_output_type() -> None:
    """Test guard preserves output type (not just str)."""
    bus = MessageBus()
    bus.subscribe("main")
    ctx = create_mock_ctx(message_bus=bus)

    # Test with dict
    result = await message_bus_guard(ctx, {"key": "value"})
    assert result == {"key": "value"}

    # Test with list
    result = await message_bus_guard(ctx, [1, 2, 3])
    assert result == [1, 2, 3]


async def test_message_bus_guard_subagent() -> None:
    """Test guard works for subagent context."""
    bus = MessageBus()
    bus.subscribe("subagent-123")
    bus.send("For subagent", source="main", target="subagent-123")
    ctx = create_mock_ctx(agent_id="subagent-123", message_bus=bus)

    with pytest.raises(ModelRetry):
        await message_bus_guard(ctx, "output text")


async def test_message_bus_guard_unsubscribed() -> None:
    """Test guard passes when agent not subscribed (no pending)."""
    bus = MessageBus()
    bus.send("Hello", source="user", target="main")
    ctx = create_mock_ctx(message_bus=bus)

    # Not subscribed, so has_pending returns False
    result = await message_bus_guard(ctx, "output text")
    assert result == "output text"


def test_attach_message_bus_guard() -> None:
    """Test attach_message_bus_guard adds validator to agent."""
    agent: Agent[AgentContext, str] = Agent(
        model="test",
        deps_type=AgentContext,
    )

    # Before attach, no output validators from our guard
    initial_validators = len(agent._output_validators)

    attach_message_bus_guard(agent)

    # After attach, one more validator
    assert len(agent._output_validators) == initial_validators + 1
