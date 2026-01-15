"""Tests for steering module."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest
from paintress_cli.steering import (
    LocalSteeringManager,
    SteeringManager,
    SteeringMessage,
    steering_output_guard,
)

# --- SteeringMessage Tests ---


def test_steering_message_create() -> None:
    """Test basic message creation."""
    msg = SteeringMessage(message_id="test-123", prompt="focus on tests")
    assert msg.message_id == "test-123"
    assert msg.prompt == "focus on tests"
    assert msg.timestamp is not None


def test_steering_message_with_custom_timestamp() -> None:
    """Test message with custom timestamp."""
    ts = datetime(2026, 1, 14, 12, 0, 0)
    msg = SteeringMessage(message_id="test-456", prompt="hello", timestamp=ts)
    assert msg.timestamp == ts


# --- LocalSteeringManager Tests ---


@pytest.fixture
def manager() -> LocalSteeringManager:
    """Create a fresh manager for each test."""
    return LocalSteeringManager(max_size=5)


async def test_local_manager_enqueue_message(manager: LocalSteeringManager) -> None:
    """Test enqueuing a message."""
    msg = await manager.enqueue("focus on the UI")
    assert msg.prompt == "focus on the UI"
    assert msg.message_id.startswith("steer-")
    assert len(msg.message_id) == 14  # "steer-" + 8 hex chars
    assert manager.has_pending()
    assert len(manager) == 1


async def test_local_manager_draw_messages_consumes(manager: LocalSteeringManager) -> None:
    """Test that draw_messages consumes the buffer."""
    await manager.enqueue("message 1")
    await manager.enqueue("message 2")
    assert len(manager) == 2

    messages = await manager.draw_messages()
    assert len(messages) == 2
    assert messages[0].prompt == "message 1"
    assert messages[1].prompt == "message 2"

    # Buffer should be empty now
    assert not manager.has_pending()
    assert len(manager) == 0

    # Second draw returns empty
    messages2 = await manager.draw_messages()
    assert messages2 == []


async def test_local_manager_buffer_size_limit(manager: LocalSteeringManager) -> None:
    """Test that buffer respects max_size."""
    # Enqueue more than max_size
    for i in range(7):
        await manager.enqueue(f"message {i}")

    # Should only have max_size messages
    assert len(manager) == 5

    # Should have the last 5 messages (oldest dropped)
    messages = await manager.draw_messages()
    prompts = [m.prompt for m in messages]
    assert prompts == ["message 2", "message 3", "message 4", "message 5", "message 6"]


async def test_local_manager_clear(manager: LocalSteeringManager) -> None:
    """Test clearing the buffer."""
    await manager.enqueue("message 1")
    await manager.enqueue("message 2")
    assert manager.has_pending()

    manager.clear()
    assert not manager.has_pending()
    assert len(manager) == 0


async def test_local_manager_empty_buffer(manager: LocalSteeringManager) -> None:
    """Test operations on empty buffer."""
    assert not manager.has_pending()
    assert len(manager) == 0

    messages = await manager.draw_messages()
    assert messages == []


async def test_local_manager_concurrent_enqueue(manager: LocalSteeringManager) -> None:
    """Test thread safety with concurrent enqueues."""

    async def enqueue_many(prefix: str, count: int) -> None:
        for i in range(count):
            await manager.enqueue(f"{prefix}-{i}")

    # Run concurrent enqueues
    await asyncio.gather(
        enqueue_many("a", 3),
        enqueue_many("b", 3),
    )

    # Should have 5 messages (max_size)
    messages = await manager.draw_messages()
    assert len(messages) == 5


def test_local_manager_max_size_property(manager: LocalSteeringManager) -> None:
    """Test max_size property."""
    assert manager.max_size == 5


def test_local_manager_repr(manager: LocalSteeringManager) -> None:
    """Test string representation."""
    repr_str = repr(manager)
    assert "LocalSteeringManager" in repr_str
    assert "pending=0" in repr_str
    assert "max_size=5" in repr_str


def test_local_manager_protocol_conformance(manager: LocalSteeringManager) -> None:
    """Test that LocalSteeringManager conforms to SteeringManager protocol."""
    assert isinstance(manager, SteeringManager)


# --- Default Manager Size Tests ---


def test_default_max_size() -> None:
    """Test default max_size is 100."""
    manager = LocalSteeringManager()
    assert manager.max_size == 100


# --- SteeringOutputGuard Tests ---


@pytest.fixture
def mock_ctx_with_steering() -> tuple:
    """Create mock RunContext with steering manager that has pending messages."""
    from unittest.mock import MagicMock

    manager = LocalSteeringManager()
    deps = MagicMock()
    deps.steering_manager = manager
    ctx = MagicMock()
    ctx.deps = deps
    return ctx, manager


@pytest.fixture
def mock_ctx_without_steering():
    """Create mock RunContext without steering manager."""
    from unittest.mock import MagicMock

    deps = MagicMock(spec=[])  # No steering_manager attribute
    ctx = MagicMock()
    ctx.deps = deps
    return ctx


async def test_guard_passes_when_no_pending(mock_ctx_with_steering: tuple) -> None:
    """Test guard passes output when no steering messages pending."""
    ctx, manager = mock_ctx_with_steering
    # No messages enqueued
    result = await steering_output_guard(ctx, "test output")
    assert result == "test output"


async def test_guard_raises_when_pending(mock_ctx_with_steering: tuple) -> None:
    """Test guard raises ModelRetry when steering messages are pending."""
    from pydantic_ai import ModelRetry

    ctx, manager = mock_ctx_with_steering
    await manager.enqueue("focus on this")

    with pytest.raises(ModelRetry) as exc_info:
        await steering_output_guard(ctx, "test output")

    assert "steering messages" in str(exc_info.value).lower()


async def test_guard_passes_without_steering_manager(mock_ctx_without_steering) -> None:
    """Test guard passes when context has no steering_manager."""
    ctx = mock_ctx_without_steering
    result = await steering_output_guard(ctx, "test output")
    assert result == "test output"
