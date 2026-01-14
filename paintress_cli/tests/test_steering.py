"""Tests for steering module."""

from __future__ import annotations

import asyncio

import pytest
from paintress_cli.steering import (
    LocalSteeringManager,
    SteeringManager,
    SteeringMessage,
)


class TestSteeringMessage:
    """Tests for SteeringMessage dataclass."""

    def test_create_message(self):
        """Test basic message creation."""
        msg = SteeringMessage(message_id="test-123", prompt="focus on tests")
        assert msg.message_id == "test-123"
        assert msg.prompt == "focus on tests"
        assert msg.timestamp is not None

    def test_message_with_custom_timestamp(self):
        """Test message with custom timestamp."""
        from datetime import datetime

        ts = datetime(2026, 1, 14, 12, 0, 0)
        msg = SteeringMessage(message_id="test-456", prompt="hello", timestamp=ts)
        assert msg.timestamp == ts


class TestLocalSteeringManager:
    """Tests for LocalSteeringManager."""

    @pytest.fixture
    def manager(self) -> LocalSteeringManager:
        """Create a fresh manager for each test."""
        return LocalSteeringManager(max_size=5)

    @pytest.mark.asyncio
    async def test_enqueue_message(self, manager: LocalSteeringManager):
        """Test enqueuing a message."""
        msg = await manager.enqueue("focus on the UI")
        assert msg.prompt == "focus on the UI"
        assert msg.message_id.startswith("steer-")
        assert len(msg.message_id) == 14  # "steer-" + 8 hex chars
        assert manager.has_pending()
        assert len(manager) == 1

    @pytest.mark.asyncio
    async def test_draw_messages_consumes(self, manager: LocalSteeringManager):
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

    @pytest.mark.asyncio
    async def test_buffer_size_limit(self, manager: LocalSteeringManager):
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

    @pytest.mark.asyncio
    async def test_clear(self, manager: LocalSteeringManager):
        """Test clearing the buffer."""
        await manager.enqueue("message 1")
        await manager.enqueue("message 2")
        assert manager.has_pending()

        manager.clear()
        assert not manager.has_pending()
        assert len(manager) == 0

    @pytest.mark.asyncio
    async def test_empty_buffer(self, manager: LocalSteeringManager):
        """Test operations on empty buffer."""
        assert not manager.has_pending()
        assert len(manager) == 0

        messages = await manager.draw_messages()
        assert messages == []

    @pytest.mark.asyncio
    async def test_concurrent_enqueue(self, manager: LocalSteeringManager):
        """Test thread safety with concurrent enqueues."""

        async def enqueue_many(prefix: str, count: int):
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

    def test_max_size_property(self, manager: LocalSteeringManager):
        """Test max_size property."""
        assert manager.max_size == 5

    def test_repr(self, manager: LocalSteeringManager):
        """Test string representation."""
        repr_str = repr(manager)
        assert "LocalSteeringManager" in repr_str
        assert "pending=0" in repr_str
        assert "max_size=5" in repr_str

    def test_protocol_conformance(self, manager: LocalSteeringManager):
        """Test that LocalSteeringManager conforms to SteeringManager protocol."""
        assert isinstance(manager, SteeringManager)


class TestDefaultManagerSize:
    """Tests for default manager configuration."""

    def test_default_max_size(self):
        """Test default max_size is 10."""
        manager = LocalSteeringManager()
        assert manager.max_size == 10
