"""Steering mechanism for injecting user guidance during agent execution.

Steering allows users to inject messages into the agent's context while
the agent is running, without interrupting execution. This is useful for:
- Providing clarification or additional context
- Redirecting the agent's approach
- Adding constraints or preferences mid-execution

Only the main agent receives steering messages; subagents execute autonomously.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic_ai import ModelRetry, RunContext

if TYPE_CHECKING:
    pass


@dataclass
class SteeringMessage:
    """A steering message to be injected into agent context.

    Attributes:
        message_id: Unique identifier for the message.
        prompt: The steering prompt content.
        timestamp: When the message was created.
    """

    message_id: str
    prompt: str
    timestamp: datetime = field(default_factory=datetime.now)


@runtime_checkable
class SteeringManager(Protocol):
    """Protocol for steering message management.

    Implementations must provide thread-safe message buffering
    with enqueue/draw semantics.
    """

    async def enqueue(self, message: str) -> SteeringMessage:
        """Add a steering message to the buffer.

        Args:
            message: The steering prompt to enqueue.

        Returns:
            The created SteeringMessage with assigned ID.
        """
        ...

    async def draw_messages(self) -> list[SteeringMessage]:
        """Draw and clear all pending steering messages.

        This is a consume operation - messages are removed from the buffer
        after being drawn. Called by the steering filter during message
        processing.

        Returns:
            List of all buffered messages (may be empty).
        """
        ...

    def has_pending(self) -> bool:
        """Check if there are pending steering messages.

        Returns:
            True if buffer contains messages.
        """
        ...

    def clear(self) -> None:
        """Clear all pending steering messages."""
        ...


class LocalSteeringManager:
    """Local in-memory steering manager for TUI.

    Uses a bounded deque for message buffering with asyncio.Lock
    for thread safety. Suitable for single-process TUI applications.

    Attributes:
        max_size: Maximum number of messages to buffer.
    """

    def __init__(self, max_size: int = 100) -> None:
        """Initialize the steering manager.

        Args:
            max_size: Maximum buffer size. Oldest messages are dropped
                     when buffer is full.
        """
        self._buffer: deque[SteeringMessage] = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
        self._max_size = max_size

    @property
    def max_size(self) -> int:
        """Maximum buffer size."""
        return self._max_size

    async def enqueue(self, message: str) -> SteeringMessage:
        """Add a steering message to the buffer.

        Thread-safe operation. If buffer is full, oldest message is dropped.

        Args:
            message: The steering prompt to enqueue.

        Returns:
            The created SteeringMessage with assigned ID.
        """
        steering = SteeringMessage(
            message_id=f"steer-{uuid.uuid4().hex[:8]}",
            prompt=message,
        )
        async with self._lock:
            self._buffer.append(steering)
        return steering

    async def draw_messages(self) -> list[SteeringMessage]:
        """Draw and clear all pending steering messages.

        Thread-safe consume operation. Returns all buffered messages
        and clears the buffer atomically.

        Returns:
            List of all buffered messages (may be empty).
        """
        async with self._lock:
            messages = list(self._buffer)
            self._buffer.clear()
        return messages

    def has_pending(self) -> bool:
        """Check if there are pending steering messages.

        Returns:
            True if buffer contains messages.
        """
        return len(self._buffer) > 0

    def clear(self) -> None:
        """Clear all pending steering messages."""
        self._buffer.clear()

    def __len__(self) -> int:
        """Return the number of pending messages."""
        return len(self._buffer)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"LocalSteeringManager(pending={len(self._buffer)}, max_size={self._max_size})"


# =============================================================================
# Steering Guard
# =============================================================================


async def steering_output_guard(ctx: RunContext[Any], output: str) -> str:
    """Output guard that checks for pending steering messages.

    This guard is used with TextOutput to trigger a ModelRetry when
    there are pending steering messages at the time of output.
    This ensures the agent processes any user guidance before completing.

    Args:
        ctx: The run context containing TUIContext with steering_manager.
        output: The text output from the agent.

    Returns:
        The output unchanged if no pending messages.

    Raises:
        ModelRetry: If there are pending steering messages.
    """
    deps = ctx.deps
    # Check if deps has steering_manager (TUIContext)
    steering_manager = getattr(deps, "steering_manager", None)
    if steering_manager is not None and steering_manager.has_pending():
        raise ModelRetry("User added steering messages, please continue the task with the new instructions.")
    return output
