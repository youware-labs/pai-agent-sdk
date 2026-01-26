"""Message bus for inter-agent communication.

This module provides a message bus for communication between agents,
inspired by Redis Streams design with subscriber cursors.

Key features:
- Subscriber-based consumption: Each subscriber maintains its own cursor
- Broadcast support: target=None delivers to all subscribers
- Bounded queue: maxlen limits memory usage, old messages auto-trimmed
- No message loss: Multiple subscribers can read the same broadcast message

Example:
    Basic usage with AgentContext::

        from pai_agent_sdk.context import AgentContext

        # Messages are sent via AgentContext.send_message()
        ctx.send_message("Please prioritize task A", source="user", target="main")

    Direct MessageBus usage (advanced)::

        from pai_agent_sdk.context import MessageBus

        bus = MessageBus(maxlen=100)
        bus.subscribe("main")
        bus.send("Hello", source="user", target="main")
        messages = bus.consume("main")  # Returns unread messages
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from jinja2 import Template


def render_template(content: str, template: str | None) -> str:
    """Render content using a Jinja2 template, or return raw content if no template.

    Args:
        content: The message content.
        template: Jinja2 template string with {{ content }} placeholder. None for raw content.

    Returns:
        Rendered string.
    """
    if template is None:
        return content
    return Template(template).render(content=content)


@dataclass
class BusMessage:
    """A message in the message bus.

    Attributes:
        id: Unique message ID (auto-assigned by MessageBus).
        content: The message content (typically markdown text).
        source: Who sent the message (e.g., "user", agent_id).
        target: Who should receive the message (agent_id, or None for broadcast).
        template: Jinja2 template string for rendering. Use {{ content }} placeholder.
        timestamp: When the message was created.
    """

    id: int
    content: str
    source: str
    target: str | None = None
    template: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def render(self) -> str:
        """Render the message using its Jinja2 template, or return raw content if no template."""
        return render_template(self.content, self.template)


class MessageBus:
    """Message bus for inter-agent communication (Redis Streams style).

    The message bus supports:
    - Targeted messages: target=agent_id for specific recipient
    - Broadcast messages: target=None for all subscribers
    - Subscriber cursors: Each subscriber tracks its read position
    - Bounded queue: Old messages auto-trimmed when maxlen exceeded

    Subscriber Lifecycle:
    - subscribe(): Register subscriber, cursor starts at current position
    - consume(): Get unread messages, advance cursor
    - unsubscribe(): Remove subscriber, free cursor (for subagents on exit)

    Thread Safety:
        This class is NOT thread-safe. It's designed to be used within
        a single asyncio event loop. For concurrent access from multiple
        coroutines, operations are atomic (no await points).

    Example:
        >>> bus = MessageBus(maxlen=100)
        >>> bus.subscribe("main")
        >>> bus.send("Hello", source="user", target="main")
        >>> bus.send("World", source="user")  # broadcast
        >>> messages = bus.consume("main")
        >>> [m.content for m in messages]
        ['Hello', 'World']
        >>> bus.consume("main")  # Already read
        []
    """

    def __init__(self, maxlen: int = 500) -> None:
        """Initialize an empty message bus.

        Args:
            maxlen: Maximum number of messages to retain. Oldest messages
                    are trimmed when exceeded. Default: 500.
        """
        self._messages: list[BusMessage] = []
        self._cursors: dict[str, int] = {}  # agent_id -> last_read_id
        self._next_id: int = 1
        self._maxlen = maxlen

    def subscribe(self, agent_id: str) -> None:
        """Register a subscriber.

        New subscribers start with cursor at the current position,
        meaning they will only receive messages sent after subscribing.

        Args:
            agent_id: Unique identifier for the subscriber.
        """
        if agent_id not in self._cursors:
            # Start from current position (only see new messages)
            self._cursors[agent_id] = self._next_id - 1

    def unsubscribe(self, agent_id: str) -> None:
        """Remove a subscriber.

        Should be called when a subagent exits to free cursor memory.

        Args:
            agent_id: The subscriber to remove.
        """
        self._cursors.pop(agent_id, None)

    def send(
        self,
        content: str,
        *,
        source: str,
        target: str | None = None,
        template: str | None = None,
    ) -> BusMessage:
        """Send a message to the bus.

        Args:
            content: The message content (markdown text).
            source: Who is sending the message (e.g., "user", agent_id).
            target: Who should receive the message. None for broadcast to all subscribers.
            template: Jinja2 template string for rendering. Use {{ content }} placeholder.

        Returns:
            The created BusMessage instance.
        """
        message = BusMessage(
            id=self._next_id,
            content=content,
            source=source,
            target=target,
            template=template,
        )
        self._next_id += 1
        self._messages.append(message)

        # Trim old messages if exceeding maxlen
        if len(self._messages) > self._maxlen:
            self._messages = self._messages[-self._maxlen :]

        return message

    def consume(self, agent_id: str) -> list[BusMessage]:
        """Consume unread messages for a subscriber.

        Returns messages that:
        1. Have id > subscriber's cursor (unread)
        2. Are targeted to this subscriber OR are broadcasts (target=None)

        Automatically subscribes if not already subscribed.

        Args:
            agent_id: The subscriber ID to consume messages for.

        Returns:
            List of unread messages, in FIFO order (oldest first).
        """
        # Auto-subscribe if not registered
        if agent_id not in self._cursors:
            self.subscribe(agent_id)

        cursor = self._cursors[agent_id]

        # Get minimum message ID in queue (for cursor correction)
        min_id = self._messages[0].id if self._messages else self._next_id

        # Correct cursor if it points to trimmed messages
        if cursor < min_id - 1:
            cursor = min_id - 1

        # Find matching unread messages
        result: list[BusMessage] = []
        for msg in self._messages:
            # Check: unread AND (broadcast OR targeted to this agent)
            if msg.id > cursor and (msg.target is None or msg.target == agent_id):
                result.append(msg)

        # Update cursor to latest message ID
        if result:
            self._cursors[agent_id] = result[-1].id
        elif self._messages:
            # Even if no matching messages, advance cursor to skip checked messages
            self._cursors[agent_id] = self._messages[-1].id

        return result

    def has_pending(self, agent_id: str) -> bool:
        """Check if there are unread messages for a subscriber.

        Args:
            agent_id: The subscriber ID to check.

        Returns:
            True if there are unread messages (broadcast or targeted).
        """
        if agent_id not in self._cursors:
            return False

        cursor = self._cursors[agent_id]
        min_id = self._messages[0].id if self._messages else self._next_id

        # Correct cursor if it points to trimmed messages
        if cursor < min_id - 1:
            cursor = min_id - 1

        return any(msg.id > cursor and (msg.target is None or msg.target == agent_id) for msg in self._messages)

    def peek(self, agent_id: str) -> list[BusMessage]:
        """Peek at unread messages without advancing cursor.

        Args:
            agent_id: The subscriber ID to check.

        Returns:
            List of unread messages (cursor not updated).
        """
        if agent_id not in self._cursors:
            return []

        cursor = self._cursors[agent_id]
        min_id = self._messages[0].id if self._messages else self._next_id

        if cursor < min_id - 1:
            cursor = min_id - 1

        return [msg for msg in self._messages if msg.id > cursor and (msg.target is None or msg.target == agent_id)]

    def clear(self) -> None:
        """Clear all messages and reset cursors."""
        self._messages.clear()
        self._cursors.clear()
        self._next_id = 1

    @property
    def subscriber_count(self) -> int:
        """Return the number of registered subscribers."""
        return len(self._cursors)

    def __len__(self) -> int:
        """Return the total number of messages in the queue."""
        return len(self._messages)

    def __bool__(self) -> bool:
        """Return True if there are any messages in the queue."""
        return len(self._messages) > 0
