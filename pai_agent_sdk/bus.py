"""Message bus for inter-agent communication.

This module provides a message bus for communication between:
- User -> main agent (replacing CLI steering)
- Main agent <-> subagent (bidirectional)
- Broadcast messages (to current active agent)

Messages are one-time consumption - once delivered to an agent,
they are removed from the bus.

Example:
    Basic usage with AgentContext::

        from pai_agent_sdk.context import AgentContext

        # Messages are sent via AgentContext.send_message()
        ctx.send_message("Please prioritize task A", source="user")

    Direct MessageBus usage (advanced)::

        from pai_agent_sdk.bus import MessageBus

        bus = MessageBus()
        bus.send("Hello", source="user", target="main")
        messages = bus.consume("main")  # Returns and removes messages
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
        content: The message content (typically markdown text).
        source: Who sent the message (e.g., "user", agent_id).
        target: Who should receive the message (agent_id, or None for current active agent).
        template: Jinja2 template string for rendering. Use {{ content }} placeholder. None means raw content.
        timestamp: When the message was created.
    """

    content: str
    source: str
    target: str | None = None
    template: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def render(self) -> str:
        """Render the message using its Jinja2 template, or return raw content if no template."""
        return render_template(self.content, self.template)


class MessageBus:
    """Message bus for inter-agent communication.

    The message bus supports:
    - User -> main agent messages (replacing CLI steering)
    - Main agent -> subagent messages
    - Subagent -> main agent messages
    - Broadcast messages (target=None, delivered to current active agent)

    Messages are one-time consumption - once consumed by an agent,
    they are removed from the bus.

    Thread Safety:
        This class is NOT thread-safe. It's designed to be used within
        a single asyncio event loop. For concurrent access from multiple
        coroutines, the consume operation is atomic (no await between
        read and remove).

    Example:
        >>> bus = MessageBus()
        >>> bus.send("Hello", source="user", target="main")
        >>> bus.send("World", source="user", target="main")
        >>> messages = bus.consume("main")
        >>> [m.content for m in messages]
        ['Hello', 'World']  # FIFO order
    """

    def __init__(self) -> None:
        """Initialize an empty message bus."""
        self._messages: list[BusMessage] = []

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
            target: Who should receive the message, or None for current active agent.
            template: Jinja2 template string for rendering. Use {{ content }} placeholder. None means raw content.

        Returns:
            The created BusMessage instance.
        """
        message = BusMessage(
            content=content,
            source=source,
            target=target,
            template=template,
        )
        self._messages.append(message)
        return message

    def consume(self, agent_id: str) -> list[BusMessage]:
        """Consume all messages targeted at the given agent.

        Messages with target=None (broadcast) or target=agent_id are consumed.
        Consumed messages are removed from the bus.

        Args:
            agent_id: The agent ID to consume messages for.

        Returns:
            List of messages for this agent, in FIFO order (oldest first).
        """
        matched: list[BusMessage] = []
        remaining: list[BusMessage] = []

        for msg in self._messages:
            if msg.target is None or msg.target == agent_id:
                matched.append(msg)
            else:
                remaining.append(msg)

        self._messages = remaining
        return matched

    def has_pending(self, agent_id: str) -> bool:
        """Check if there are pending messages for the given agent.

        Args:
            agent_id: The agent ID to check.

        Returns:
            True if there are pending messages (target=None or target=agent_id).
        """
        return any(msg.target is None or msg.target == agent_id for msg in self._messages)

    def peek(self, agent_id: str) -> list[BusMessage]:
        """Peek at pending messages without consuming them.

        Args:
            agent_id: The agent ID to check.

        Returns:
            List of messages for this agent (not removed from bus).
        """
        return [msg for msg in self._messages if msg.target is None or msg.target == agent_id]

    def clear(self) -> None:
        """Clear all messages from the bus."""
        self._messages.clear()

    def __len__(self) -> int:
        """Return the total number of pending messages."""
        return len(self._messages)

    def __bool__(self) -> bool:
        """Return True if there are any pending messages."""
        return len(self._messages) > 0
