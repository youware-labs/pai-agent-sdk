"""TUI session management.

TUIContext extends AgentContext as an empty shell for future TUI-specific
extensions. Currently all functionality is provided by the base AgentContext,
including message bus for user steering.

The message bus is used for injecting user guidance during agent execution:
- User sends messages via ctx.send_message("guidance", source="user")
- SDK's inject_bus_messages filter handles injection
- SDK's message_bus_guard ensures messages are processed before completion
"""

from __future__ import annotations

from typing import Any

from pai_agent_sdk.context import AgentContext


class TUIContext(AgentContext):
    """TUI context extending AgentContext.

    Currently an empty shell that inherits all functionality from AgentContext.
    The message bus (inherited) handles user steering:
    """

    def __init__(self, **data: Any) -> None:
        """Initialize TUIContext."""
        super().__init__(**data)
