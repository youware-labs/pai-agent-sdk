"""Bus message injection filter.

This filter injects pending bus messages into the conversation
at the start of each LLM request, enabling real-time communication
between user and agents, or between agents.
"""

from __future__ import annotations

import uuid

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.events import BusMessageInfo, MessageReceivedEvent


async def inject_bus_messages(
    ctx: RunContext[AgentContext],
    messages: list[ModelMessage],
) -> list[ModelMessage]:
    """Inject pending bus messages into the conversation.

    This filter consumes pending messages from the message bus
    and injects them as user prompt parts into the last ModelRequest.
    Messages are consumed (removed from bus) after injection.

    Injection:
        Messages are rendered using their template and appended
        to the last ModelRequest's parts list.

    Filter Order:
        This filter should run BEFORE inject_runtime_instructions
        to ensure messages are visible before runtime context.

    Args:
        ctx: Run context containing AgentContext.
        messages: Current message history.

    Returns:
        Modified message history with injected bus messages.
    """
    if not messages or not isinstance(messages[-1], ModelRequest):
        return messages

    # Consume messages for current agent
    agent_id = ctx.deps._agent_id
    pending = ctx.deps.message_bus.consume(agent_id)

    if not pending:
        return messages

    # Render messages with structured format
    parts = [
        UserPromptPart(content=f'<bus-message source="{msg.source}">\n{msg.render()}\n</bus-message>')
        for msg in pending
    ]

    # Emit single event with all messages
    event = MessageReceivedEvent(
        event_id=f"bus-recv-{uuid.uuid4().hex[:8]}",
        messages=[
            BusMessageInfo(
                content=msg.content,
                source=msg.source,
                target=msg.target,
                template=msg.template,
            )
            for msg in pending
        ],
    )
    await ctx.deps.emit_event(event)

    # Inject into last message's parts
    messages[-1].parts = [*messages[-1].parts, *parts]

    return messages
