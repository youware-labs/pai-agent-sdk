"""Handoff message history processor.

This module provides a history processor that injects handoff summaries
into the message history when a context reset occurs.

Note:
    This processor must be used together with `pai_agent_sdk.toolsets.context.handoff.HandoffTool`.
    See HandoffTool for usage example.
"""

from uuid import uuid4

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.events import HandoffCompleteEvent, HandoffFailedEvent, HandoffStartEvent


async def process_handoff_message(
    ctx: RunContext[AgentContext],
    message_history: list[ModelMessage],
) -> list[ModelMessage]:
    """Inject handoff summary into message history after context reset.

    This is a pydantic-ai history_processor that can be passed to Agent's
    history_processors parameter. When a handoff occurs, the previous context
    is cleared but a summary message is preserved in ctx.deps.handoff_message.

    Note: Subagents created via enter_subagent() have handoff_message cleared,
    so they won't be affected by the main agent's handoff state.

    Args:
        ctx: Runtime context containing AgentContext with handoff_message.
        message_history: Current message history to process.

    Returns:
        Processed message history with handoff summary injected, or unchanged
        history if no handoff is pending.

    Example:
        agent = Agent(
            'openai:gpt-4',
            deps_type=AgentContext,
            history_processors=[process_handoff_message],
        )
    """
    if not ctx.deps.handoff_message:
        return message_history

    # Generate event_id to correlate start/complete events
    event_id = uuid4().hex[:8]
    agent_ctx = ctx.deps
    original_message_count = len(message_history)

    try:
        # Emit start event
        await agent_ctx.emit_event(HandoffStartEvent(event_id=event_id, message_count=original_message_count))

        # Find the last true user input ModelRequest (has UserPromptPart, no ToolReturnPart)
        last_user_request: ModelRequest | None = None
        for msg in reversed(message_history):
            if not isinstance(msg, ModelRequest):
                continue
            has_user_prompt = any(isinstance(p, UserPromptPart) for p in msg.parts)
            has_tool_return = any(isinstance(p, ToolReturnPart) for p in msg.parts)
            if has_user_prompt and not has_tool_return:
                last_user_request = msg
                break

        if not last_user_request:
            return message_history

        handoff_content = ctx.deps.handoff_message

        # Append handoff summary after user's current request
        handoff_part = UserPromptPart(
            content=f"<context-handoff>\n{handoff_content}\n</context-handoff>",
        )
        last_user_request.parts = [*last_user_request.parts, handoff_part]

        # Generate a unique tool call id
        tool_call_id = f"handoff-{ctx.deps.run_id}"

        # Clear handoff state
        ctx.deps.handoff_message = None

        # Return truncated history with handoff marker
        result = [
            last_user_request,
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_call_id=tool_call_id,
                        tool_name="handoff",
                        args={"_": "context-reset"},
                    ),
                ],
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_call_id=tool_call_id,
                        tool_name="handoff",
                        content="Handoff complete. Continue with the task using the context summary above.",
                    ),
                ],
            ),
        ]

        # Emit complete event with the actual handoff content
        await agent_ctx.emit_event(
            HandoffCompleteEvent(
                event_id=event_id,
                handoff_content=handoff_content,
                original_message_count=original_message_count,
            )
        )

        return result

    except Exception as e:
        # Emit failed event so consumers know handoff did not succeed
        await agent_ctx.emit_event(
            HandoffFailedEvent(event_id=event_id, error=str(e), message_count=original_message_count)
        )
        # On error, return original history
        return message_history
