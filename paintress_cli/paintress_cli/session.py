"""TUI session management with steering capability.

TUIContext extends AgentContext with:
- Steering message injection (user can guide agent during execution)
- Built-in history processor for steering
- Event emission via inherited agent_stream_queues

The steering filter is integrated into get_history_processors() and
emits SteeringInjectedEvent when messages are injected.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from pydantic import PrivateAttr
from pydantic_ai import RunContext
from pydantic_ai._agent_graph import HistoryProcessor
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    UserPromptPart,
)

from pai_agent_sdk.context import AgentContext
from paintress_cli.events import SteeringInjectedEvent
from paintress_cli.logging import get_logger
from paintress_cli.steering import LocalSteeringManager, SteeringMessage

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Steering Message Rendering
# -----------------------------------------------------------------------------


def render_steering_messages(messages: list[SteeringMessage]) -> list[ModelRequestPart]:
    """Render steering messages as user prompt parts.

    Formats multiple steering messages into a single XML-structured
    prompt that the agent can interpret.

    Args:
        messages: List of steering messages to render.

    Returns:
        List containing a single UserPromptPart with formatted content.
    """
    prompts = "\n".join([m.prompt for m in messages])
    content = f"""<steering>
{prompts}
</steering>

<system-reminder>
The user has provided additional guidance during task execution.
Review the <steering> content carefully, consider how it affects your current approach,
and adjust your work accordingly while continuing toward the goal.
</system-reminder>
"""
    return [UserPromptPart(content=content)]


# -----------------------------------------------------------------------------
# TUIContext
# -----------------------------------------------------------------------------


class TUIContext(AgentContext):
    """TUI context with steering capability.

    Extends AgentContext with:
    - LocalSteeringManager for buffering steering messages
    - Built-in history processor for injecting steering into agent context
    - Event emission via agent_stream_queues

    Steering is always enabled - users can enqueue messages via steering_manager
    while the agent is running, and they will be injected on the next LLM call.

    Example:
        async with TUIContext(env=env) as ctx:
            # Enqueue steering while agent runs
            await ctx.steering_manager.enqueue("Focus on performance")

            # The steering filter automatically injects into next LLM call
            # and emits SteeringInjectedEvent to agent_stream_queues
    """

    _steering_manager: LocalSteeringManager = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        """Initialize TUIContext with steering manager."""
        super().__init__(**data)
        object.__setattr__(self, "_steering_manager", LocalSteeringManager())

    @property
    def steering_manager(self) -> LocalSteeringManager:
        """Access the steering manager for enqueueing messages."""
        return self._steering_manager

    def get_history_processors(self) -> list:
        """Return history processors including steering injection.

        Adds the steering filter after base AgentContext processors.

        Returns:
            List of history processor functions.
        """
        processors = super().get_history_processors()
        processors.append(create_steering_filter(self))
        return processors

    def create_subagent_context(
        self,
        agent_name: str,
        agent_id: str | None = None,
        **override: Any,
    ) -> AgentContext:
        """Create subagent context without steering.

        Subagents do not receive steering - they execute autonomously.
        Returns a plain AgentContext, not TUIContext.

        Args:
            agent_name: Name of the subagent.
            agent_id: ID for the subagent.
            **override: Additional fields to override.

        Returns:
            Plain AgentContext for subagent (no steering capability).
        """
        # Use parent's method which returns AgentContext
        return super().create_subagent_context(
            agent_name=agent_name,
            agent_id=agent_id,
            **override,
        )


# -----------------------------------------------------------------------------
# Steering Filter Factory
# -----------------------------------------------------------------------------


def create_steering_filter(
    context: TUIContext,
) -> HistoryProcessor[AgentContext]:
    """Create a history processor that injects steering messages.

    This factory creates an async function that:
    1. Checks for pending steering messages in context._steering_manager
    2. Draws messages from buffer (consuming them)
    3. Appends to the last ModelRequest
    4. Emits SteeringInjectedEvent via context

    Args:
        context: The TUIContext containing the steering manager.

    Returns:
        An async HistoryProcessor function with proper signature.
    """

    async def inject_steering(
        ctx: RunContext[AgentContext],
        message_history: list[ModelMessage],
    ) -> list[ModelMessage]:
        """Inject pending steering messages into message history."""
        # Only inject into requests (not responses)
        if not message_history or not isinstance(message_history[-1], ModelRequest):
            return message_history

        # Check for pending messages (fast path)
        if not context._steering_manager.has_pending():
            return message_history

        # Draw pending messages
        try:
            steering_messages = await context._steering_manager.draw_messages()
        except Exception:
            logger.exception("Failed to draw steering messages")
            return message_history

        if not steering_messages:
            return message_history

        # Inject into the last request
        rendered = render_steering_messages(steering_messages)
        message_history[-1] = ModelRequest(
            parts=[*message_history[-1].parts, *rendered],
        )

        logger.debug(
            "Injected %d steering message(s): %s",
            len(steering_messages),
            steering_messages[0].prompt[:50] if steering_messages else "",
        )

        # Emit event via context (which is TUIContext)
        # Include full content for user audit
        full_content = "\n".join(m.prompt for m in steering_messages)

        # Add to user_prompts for compact/handoff recovery
        steering_text = f"<steering>\n{full_content}\n</steering>"
        if context.user_prompts is None:
            context.user_prompts = steering_text
        elif isinstance(context.user_prompts, str):
            context.user_prompts = context.user_prompts + "\n\n" + steering_text
        else:
            # Sequence[UserContent] - convert to list and append string
            context.user_prompts = [*context.user_prompts, steering_text]

        event = SteeringInjectedEvent(
            event_id=f"steer-{uuid.uuid4().hex[:8]}",
            message_count=len(steering_messages),
            content=full_content,
        )
        await context.emit_event(event)

        return message_history

    return inject_steering
