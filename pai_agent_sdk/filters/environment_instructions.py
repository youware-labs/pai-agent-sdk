"""Environment instructions history processor factory.

This module provides a factory function that creates a history processor
for injecting environment context instructions (file system, shell configuration)
into the message history before each model request.

Example::

    from contextlib import AsyncExitStack
    from pydantic_ai import Agent

    from pai_agent_sdk.context import AgentContext
    from pai_agent_sdk.environment.local import LocalEnvironment
    from pai_agent_sdk.filters.environment_instructions import create_environment_instructions_filter

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(LocalEnvironment())
        ctx = await stack.enter_async_context(
            AgentContext(env=env)
        )
        env_filter = create_environment_instructions_filter(env)
        agent = Agent(
            'openai:gpt-4',
            deps_type=AgentContext,
            history_processors=[env_filter],
        )
        result = await agent.run('Your prompt here', deps=ctx)
"""

from collections.abc import Awaitable, Callable
from typing import Any

from agent_environment import Environment
from pydantic_ai import RetryPromptPart
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai.tools import RunContext


def create_environment_instructions_filter(
    env: Environment,
) -> Callable[[RunContext[Any], list[ModelMessage]], Awaitable[list[ModelMessage]]]:
    """Create a history processor that injects environment instructions.

    This factory function creates a pydantic-ai history_processor that appends
    environment context instructions (file system paths, shell configuration)
    to the last ModelRequest in the message history.

    Args:
        env: Environment instance to get context instructions from.

    Returns:
        A history processor function compatible with pydantic-ai Agent.

    Example:
        env_filter = create_environment_instructions_filter(env)
        agent = Agent(
            'openai:gpt-4',
            history_processors=[env_filter],
        )
    """

    async def inject_environment_instructions(
        ctx: RunContext[Any],
        message_history: list[ModelMessage],
    ) -> list[ModelMessage]:
        """Inject environment instructions into the last ModelRequest.

        Args:
            ctx: Runtime context (not used, but required by history_processor signature).
            message_history: Current message history to process.

        Returns:
            Processed message history with environment instructions injected into
            the last ModelRequest, or unchanged history if no ModelRequest found.
        """
        _ = ctx  # Unused, but required by history_processor signature

        # Find the last ModelRequest in message history
        last_request: ModelRequest | None = None
        for msg in reversed(message_history):
            if isinstance(msg, ModelRequest):
                last_request = msg
                break

        if not last_request:
            return message_history

        # Skip injection if last_request contains ToolReturnPart (tool response)
        # We only inject environment instructions on user input, not tool responses or retry prompts
        if any(isinstance(part, (ToolReturnPart, RetryPromptPart)) for part in last_request.parts):
            return message_history

        # Get environment instructions
        instructions = await env.get_context_instructions()

        if not instructions:
            return message_history

        # Append environment instructions as a UserPromptPart
        env_part = UserPromptPart(content=instructions)
        last_request.parts = [*last_request.parts, env_part]

        return message_history

    return inject_environment_instructions
