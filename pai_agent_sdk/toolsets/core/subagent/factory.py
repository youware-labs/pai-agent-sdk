"""Factory functions for creating subagent tools.

This module provides:
- create_subagent_tool: Create BaseTool from a call function
- create_subagent_call_func: Create a BaseTool.call compatible function from a pydantic-ai Agent
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Container
from inspect import isawaitable
from typing import Annotated, Any, cast
from uuid import uuid4

from pydantic import Field
from pydantic_ai import Agent, AgentRunResult, RunContext, UsageLimits
from pydantic_ai.models import Model

from pai_agent_sdk.context import AgentContext, ModelConfig
from pai_agent_sdk.events import SubagentCompleteEvent, SubagentStartEvent
from pai_agent_sdk.toolsets.core.base import BaseTool
from pai_agent_sdk.usage import InternalUsage

# Type alias for instruction functions
InstructionFunc = Callable[[RunContext[AgentContext]], str | None]

# Type alias for availability check functions
AvailabilityCheckFunc = Callable[[RunContext[AgentContext]], bool]

# Type alias for BaseTool.call compatible function
SubagentCallFunc = Callable[..., Awaitable[str]]


def create_subagent_tool(
    name: str,
    description: str,
    call_func: SubagentCallFunc,
    *,
    instruction: str | InstructionFunc | None = None,
    availability_check: AvailabilityCheckFunc | None = None,
) -> type[BaseTool]:
    """Create a BaseTool subclass that wraps a subagent call function.

    This factory function creates a tool class that uses the provided call_func
    directly as the tool's call method. The call_func should have a signature
    compatible with BaseTool.call: (ctx: RunContext[AgentContext], **kwargs) -> str

    Use create_subagent_call_func() to create a compatible call_func from a
    pydantic-ai Agent.

    Args:
        name: Tool name used for invocation.
        description: Tool description shown to the model.
        call_func: Async function with signature (ctx: RunContext[AgentContext], **kwargs) -> str.
                   Use create_subagent_call_func() to create this from an Agent.
        instruction: Optional instruction for system prompt. Can be a string or
                     a callable that takes RunContext and returns a string.
        availability_check: Optional callable that returns True if the tool is available.
                            Called dynamically each time is_available() is invoked.

    Returns:
        A BaseTool subclass that can be used with Toolset.

    Example::

        from pydantic_ai import Agent

        # Create an agent
        search_agent: Agent[AgentContext, str] = Agent(...)

        # Create the call function using create_subagent_call_func
        search_call = create_subagent_call_func(search_agent)

        # Create the tool
        SearchTool = create_subagent_tool(
            name="search",
            description="Search the web for information",
            call_func=search_call,
            instruction="Use this tool to search for current information.",
        )
    """

    class DynamicSubagentTool(BaseTool):
        """Dynamically created subagent tool."""

        # These will be set by the closure
        name = ""  # Placeholder, will be overwritten
        description = ""  # Placeholder, will be overwritten

        def is_available(self, ctx: RunContext[AgentContext]) -> bool:
            if availability_check is None:
                return True
            return availability_check(ctx)

        def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
            if instruction is None:
                return None
            if callable(instruction):
                return instruction(ctx)
            return instruction

        async def call(self, ctx: RunContext[AgentContext], /, **kwargs: Any) -> str:
            # Placeholder - will be replaced by actual call_func
            raise NotImplementedError  # pragma: no cover

    # Set class attributes from closure variables
    DynamicSubagentTool.name = name
    DynamicSubagentTool.description = description

    # Use call_func directly as the call method
    # call_func should already have the correct signature from create_subagent_call_func
    DynamicSubagentTool.call = call_func  # type: ignore[method-assign]

    # Set a meaningful class name for debugging
    DynamicSubagentTool.__name__ = f"{_to_pascal_case(name)}Tool"
    DynamicSubagentTool.__qualname__ = DynamicSubagentTool.__name__

    return DynamicSubagentTool


def _to_pascal_case(name: str) -> str:
    """Convert snake_case or kebab-case to PascalCase."""
    parts = name.replace("-", "_").split("_")
    return "".join(part.capitalize() for part in parts)


async def _run_subagent_iter(
    agent: Agent[AgentContext, Any],
    sub_ctx: AgentContext,
    prompt: str,
    message_history: list[Any] | None,
) -> AgentRunResult:
    """Run subagent iteration and stream events to subagent's queue.

    Events are emitted to sub_ctx (subagent context) so they go to the
    subagent's queue keyed by agent_id, not the parent's queue.

    Args:
        agent: The subagent to run.
        sub_ctx: Subagent's context (events emitted here).
        prompt: The prompt to send to the subagent.
        message_history: Optional conversation history for resume.

    Returns:
        AgentRunResult from the subagent execution.
    """
    async with agent.iter(
        prompt,
        deps=sub_ctx,
        usage_limits=UsageLimits(request_limit=1000),
        message_history=message_history,
    ) as run:
        async for node in run:
            if Agent.is_user_prompt_node(node) or Agent.is_end_node(node):
                continue

            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as request_stream:
                    async for event in request_stream:
                        await sub_ctx.emit_event(event)

    return cast(AgentRunResult, run.result)


def generate_unique_id(existing: Container[str], *, max_retries: int = 10) -> str:
    """Generate a unique 4-character ID with collision detection.

    First tries using the last 4 characters of run_id. If that collides
    with existing IDs, generates random UUIDs until a unique one is found.

    Args:
        run_id: The current run ID to derive initial ID from.
        existing: Container of existing IDs to check against.
        max_retries: Maximum number of UUID generation attempts (default 10).

    Returns:
        A unique 4-character ID string.

    Raises:
        RuntimeError: If unable to generate unique ID within max_retries.
    """
    for _ in range(max_retries):
        agent_id = uuid4().hex[:4]
        if agent_id not in existing:
            return agent_id

    raise RuntimeError(f"Failed to generate unique agent_id after {max_retries} retries")


def create_subagent_call_func(
    agent: Agent[AgentContext, Any],
    *,
    model_cfg: ModelConfig | None = None,
) -> SubagentCallFunc:
    """Create a BaseTool.call compatible function from a pydantic-ai Agent.

    This function creates a call method that:
    - Has the correct signature for BaseTool.call: (ctx: RunContext[AgentContext], **kwargs) -> str
    - Generates stable agent_id in format {agent.name}-{short_id}
    - Registers the agent in parent's agent_registry
    - Manages subagent_history for conversation continuity
    - Records usage in extra_usages
    - Streams events to parent context

    Args:
        agent: A pydantic-ai Agent with AgentContext as deps type.
        model_cfg: Optional ModelConfig to override in subagent context.
                   If None, subagent inherits parent's model_cfg.

    Returns:
        A function compatible with BaseTool.call signature.

    Example::

        from pydantic_ai import Agent

        search_agent: Agent[AgentContext, str] = Agent(...)

        # Create the call function
        search_call = create_subagent_call_func(search_agent)

        # Pass to create_subagent_tool
        SearchTool = create_subagent_tool("search", "Search the web", search_call)
    """
    agent_name = agent.name or "subagent"

    async def call_func(
        self: BaseTool,
        ctx: RunContext[AgentContext],
        prompt: Annotated[str, Field(description="The prompt to send to the subagent")],
        agent_id: Annotated[str | None, Field(description="Optional agent ID to resume")] = None,
    ) -> str:
        """Execute the agent with the given prompt."""
        deps = ctx.deps

        # Generate stable agent_id if not provided
        if not agent_id:
            short_id = generate_unique_id(deps.subagent_history)
            agent_id = f"{agent_name}-{short_id}"

        # Create subagent context (handles registration in agent_registry)
        override_kwargs: dict[str, Any] = {}
        if model_cfg is not None:
            override_kwargs["model_cfg"] = model_cfg

        error_msg = ""
        success = True
        result_output = ""
        request_count = 0

        async with deps.create_subagent_context(agent_name, agent_id=agent_id, **override_kwargs) as sub_ctx:
            # Emit start event to subagent's queue (inside context so sub_ctx.start_at is set)
            prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
            await sub_ctx.emit_event(
                SubagentStartEvent(
                    event_id=agent_id,  # Use agent_id as event_id to correlate Start/Complete
                    agent_id=agent_id,
                    agent_name=agent_name,
                    prompt_preview=prompt_preview,
                )
            )

            # Apply model wrapper if configured
            original_model = agent.model
            if deps.model_wrapper is not None:
                wrapper_context = deps.get_wrapper_context()
                wrapped = deps.model_wrapper(cast(Model, original_model), agent_name, wrapper_context)
                agent.model = await wrapped if isawaitable(wrapped) else wrapped

            try:
                result = await _run_subagent_iter(agent, sub_ctx, prompt, deps.subagent_history.get(agent_id))
                result_output = result.output
                request_count = result.usage().requests

                # Store message history for future resume
                deps.subagent_history[agent_id] = result.all_messages()

                # Record usage in extra_usages
                if ctx.tool_call_id:
                    model_id = cast(Model, agent.model).model_name
                    deps.add_extra_usage(
                        agent=agent_id,
                        internal_usage=InternalUsage(model_id=model_id, usage=result.usage()),
                        uuid=ctx.tool_call_id,
                    )

            except Exception as e:
                success = False
                error_msg = str(e)
                raise

            finally:
                # Restore original model to avoid side effects on shared agent
                agent.model = original_model

                # Emit complete event to subagent's queue (use sub_ctx.elapsed_time for duration)
                elapsed = sub_ctx.elapsed_time
                duration = elapsed.total_seconds() if elapsed else 0.0
                result_preview = result_output[:500] + "..." if len(result_output) > 500 else result_output
                await sub_ctx.emit_event(
                    SubagentCompleteEvent(
                        event_id=agent_id,  # Same event_id as Start for correlation
                        agent_id=agent_id,
                        agent_name=agent_name,
                        success=success,
                        request_count=request_count,
                        result_preview=result_preview,
                        error=error_msg,
                        duration_seconds=duration,
                    )
                )

        # Return formatted result
        return f"""<id>{agent_id}</id>
<response>{result.output}</response>
"""

    return call_func  # type: ignore[return-value]
