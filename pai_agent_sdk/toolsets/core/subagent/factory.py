"""Factory functions for creating subagent tools.

This module provides:
- SubagentCallFunc: Protocol for subagent call functions
- create_subagent_tool: Create BaseTool from a SubagentCallFunc
- create_subagent_call_func: Create SubagentCallFunc from a pydantic-ai Agent
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Container
from typing import Annotated, Any, Protocol, cast, runtime_checkable
from uuid import uuid4

from pydantic import Field
from pydantic_ai import Agent, AgentRunResult, RunContext
from pydantic_ai.usage import RunUsage

from pai_agent_sdk.context import AgentContext, ModelConfig
from pai_agent_sdk.toolsets.core.base import BaseTool


@runtime_checkable
class SubagentCallFunc(Protocol):
    """Protocol for subagent call functions.

    The first parameter must be AgentContext (the subagent context),
    followed by user-defined parameters. Returns (output, RunUsage) tuple.
    """

    async def __call__(self, ctx: AgentContext, /, **kwargs: Any) -> tuple[Any, RunUsage]: ...


# Type alias for instruction functions
InstructionFunc = Callable[[RunContext[AgentContext]], str | None]

# Type alias for availability check functions
AvailabilityCheckFunc = Callable[[], bool]


def create_subagent_tool(
    name: str,
    description: str,
    call_func: SubagentCallFunc,
    *,
    instruction: str | InstructionFunc | None = None,
    availability_check: AvailabilityCheckFunc | None = None,
    model_cfg: ModelConfig | None = None,
) -> type[BaseTool]:
    """Create a BaseTool subclass that wraps a subagent call function.

    This factory function creates a tool class that:
    - Uses the call_func's parameter signature (excluding ctx) as tool parameters
    - Automatically records RunUsage to ctx.deps.extra_usage
    - Converts the output to string for LLM consumption

    For streaming, use ctx.deps.agent_stream_queues[tool_call_id] to send events.

    Args:
        name: Tool name used for invocation.
        description: Tool description shown to the model.
        call_func: Async function with signature (ctx, **kwargs) -> (output, RunUsage).
                   The function's parameters (after ctx) define the tool's input schema.
        instruction: Optional instruction for system prompt. Can be a string or
                     a callable that takes RunContext and returns a string.
        availability_check: Optional callable that returns True if the tool is available.
                            Called dynamically each time is_available() is invoked.
        model_cfg: Optional ModelConfig to override in subagent context.
                   If None, subagent inherits parent's model_cfg.

    Returns:
        A BaseTool subclass that can be used with Toolset.

    Example::

        async def search(
            ctx: AgentContext,  # This is the subagent context (auto-created)
            query: str,
            max_results: int = 10,
        ) -> tuple[str, RunUsage]:
            agent = get_search_agent()
            result = await agent.run(f"Search: {query}, max: {max_results}", deps=ctx)
            return str(result.output), result.usage()

        SearchTool = create_subagent_tool(
            name="search",
            description="Search the web for information",
            call_func=search,
            instruction="Use this tool to search for current information.",
        )

        # For streaming, access the parent context's stream queue:
        async def search_with_stream(
            ctx: AgentContext,
            query: str,
        ) -> tuple[str, RunUsage]:
            # ctx.parent_run_id is the tool_call_id
            # Access parent's stream queue via the shared reference
            agent = get_search_agent()
            async for event in agent.run_stream_events(query, deps=ctx):
                # Forward events as needed
                pass
            result = await agent.run(query, deps=ctx)
            return str(result.output), result.usage()
    """

    class DynamicSubagentTool(BaseTool):
        """Dynamically created subagent tool."""

        # These will be set by the closure
        name = ""  # Placeholder, will be overwritten
        description = ""  # Placeholder, will be overwritten

        def __init__(self, ctx: AgentContext) -> None:
            super().__init__(ctx)

        def is_available(self) -> bool:
            if availability_check is None:
                return True
            return availability_check()

        def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
            if instruction is None:
                return None
            if callable(instruction):
                return instruction(ctx)
            return instruction

        async def call(self, ctx: RunContext[AgentContext], /, **kwargs: Any) -> str:
            # Placeholder - will be replaced by _create_call_method
            raise NotImplementedError  # pragma: no cover

    # Set class attributes from closure variables
    DynamicSubagentTool.name = name
    DynamicSubagentTool.description = description

    # Copy the call signature from call_func to DynamicSubagentTool.call
    # This allows pydantic-ai to inspect the correct parameters
    DynamicSubagentTool.call = _create_call_method(call_func, model_cfg=model_cfg)  # type: ignore[method-assign]

    # Set a meaningful class name for debugging
    DynamicSubagentTool.__name__ = f"{_to_pascal_case(name)}Tool"
    DynamicSubagentTool.__qualname__ = DynamicSubagentTool.__name__

    return DynamicSubagentTool


def _create_call_method(
    call_func: SubagentCallFunc,
    *,
    model_cfg: ModelConfig | None = None,
) -> Callable[..., Awaitable[str]]:
    """Create a call method with the correct signature from call_func.

    The first parameter (ctx: AgentContext) is replaced with RunContext[AgentContext]
    for pydantic-ai compatibility. The actual call_func receives the subagent context.
    """

    async def call(self: BaseTool, ctx: RunContext[AgentContext], /, **kwargs: Any) -> str:
        """Execute the subagent call and record usage."""
        override_kwargs: dict[str, Any] = {}
        if model_cfg is not None:
            override_kwargs["model_cfg"] = model_cfg
        async with ctx.deps.enter_subagent(self.name, agent_id=ctx.tool_call_id, **override_kwargs) as sub_ctx:
            output, usage = await call_func(sub_ctx, **kwargs)

        # Record usage in extra_usages
        if ctx.tool_call_id:
            ctx.deps.add_extra_usage(agent=self.name, usage=usage, uuid=ctx.tool_call_id)

        # Convert output to string for LLM
        return str(output)

    # Copy the signature from call_func, but replace ctx type
    original_sig = inspect.signature(call_func)
    params = list(original_sig.parameters.values())

    # Replace first param (ctx: AgentContext) with (ctx: RunContext[AgentContext])
    if params:
        first_param = params[0]
        new_first_param = first_param.replace(annotation=RunContext[AgentContext])
        params[0] = new_first_param

    # Create new signature with RunContext and str return type
    new_sig = original_sig.replace(parameters=params, return_annotation=str)
    call.__signature__ = new_sig  # type: ignore[attr-defined]

    # Copy docstring and name
    call.__doc__ = call_func.__doc__ or "Execute the subagent call and record usage."
    call.__name__ = "call"
    call.__qualname__ = "call"

    # Build annotations with RunContext[AgentContext] for the first param
    original_annotations = getattr(call_func, "__annotations__", {})
    new_annotations: dict[str, Any] = {}
    first_param_name = params[0].name if params else "ctx"
    for key, value in original_annotations.items():
        if key == first_param_name:
            # Replace ctx: AgentContext with ctx: RunContext[AgentContext]
            new_annotations[key] = RunContext[AgentContext]
        elif key != "return":
            new_annotations[key] = value
    new_annotations["return"] = str
    call.__annotations__ = new_annotations

    return call


def _to_pascal_case(name: str) -> str:
    """Convert snake_case or kebab-case to PascalCase."""
    parts = name.replace("-", "_").split("_")
    return "".join(part.capitalize() for part in parts)


def generate_unique_id(run_id: str, existing: Container[str], *, max_retries: int = 10) -> str:
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
    agent_id = run_id[-4:]
    if agent_id not in existing:
        return agent_id

    for _ in range(max_retries):
        agent_id = uuid4().hex[:4]
        if agent_id not in existing:
            return agent_id

    raise RuntimeError(f"Failed to generate unique agent_id after {max_retries} retries")


def create_subagent_call_func(
    agent: Agent[AgentContext, Any],
) -> SubagentCallFunc:
    """Create a SubagentCallFunc from a pydantic-ai Agent.

    Wraps a pydantic-ai Agent into a SubagentCallFunc that can be used
    directly or passed to create_subagent_tool().

    Args:
        agent: A pydantic-ai Agent with AgentContext as deps type.

    Returns:
        A SubagentCallFunc with signature (ctx: AgentContext, prompt: str) -> tuple[Any, RunUsage]

    Example::

        from pydantic_ai import Agent

        search_agent: Agent[AgentContext, str] = Agent(...)

        # Create the call function
        search_func = create_subagent_call_func(search_agent)

        # Direct usage
        output, usage = await search_func(ctx, prompt="Search for Python tutorials")

        # Or pass to create_subagent_tool
        SearchTool = create_subagent_tool("search", "Search the web", search_func)
    """

    async def call_func(
        ctx: AgentContext,
        prompt: Annotated[str, Field(description="The prompt to send to the subagent")],
        agent_id: Annotated[str | None, Field(description="Optional agent ID to resume")] = None,
    ) -> tuple[str, RunUsage]:
        """Execute the agent with the given prompt."""
        if not agent_id:
            agent_id = generate_unique_id(ctx.run_id, ctx.subagent_history)
        async with agent.iter(prompt, deps=ctx, message_history=ctx.subagent_history.get(agent_id)) as run:
            async for node in run:
                if Agent.is_user_prompt_node(node) or Agent.is_end_node(node):
                    continue

                elif Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                    async with node.stream(run.ctx) as request_stream:
                        async for event in request_stream:
                            await ctx.emit_event(event)

        result = cast(AgentRunResult, run.result)
        ctx.subagent_history[agent_id] = result.all_messages()

        rendered_result = f"""<id>{agent_id}</id>
<response>{result.output}</response>
"""
        return rendered_result, result.usage()

    return call_func  # type: ignore[return-value]
