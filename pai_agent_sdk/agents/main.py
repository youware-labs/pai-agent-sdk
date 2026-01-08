"""Main agent factory for creating configured agents.

This module provides the create_agent context manager for building agents
with proper environment and context lifecycle management.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic

import jinja2
from pydantic_ai import Agent
from pydantic_ai._agent_graph import CallToolsNode, HistoryProcessor, ModelRequestNode
from pydantic_ai.messages import ModelMessage, UserContent
from pydantic_ai.models import Model
from pydantic_ai.run import AgentRun
from typing_extensions import TypeVar

from pai_agent_sdk.agents.compact import create_compact_filter
from pai_agent_sdk.agents.models import infer_model
from pai_agent_sdk.context import (
    AgentContext,
    AgentInfo,
    ModelConfig,
    ResumableState,
    StreamEvent,
    SubagentStreamEvent,
    ToolConfig,
)
from pai_agent_sdk.environment.base import Environment
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.filters.environment_instructions import create_environment_instructions_filter
from pai_agent_sdk.filters.system_prompt import create_system_prompt_filter
from pai_agent_sdk.toolsets.core.base import BaseTool, GlobalHooks, Toolset
from pai_agent_sdk.utils import add_toolset_instructions

if TYPE_CHECKING:
    from pydantic_ai import ModelSettings
    from pydantic_ai.toolsets import AbstractToolset

    from pai_agent_sdk.subagents import SubagentConfig

# =============================================================================
# Type Variables
# =============================================================================

AgentDepsT = TypeVar("AgentDepsT", bound=AgentContext, default=AgentContext)
OutputT = TypeVar("OutputT", default=str)


# =============================================================================
# Agent Runtime
# =============================================================================


@dataclass
class AgentRuntime(Generic[AgentDepsT, OutputT]):
    """Container for agent runtime components.

    This dataclass holds all the components needed to run an agent,
    providing a clean interface for accessing the environment, context,
    and agent instance.

    Attributes:
        env: The environment instance managing resources.
        ctx: The agent context for session state.
        agent: The configured pydantic-ai Agent instance.

    Example:
        async with create_agent("openai:gpt-4") as runtime:
            result = await runtime.agent.run("Hello", deps=runtime.ctx)
            print(result.output)
    """

    env: Environment
    ctx: AgentDepsT
    agent: Agent[AgentDepsT, OutputT]
    core_toolset: Toolset[AgentDepsT] | None


# =============================================================================
# System Prompt Loading
# =============================================================================


def _load_system_prompt(template_vars: dict[str, Any] | None = None) -> str:
    """Load and render system prompt from the prompts directory.

    Args:
        template_vars: Variables to pass to Jinja2 template.

    Returns:
        Rendered system prompt string, or empty string if file not found.
    """
    prompt_path = Path(__file__).parent / "prompts" / "main.md"
    if not prompt_path.exists():
        return ""

    template_content = prompt_path.read_text()
    if not template_content.strip():
        return ""

    # Render with Jinja2
    env = jinja2.Environment(autoescape=False)  # noqa: S701
    template = env.from_string(template_content)
    return template.render(**(template_vars or {}))


# =============================================================================
# Agent Factory
# =============================================================================


@asynccontextmanager
async def create_agent(
    model: str | Model,
    *,
    # --- Model Configuration ---
    model_settings: ModelSettings | None = None,
    output_type: type[OutputT] = str,  # type: ignore[assignment]
    # --- Environment ---
    env: Environment | type[Environment] = LocalEnvironment,
    env_kwargs: dict[str, Any] | None = None,
    # --- Context ---
    context_type: type[AgentDepsT] = AgentContext,  # type: ignore[assignment]
    model_cfg: ModelConfig | None = None,
    tool_config: ToolConfig | None = None,
    extra_context_kwargs: dict[str, Any] | None = None,
    state: ResumableState | None = None,
    # --- Toolset ---
    tools: Sequence[type[BaseTool]] | None = None,
    toolsets: Sequence[AbstractToolset[Any]] | None = None,
    pre_hooks: dict[str, Any] | None = None,
    post_hooks: dict[str, Any] | None = None,
    global_hooks: GlobalHooks | None = None,
    toolset_max_retries: int = 3,
    toolset_timeout: float | None = None,
    skip_unavailable_tools: bool = True,
    # --- Compact Filter ---
    compact_model: str | Model | None = None,
    compact_model_settings: ModelSettings | None = None,
    compact_model_cfg: ModelConfig | None = None,
    # --- Subagent ---
    subagent_configs: Sequence[SubagentConfig] | None = None,
    include_builtin_subagents: bool = False,
    # --- Agent ---
    agent_tools: Sequence[Any] | None = None,
    system_prompt: str | None = None,
    system_prompt_template_vars: dict[str, Any] | None = None,
    history_processors: Sequence[HistoryProcessor[AgentDepsT]] | None = None,
    retries: int = 1,
    output_retries: int = 3,
    defer_model_check: bool = False,
    end_strategy: str = "exhaustive",
) -> AsyncIterator[AgentRuntime[AgentDepsT, OutputT]]:
    """Create and configure an agent with managed lifecycle.

    This context manager handles the full lifecycle of Environment, AgentContext,
    and Agent creation. It yields an AgentRuntime containing all three components.

    Args:
        model: Model string (e.g., "openai:gpt-4") or Model instance.

        model_settings: Optional model settings for inference configuration.
        output_type: Expected output type for the agent. Defaults to str.

        env: Environment instance or class. Defaults to LocalEnvironment.
        env_kwargs: Keyword arguments for Environment instantiation.

        context_type: AgentContext subclass to use. Defaults to AgentContext.
        model_cfg: ModelConfig for context window and capability settings.
        tool_config: ToolConfig for API keys and tool-specific settings.
        extra_context_kwargs: Additional kwargs passed to context_type constructor.
        state: ResumableState to restore session from. Defaults to None.

        tools: Sequence of BaseTool classes to include in the toolset.
        toolsets: Additional AbstractToolset instances to include.

        subagent_configs: Sequence of SubagentConfig for custom subagents.
        include_builtin_subagents: If True, include builtin subagents from presets/.
        pre_hooks: Dict mapping tool names to pre-hook functions.
        post_hooks: Dict mapping tool names to post-hook functions.
        global_hooks: GlobalHooks instance for all tools.
        toolset_max_retries: Max retries for tool execution. Defaults to 3.
        toolset_timeout: Default timeout for tool execution.
        skip_unavailable_tools: Skip tools where is_available() returns False.

        compact_model: Model for compact filter. Falls back to AgentSettings.
        compact_model_settings: Model settings for compact filter.
        compact_model_cfg: ModelConfig for compact filter. Defaults to main model_cfg.

        agent_tools: Additional tools to pass directly to Agent (pydantic-ai Tool objects).
        system_prompt: Custom system prompt(s). If None, loads from main.md.
        system_prompt_template_vars: Variables for Jinja2 template rendering.
        history_processors: Sequence of history processor functions.
        retries: Number of retries for agent run. Defaults to 1.
        output_retries: Number of retries for output parsing. Defaults to 3.
        defer_model_check: Defer model validation. Defaults to False.
        end_strategy: Strategy for ending agent run. Defaults to "exhaustive".

    Yields:
        AgentRuntime containing env, ctx, and agent.

    Example:
        Basic usage::

            async with create_agent("openai:gpt-4") as runtime:
                result = await runtime.agent.run("Hello", deps=runtime.ctx)
                print(result.output)

        With custom tools and configuration::

            async with create_agent(
                "anthropic:claude-3-5-sonnet",
                tools=[ReadFileTool, WriteFileTool],
                model_cfg=ModelConfig(context_window=200000),
                global_hooks=GlobalHooks(pre=my_pre_hook),
            ) as runtime:
                result = await runtime.agent.run("Read config.json", deps=runtime.ctx)

        With custom environment::

            async with create_agent(
                "openai:gpt-4",
                env=DockerEnvironment,
                env_kwargs={"image": "python:3.11"},
            ) as runtime:
                result = await runtime.agent.run("Run tests", deps=runtime.ctx)
    """
    async with AsyncExitStack() as stack:
        # --- Environment Setup ---
        if isinstance(env, Environment):
            entered_env = env
            # If already an instance, enter it
            await stack.enter_async_context(env)
        else:
            # Create and enter new environment instance
            entered_env = await stack.enter_async_context(env(**(env_kwargs or {})))

        # --- Build Configs ---
        effective_model_cfg = model_cfg or ModelConfig()
        effective_tool_config = tool_config or ToolConfig()

        # --- Context Setup ---
        ctx = await stack.enter_async_context(
            context_type(
                file_operator=entered_env.file_operator,
                shell=entered_env.shell,
                resources=entered_env.resources,
                model_cfg=effective_model_cfg,
                tool_config=effective_tool_config,
                **(extra_context_kwargs or {}),
            ).with_state(state)
        )

        # --- Toolset Setup ---
        all_toolsets: list[AbstractToolset[Any]] = []
        core_toolset: Toolset[AgentDepsT] | None = None

        # Create Toolset from BaseTool classes if provided
        if tools:
            core_toolset = Toolset(
                ctx,
                tools=tools,
                pre_hooks=pre_hooks,
                post_hooks=post_hooks,
                global_hooks=global_hooks,
                max_retries=toolset_max_retries,
                timeout=toolset_timeout,
                skip_unavailable=skip_unavailable_tools,
            )

            # Add subagent tools if requested
            if subagent_configs or include_builtin_subagents:
                from pai_agent_sdk.subagents import get_builtin_subagent_configs

                all_subagent_configs = list(subagent_configs) if subagent_configs else []
                if include_builtin_subagents:
                    all_subagent_configs.extend(get_builtin_subagent_configs().values())

                if all_subagent_configs:
                    core_toolset = core_toolset.with_subagents(
                        all_subagent_configs,
                        model=model,
                        model_settings=model_settings,
                    )

            all_toolsets.append(core_toolset)

        # Add user-provided toolsets
        if toolsets:
            all_toolsets.extend(toolsets)

        # Add environment toolsets
        all_toolsets.extend(entered_env.toolsets)

        # --- System Prompt ---
        effective_system_prompt: str | Sequence[str]
        if system_prompt is not None:
            effective_system_prompt = system_prompt
        else:
            # Load from template
            loaded_prompt = _load_system_prompt(system_prompt_template_vars)
            effective_system_prompt = loaded_prompt if loaded_prompt else ""

        # --- History Processors ---
        # Combine context's processors with built-in and user-provided ones
        all_processors: list[HistoryProcessor[AgentDepsT]] = [
            *ctx.get_history_processors(),
            create_compact_filter(
                model=compact_model,
                model_settings=compact_model_settings,
                model_cfg=compact_model_cfg or effective_model_cfg,
            ),
            create_environment_instructions_filter(entered_env),
            create_system_prompt_filter(system_prompt=effective_system_prompt),
        ]
        if history_processors:
            all_processors.extend(history_processors)

        # --- Create Agent ---
        agent: Agent[AgentDepsT, OutputT] = add_toolset_instructions(
            Agent(
                model=infer_model(model) if isinstance(model, str) else model,
                system_prompt=effective_system_prompt,
                model_settings=model_settings,
                deps_type=context_type,
                output_type=output_type,
                tools=agent_tools or (),
                toolsets=all_toolsets if all_toolsets else None,
                history_processors=all_processors if all_processors else None,
                retries=retries,
                output_retries=output_retries,
                defer_model_check=defer_model_check,
                end_strategy=end_strategy,  # type: ignore[arg-type]
            ),
            all_toolsets,
        )

        yield AgentRuntime(env=entered_env, ctx=ctx, agent=agent, core_toolset=core_toolset)


# =============================================================================
# Stream Hook Types
# =============================================================================


@dataclass
class NodeHookContext(Generic[AgentDepsT, OutputT]):
    """Context passed to node-level hooks (pre/post node.stream).

    Attributes:
        agent_info: Metadata about the current agent.
        node: The current graph node (ModelRequestNode or CallToolsNode).
        run: The AgentRun instance from agent.iter().
    """

    agent_info: AgentInfo
    node: ModelRequestNode[AgentDepsT, OutputT] | CallToolsNode[AgentDepsT, OutputT]
    run: AgentRun[AgentDepsT, OutputT]


@dataclass
class EventHookContext(Generic[AgentDepsT, OutputT]):
    """Context passed to event-level hooks (pre/post each event yield).

    Attributes:
        agent_info: Metadata about the current agent.
        event: The stream event being yielded.
        node: The current graph node.
        run: The AgentRun instance from agent.iter().
    """

    agent_info: AgentInfo
    event: SubagentStreamEvent
    node: ModelRequestNode[AgentDepsT, OutputT] | CallToolsNode[AgentDepsT, OutputT]
    run: AgentRun[AgentDepsT, OutputT]


# Hook type aliases
NodeHook = Callable[[NodeHookContext[AgentDepsT, OutputT]], Awaitable[None]]
EventHook = Callable[[EventHookContext[AgentDepsT, OutputT]], Awaitable[None]]


# =============================================================================
# Agent Streamer
# =============================================================================


@dataclass
class AgentStreamer(Generic[AgentDepsT, OutputT]):
    """Async iterator for streaming agent events with interrupt capability.

    This class wraps the merged event stream and provides control methods
    for interrupting the stream.

    Example::

        async with stream_agent(agent, "Hello", ctx=ctx) as streamer:
            async for event in streamer:
                print(f"[{event.agent_name}] {event.event}")
                if should_stop:
                    streamer.interrupt()
                    break
    """

    _event_generator: AsyncIterator[StreamEvent]
    _cancel_event: asyncio.Event
    _tasks: list[asyncio.Task[None]] = field(default_factory=list)

    def interrupt(self) -> None:
        """Interrupt the stream, causing iteration to stop."""
        self._cancel_event.set()

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self._event_generator

    async def __anext__(self) -> StreamEvent:
        return await self._event_generator.__anext__()


# =============================================================================
# Stream Agent
# =============================================================================


@asynccontextmanager
async def stream_agent(  # noqa: C901
    agent: Agent[AgentDepsT, OutputT],
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    ctx: AgentDepsT,
    message_history: Sequence[ModelMessage] | None = None,
    # Hooks
    pre_node_hook: NodeHook[AgentDepsT, OutputT] | None = None,
    post_node_hook: NodeHook[AgentDepsT, OutputT] | None = None,
    pre_event_hook: EventHook[AgentDepsT, OutputT] | None = None,
    post_event_hook: EventHook[AgentDepsT, OutputT] | None = None,
) -> AsyncIterator[AgentStreamer[AgentDepsT, OutputT]]:
    """Stream agent execution with subagent event aggregation.

    This context manager runs the agent and yields a streamer that merges
    events from the main agent and all subagents into a single stream.

    Args:
        agent: The pydantic-ai Agent to run.
        user_prompt: The prompt to send to the agent. Can be string or
            sequence of UserContent for multimodal input.
        ctx: The AgentContext for this run.
        message_history: Optional conversation history.
        pre_node_hook: Called before node.stream() starts.
        post_node_hook: Called after node.stream() completes.
        pre_event_hook: Called before each event is yielded.
        post_event_hook: Called after each event is yielded.

    Yields:
        AgentStreamer that can be iterated for StreamEvent objects.
        Each event contains agent_id, agent_name, and the raw event.

    Example::

        async with create_agent("openai:gpt-4") as runtime:
            async with stream_agent(
                runtime.agent,
                "Search for Python tutorials",
                ctx=runtime.ctx,
            ) as streamer:
                async for event in streamer:
                    if event.agent_name == "main":
                        # Handle main agent events
                        pass
                    else:
                        # Handle subagent events
                        pass
    """
    output_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
    cancel_event = asyncio.Event()
    main_done = asyncio.Event()
    poll_done = asyncio.Event()

    # Register main agent
    main_agent_info = AgentInfo(agent_id="main", agent_name="main")
    ctx.agent_registry["main"] = main_agent_info

    async def process_node(
        node: ModelRequestNode[AgentDepsT, OutputT] | CallToolsNode[AgentDepsT, OutputT],
        run: AgentRun[AgentDepsT, OutputT],
    ) -> bool:
        """Process a single node with hooks. Returns False if cancelled."""
        # PRE NODE HOOK
        if pre_node_hook:
            await pre_node_hook(NodeHookContext(agent_info=main_agent_info, node=node, run=run))

        async with node.stream(run.ctx) as request_stream:
            async for event in request_stream:
                if cancel_event.is_set():
                    return False

                # PRE EVENT HOOK
                if pre_event_hook:
                    await pre_event_hook(EventHookContext(agent_info=main_agent_info, event=event, node=node, run=run))

                await output_queue.put(StreamEvent(agent_id="main", agent_name="main", event=event))

                # POST EVENT HOOK
                if post_event_hook:
                    await post_event_hook(EventHookContext(agent_info=main_agent_info, event=event, node=node, run=run))

        # POST NODE HOOK
        if post_node_hook:
            await post_node_hook(NodeHookContext(agent_info=main_agent_info, node=node, run=run))
        return True

    async def run_main() -> None:
        """Run the main agent and push events to output_queue."""
        try:
            async with agent.iter(user_prompt, deps=ctx, message_history=message_history) as run:
                async for node in run:
                    if cancel_event.is_set():
                        return

                    if Agent.is_user_prompt_node(node) or Agent.is_end_node(node):
                        continue

                    if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):  # noqa: SIM102
                        if not await process_node(node, run):
                            return
        finally:
            main_done.set()

    async def poll_subagents() -> None:
        """Poll subagent stream queues and push events to output_queue."""
        try:
            while True:
                # Check exit condition: main done and all queues empty
                if main_done.is_set():
                    all_empty = all(q.empty() for q in ctx.subagent_stream_queues.values())
                    if all_empty:
                        return

                if cancel_event.is_set():
                    return

                # Collect events from all subagent queues
                for agent_id, queue in list(ctx.subagent_stream_queues.items()):
                    try:
                        event = queue.get_nowait()
                        agent_info = ctx.agent_registry.get(agent_id)
                        await output_queue.put(
                            StreamEvent(
                                agent_id=agent_id,
                                agent_name=agent_info.agent_name if agent_info else "unknown",
                                event=event,
                            )
                        )
                    except asyncio.QueueEmpty:
                        pass

                await asyncio.sleep(0.001)  # Yield control to avoid busy loop
        finally:
            poll_done.set()

    async def generate_events() -> AsyncIterator[StreamEvent]:
        """Consume from output_queue and yield events."""
        while True:
            # Check exit condition: poll done and output queue empty
            if poll_done.is_set() and output_queue.empty():
                return

            if cancel_event.is_set():
                return

            try:
                event = await asyncio.wait_for(output_queue.get(), timeout=0.1)
                yield event
            except TimeoutError:
                continue

    # Start producer tasks
    main_task = asyncio.create_task(run_main())
    poll_task = asyncio.create_task(poll_subagents())

    streamer: AgentStreamer[AgentDepsT, OutputT] = AgentStreamer(
        _event_generator=generate_events(),
        _cancel_event=cancel_event,
        _tasks=[main_task, poll_task],
    )

    try:
        yield streamer
    finally:
        cancel_event.set()
        # Wait for tasks to complete
        await asyncio.gather(main_task, poll_task, return_exceptions=True)
