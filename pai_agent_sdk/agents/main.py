"""Main agent factory for creating configured agents.

This module provides the create_agent function for building agents
with proper environment and context lifecycle management.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, cast

import jinja2
from agent_environment import Environment
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, UsageLimits
from pydantic_ai._agent_graph import CallToolsNode, HistoryProcessor, ModelRequestNode
from pydantic_ai.messages import ModelMessage, UserContent
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.output import OutputSpec
from pydantic_ai.run import AgentRun
from typing_extensions import TypeVar

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.agents.compact import create_compact_filter
from pai_agent_sdk.agents.models import infer_model
from pai_agent_sdk.context import (
    AgentContext,
    AgentInfo,
    AgentStreamEvent,
    ModelConfig,
    ResumableState,
    RunContextMetadata,
    StreamEvent,
    ToolConfig,
)
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.filters.environment_instructions import create_environment_instructions_filter
from pai_agent_sdk.filters.system_prompt import create_system_prompt_filter
from pai_agent_sdk.toolsets.core.base import BaseTool, GlobalHooks, Toolset
from pai_agent_sdk.utils import AgentDepsT, add_toolset_instructions

if TYPE_CHECKING:
    from pydantic_ai import ModelSettings
    from pydantic_ai.toolsets import AbstractToolset

    from pai_agent_sdk.subagents import SubagentConfig

logger = get_logger(__name__)

# =============================================================================
# Exceptions
# =============================================================================


class AgentInterrupted(Exception):
    """Raised when agent execution is interrupted by user.

    This exception is raised when `AgentStreamer.interrupt()` is called,
    providing immediate cancellation of all running tasks.
    """

    pass


# =============================================================================
# Type Variables
# =============================================================================

OutputT = TypeVar("OutputT")


# =============================================================================
# Agent Runtime
# =============================================================================


@dataclass
class AgentRuntime(Generic[AgentDepsT, OutputT]):
    """Container for agent runtime components with lifecycle management.

    This dataclass holds all the components needed to run an agent,
    providing a clean interface for accessing the environment, context,
    and agent instance. It also acts as an async context manager to
    manage the lifecycle of env, ctx, and agent.

    Attributes:
        env: The environment instance managing resources.
        ctx: The agent context for session state.
        agent: The configured pydantic-ai Agent instance.
        core_toolset: The core toolset with BaseTool instances.

    Example:
        runtime = create_agent("openai:gpt-4")
        async with runtime:
            result = await runtime.agent.run("Hello", deps=runtime.ctx)
            print(result.output)

        # Or with external env management:
        async with env:
            runtime = create_agent("openai:gpt-4", env=env)
            async with runtime:  # Only enters ctx and agent
                result = await runtime.agent.run("Hello", deps=runtime.ctx)
    """

    env: Environment
    ctx: AgentDepsT
    agent: Agent[AgentDepsT, OutputT]
    core_toolset: Toolset[AgentDepsT] | None
    _exit_stack: AsyncExitStack | None = field(default=None, repr=False)

    async def __aenter__(self) -> AgentRuntime[AgentDepsT, OutputT]:
        """Enter the runtime, managing env/ctx/agent lifecycles.

        Only enters components that are not already entered (checked via _entered flag).
        Uses AsyncExitStack to track what was entered, ensuring proper cleanup.
        """
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        # Enter in order: env -> ctx -> agent
        # Only enter if not already entered
        if not self.env._entered:
            await self._exit_stack.enter_async_context(self.env)
        if not self.ctx._entered:
            await self._exit_stack.enter_async_context(self.ctx)
        # Agent uses reference counting, safe to enter multiple times
        await self._exit_stack.enter_async_context(self.agent)

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool | None:
        """Exit the runtime, cleaning up only what we entered.

        AsyncExitStack automatically exits in reverse order, and only
        exits contexts that were entered via enter_async_context().
        """
        if self._exit_stack:
            result = await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
            self._exit_stack = None
            return result
        return None


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


def create_agent(
    model: Model | KnownModelName | str | None,
    *,
    # --- Model Configuration ---
    model_settings: ModelSettings | None = None,
    output_type: OutputSpec[OutputT] = str,  # type: ignore[assignment]
    # --- Environment ---
    env: Environment | type[Environment] = LocalEnvironment,
    env_kwargs: dict[str, Any] | None = None,
    # --- Context ---
    context_type: type[AgentDepsT] = AgentContext,  # type: ignore[assignment]
    model_cfg: ModelConfig | None = None,
    tool_config: ToolConfig | None = None,
    extra_context_kwargs: dict[str, Any] | None = None,
    state: ResumableState | None = None,
    need_user_approve_tools: Sequence[str] | None = None,
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
    agent_name: str = "main",
    system_prompt: str | None = None,
    system_prompt_template_vars: dict[str, Any] | None = None,
    history_processors: Sequence[HistoryProcessor[AgentDepsT]] | None = None,
    retries: int = 1,
    output_retries: int = 3,
    defer_model_check: bool = False,
    end_strategy: str = "exhaustive",
    metadata: RunContextMetadata | None = None,
) -> AgentRuntime[AgentDepsT, OutputT]:
    """Create and configure an agent runtime.

    This function creates an AgentRuntime containing Environment, AgentContext,
    and Agent. The runtime should be used as an async context manager to manage
    the lifecycle of these components.

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
        need_user_approve_tools: Tools requiring user approval before execution.

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
        agent_name: Name of the agent for logging.
        system_prompt: Custom system prompt(s). If None, loads from main.md.
        system_prompt_template_vars: Variables for Jinja2 template rendering.
        history_processors: Sequence of history processor functions.
        retries: Number of retries for agent run. Defaults to 1.
        output_retries: Number of retries for output parsing. Defaults to 3.
        defer_model_check: Defer model validation. Defaults to False.
        end_strategy: Strategy for ending agent run. Defaults to "exhaustive".
        metadata: Optional RunContextMetadata for context management.

    Returns:
        AgentRuntime containing env, ctx, and agent. Use as async context manager.

    Example:
        Basic usage::

            runtime = create_agent("openai:gpt-4")
            async with runtime:
                result = await runtime.agent.run("Hello", deps=runtime.ctx)
                print(result.output)

        With custom tools and configuration::

            runtime = create_agent(
                "anthropic:claude-3-5-sonnet",
                tools=[ReadFileTool, WriteFileTool],
                model_cfg=ModelConfig(context_window=200000),
                global_hooks=GlobalHooks(pre=my_pre_hook),
            )
            async with runtime:
                result = await runtime.agent.run("Read config.json", deps=runtime.ctx)

        With external environment management::

            async with DockerEnvironment(image="python:3.11") as docker_env:
                runtime = create_agent("openai:gpt-4", env=docker_env)
                async with runtime:  # Only enters ctx and agent
                    result = await runtime.agent.run("Run tests", deps=runtime.ctx)
    """
    # --- Environment Setup ---
    actual_env = env if isinstance(env, Environment) else env(**(env_kwargs or {}))
    logger.debug("Environment created: %s", type(actual_env).__name__)

    # --- Build Configs ---
    effective_model_cfg = model_cfg or ModelConfig()
    effective_tool_config = tool_config or ToolConfig()

    # --- Context Setup ---
    ctx = context_type(
        env=actual_env,
        model_cfg=effective_model_cfg,
        tool_config=effective_tool_config,
        need_user_approve_tools=list(need_user_approve_tools) if need_user_approve_tools else [],
        **(extra_context_kwargs or {}),
    ).with_state(state)
    logger.debug("Context created: %s (run_id=%s)", type(ctx).__name__, ctx.run_id)

    # --- History Processors ---
    # Combine context's processors with built-in and user-provided ones
    all_processors: list[HistoryProcessor[AgentDepsT]] = [
        *ctx.get_history_processors(),
        create_compact_filter(
            model=compact_model,
            model_settings=compact_model_settings,
            model_cfg=compact_model_cfg or effective_model_cfg,
            main_model=model,
            main_model_settings=model_settings,
        ),
        create_environment_instructions_filter(actual_env),
    ]
    if history_processors:
        all_processors.extend(history_processors)

    # --- Toolset Setup ---
    all_toolsets: list[AbstractToolset[Any]] = []
    core_toolset: Toolset[AgentDepsT] | None = None

    # Create Toolset from BaseTool classes if provided
    tools = tools or []
    logger.debug("Creating core toolset with %d tools", len(tools))
    core_toolset = Toolset(
        tools=tools,
        pre_hooks=pre_hooks,
        post_hooks=post_hooks,
        global_hooks=global_hooks,
        max_retries=toolset_max_retries,
        timeout=toolset_timeout,
        skip_unavailable=skip_unavailable_tools,
        toolset_id="core",
    )

    # Add subagent tools if requested
    if subagent_configs or include_builtin_subagents:
        from pai_agent_sdk.subagents import get_builtin_subagent_configs

        all_subagent_configs = list(subagent_configs) if subagent_configs else []
        if include_builtin_subagents:
            all_subagent_configs.extend(get_builtin_subagent_configs().values())

        if all_subagent_configs:
            logger.debug("Adding %d subagent configs to toolset", len(all_subagent_configs))
            core_toolset = core_toolset.with_subagents(
                all_subagent_configs,
                model=model,
                model_settings=model_settings,
                history_processors=[*ctx.get_history_processors()],
            )

    all_toolsets.append(core_toolset)

    # Add user-provided toolsets
    if toolsets:
        all_toolsets.extend(toolsets)

    # Add environment toolsets (will be available after env enters)
    all_toolsets.extend(actual_env.toolsets)

    # --- System Prompt ---
    effective_system_prompt: str | Sequence[str]
    if system_prompt is not None:
        effective_system_prompt = system_prompt
    else:
        # Load from template
        loaded_prompt = _load_system_prompt(system_prompt_template_vars)
        effective_system_prompt = loaded_prompt if loaded_prompt else ""

    # --- Create Agent ---
    logger.debug("Creating agent with model=%s, output_type=%s", model, output_type)
    agent: Agent[AgentDepsT, OutputT] = add_toolset_instructions(
        Agent(
            model=infer_model(model) if isinstance(model, str) else model,
            system_prompt=effective_system_prompt,
            model_settings=model_settings,
            deps_type=context_type,
            output_type=output_type,
            tools=agent_tools or (),
            toolsets=all_toolsets if all_toolsets else None,
            history_processors=[
                *(all_processors or []),
                create_system_prompt_filter(system_prompt=effective_system_prompt),
            ],
            retries=retries,
            output_retries=output_retries,
            defer_model_check=defer_model_check,
            end_strategy=end_strategy,  # type: ignore[arg-type]
            metadata=cast(dict[str, Any], metadata) if metadata else None,
            name=agent_name,
        ),
        all_toolsets,
    )

    logger.debug(
        "Agent created: toolsets=%d, history_processors=%d",
        len(all_toolsets) if all_toolsets else 0,
        len(all_processors) if all_processors else 0,
    )
    return AgentRuntime(env=actual_env, ctx=ctx, agent=agent, core_toolset=core_toolset)


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
        output_queue: Queue for emitting custom StreamEvent to the output stream.
    """

    agent_info: AgentInfo
    node: ModelRequestNode[AgentDepsT, OutputT] | CallToolsNode[AgentDepsT, OutputT]
    run: AgentRun[AgentDepsT, OutputT]
    output_queue: asyncio.Queue[StreamEvent]


@dataclass
class EventHookContext(Generic[AgentDepsT, OutputT]):
    """Context passed to event-level hooks (pre/post each event yield).

    Attributes:
        agent_info: Metadata about the current agent.
        event: The stream event being yielded.
        node: The current graph node.
        run: The AgentRun instance from agent.iter().
        output_queue: Queue for emitting custom StreamEvent to the output stream.
    """

    agent_info: AgentInfo
    event: AgentStreamEvent
    node: ModelRequestNode[AgentDepsT, OutputT] | CallToolsNode[AgentDepsT, OutputT]
    run: AgentRun[AgentDepsT, OutputT]
    output_queue: asyncio.Queue[StreamEvent]


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

    Attributes:
        run: The AgentRun instance. None until agent.iter() starts, available during
            and after streaming. Use to access messages, usage, and result.
        exception: The exception captured during streaming, if any. Available after
            streaming completes. Check this or call raise_if_exception() after iteration.

    Example::

        async with stream_agent(agent, "Hello", ctx=ctx) as streamer:
            async for event in streamer:
                print(f"[{event.agent_name}] {event.event}")
                if streamer.run:
                    print(f"Messages so far: {len(streamer.run.all_messages())}")
                if should_stop:
                    streamer.interrupt()
                    break
            # After streaming, AgentInterrupted is raised automatically
            # Access final result and usage
            if streamer.run:
                print(f"Usage: {streamer.run.usage()}")
    """

    _event_generator: AsyncIterator[StreamEvent]
    _tasks: list[asyncio.Task[None]] = field(default_factory=list)
    run: AgentRun[AgentDepsT, OutputT] | None = None
    exception: BaseException | None = None
    _interrupted: bool = False

    def interrupt(self) -> None:
        """Interrupt the stream immediately, cancelling all running tasks.

        This method provides hard cancellation - all running tasks are cancelled
        immediately via asyncio.Task.cancel(). When the context manager exits,
        AgentInterrupted will be raised.
        """
        self._interrupted = True
        for task in self._tasks:
            if not task.done():
                task.cancel()

    def raise_if_exception(self) -> None:
        """Raise the captured exception if any occurred during streaming.

        Call this after iteration completes to propagate any errors from
        the main agent or subagent tasks.

        Raises:
            AgentInterrupted: If interrupt() was called.
            BaseException: Any other exception that occurred during streaming.
        """
        # Check stored exception first
        if self.exception is not None:
            raise self.exception

        # Also check tasks for exceptions (in case called before context manager exits)
        for task in self._tasks:
            if task.done() and not task.cancelled():
                exc = task.exception()
                if exc is not None:
                    raise exc

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self._event_generator

    async def __anext__(self) -> StreamEvent:
        return await self._event_generator.__anext__()


# =============================================================================
# Stream Agent
# =============================================================================


@asynccontextmanager
async def stream_agent(  # noqa: C901
    runtime: AgentRuntime[AgentDepsT, OutputT],
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    usage_limits: UsageLimits | None = None,
    # Hooks
    pre_node_hook: NodeHook[AgentDepsT, OutputT] | None = None,
    post_node_hook: NodeHook[AgentDepsT, OutputT] | None = None,
    pre_event_hook: EventHook[AgentDepsT, OutputT] | None = None,
    post_event_hook: EventHook[AgentDepsT, OutputT] | None = None,
    metadata: RunContextMetadata | None = None,
    # Error handling
    raise_on_error: bool = True,
) -> AsyncIterator[AgentStreamer[AgentDepsT, OutputT]]:
    """Stream agent execution with subagent event aggregation.

    This context manager runs the agent and yields a streamer that merges
    events from the main agent and all subagents into a single stream.

    Lifecycle Management:
        This function automatically manages the runtime lifecycle internally.
        When called, it will:
        1. Enter the runtime (env -> ctx -> agent) if not already entered
        2. Execute the agent with the given prompt
        3. Exit the runtime when streaming completes

        The runtime uses `_entered` flags to avoid double-entering components
        that are already active. This means you can safely call stream_agent
        without manually entering the runtime first.

        Note: Manual runtime lifecycle management is not recommended.
        Let stream_agent handle it automatically for proper resource cleanup.

    Args:
        runtime: The AgentRuntime containing agent and context.
        user_prompt: The prompt to send to the agent. Can be string or
            sequence of UserContent for multimodal input.
        message_history: Optional conversation history.
        deferred_tool_results: Results from deferred tool calls.
        pre_node_hook: Called before node.stream() starts.
        post_node_hook: Called after node.stream() completes.
        pre_event_hook: Called before each event is yielded.
        post_event_hook: Called after each event is yielded.
        metadata: Optional RunContextMetadata for context management.
        raise_on_error: If True (default), exceptions during streaming are re-raised
            immediately. If False, exceptions are captured in streamer.exception
            and can be checked after iteration via raise_if_exception().

    Yields:
        AgentStreamer that can be iterated for StreamEvent objects.
        Each event contains agent_id, agent_name, and the raw event.

    Example::

        # Recommended: Let stream_agent manage the runtime lifecycle
        runtime = create_agent("openai:gpt-4")
        async with stream_agent(
            runtime,
            "Search for Python tutorials",
        ) as streamer:
            async for event in streamer:
                if event.agent_name == "main":
                    # Handle main agent events
                    pass
                else:
                    # Handle subagent events
                    pass
    """
    # Extract agent and ctx from runtime
    agent = runtime.agent
    ctx = runtime.ctx

    # Enable streaming for emit_event
    ctx._stream_queue_enabled = True

    output_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
    main_done = asyncio.Event()
    poll_done = asyncio.Event()

    logger.debug(
        "Starting stream_agent with user_prompt=%s",
        user_prompt[:100] if isinstance(user_prompt, str) else type(user_prompt),
    )

    # Build main agent info
    main_agent_info = AgentInfo(agent_id="main", agent_name=agent.name or "main")
    ctx.user_prompts = user_prompt

    async def process_node(
        node: ModelRequestNode[AgentDepsT, OutputT] | CallToolsNode[AgentDepsT, OutputT],
        run: AgentRun[AgentDepsT, OutputT],
    ) -> None:
        """Process a single node with hooks."""
        # PRE NODE HOOK
        logger.debug("Processing node: %s", type(node).__name__)
        if pre_node_hook:
            await pre_node_hook(
                NodeHookContext(agent_info=main_agent_info, node=node, run=run, output_queue=output_queue)
            )
        async with node.stream(run.ctx) as request_stream:
            async for event in request_stream:
                # PRE EVENT HOOK
                if pre_event_hook:
                    await pre_event_hook(
                        EventHookContext(
                            agent_info=main_agent_info, event=event, node=node, run=run, output_queue=output_queue
                        )
                    )

                await output_queue.put(
                    StreamEvent(
                        agent_id=main_agent_info.agent_id,
                        agent_name=main_agent_info.agent_name,
                        event=ctx.tool_id_wrapper.wrap_event(event),
                    )
                )

                # POST EVENT HOOK
                if post_event_hook:
                    await post_event_hook(
                        EventHookContext(
                            agent_info=main_agent_info, event=event, node=node, run=run, output_queue=output_queue
                        )
                    )

        # POST NODE HOOK
        logger.debug("Node completed: %s", type(node).__name__)
        if post_node_hook:
            await post_node_hook(
                NodeHookContext(agent_info=main_agent_info, node=node, run=run, output_queue=output_queue)
            )

    async def run_main() -> None:
        """Run the main agent and push events to output_queue."""
        logger.debug("Main agent task started")
        try:
            async with (
                runtime,
                agent.iter(
                    user_prompt,
                    deps=ctx,
                    usage_limits=usage_limits,
                    message_history=message_history,
                    deferred_tool_results=deferred_tool_results,
                    metadata=cast(dict[str, Any], metadata) if metadata else None,
                ) as run,
            ):
                streamer.run = run  # Expose run immediately
                async for node in run:
                    if Agent.is_user_prompt_node(node) or Agent.is_end_node(node):
                        continue

                    if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                        await process_node(node, run)
        except Exception:
            logger.exception("Error in main agent task")
            raise
        finally:
            logger.debug("Main agent task finished")
            main_done.set()

    async def poll_subagents() -> None:
        """Poll subagent stream queues and push events to output_queue."""
        logger.debug("Subagent polling task started")
        try:
            while True:
                # Check exit condition: main done and all queues empty
                if main_done.is_set():
                    all_empty = all(q.empty() for q in ctx.agent_stream_queues.values())
                    if all_empty:
                        return

                # Collect events from all subagent queues
                for agent_id, queue in list(ctx.agent_stream_queues.items()):
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
            logger.debug("Subagent polling task finished")
            poll_done.set()

    async def generate_events() -> AsyncIterator[StreamEvent]:
        """Consume from output_queue and yield events.

        Also monitors main_task for exceptions and propagates them immediately.
        """
        while True:
            # Check if main_task failed - propagate exception immediately
            if main_task.done() and not main_task.cancelled():
                exc = main_task.exception()
                if exc is not None:
                    raise exc

            # Check exit condition: poll done and output queue empty
            if poll_done.is_set() and output_queue.empty():
                # Final check for main_task exception before returning
                if main_task.done() and not main_task.cancelled():
                    exc = main_task.exception()
                    if exc is not None:
                        raise exc
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
        _tasks=[main_task, poll_task],
    )

    try:
        yield streamer
    except Exception as e:
        logger.exception("Uncaught exception in stream_agent context")
        streamer.exception = e
        if raise_on_error:
            raise  # Re-raise so caller can handle it
    else:
        if (run := streamer.run) and (result := run.result) and isinstance(result.output, DeferredToolRequests):
            result.output = ctx.tool_id_wrapper.wrap_deferred_tool_requests(result.output)
    finally:
        # Wait for tasks to complete and capture any exception
        results = await asyncio.gather(main_task, poll_task, return_exceptions=True)

        # Find first real exception (non-CancelledError)
        exceptions = [r for r in results if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError)]

        if streamer._interrupted:
            streamer.exception = AgentInterrupted("Agent execution was interrupted")
        elif exceptions:
            streamer.exception = exceptions[0]
