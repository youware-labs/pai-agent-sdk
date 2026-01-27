"""Main agent factory for creating configured agents.

This module provides the create_agent function for building agents
with proper environment and context lifecycle management.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, cast

import jinja2
from agent_environment import Environment
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, UsageLimits, UserError
from pydantic_ai._agent_graph import CallToolsNode, HistoryProcessor, ModelRequestNode
from pydantic_ai.messages import ModelMessage, UserContent
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.output import OutputSpec
from pydantic_ai.run import AgentRun
from typing_extensions import TypeVar

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.agents.compact import create_compact_filter
from pai_agent_sdk.agents.guards import attach_message_bus_guard
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
from pai_agent_sdk.events import (
    AgentExecutionCompleteEvent,
    AgentExecutionFailedEvent,
    AgentExecutionStartEvent,
    LifecycleEvent,
    ModelRequestCompleteEvent,
    ModelRequestStartEvent,
    ToolCallsCompleteEvent,
    ToolCallsStartEvent,
)
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
# Lifecycle Tracking
# =============================================================================


@dataclass
class LifecycleTracker:
    """Tracks lifecycle state during agent execution."""

    loop_index: int = 0


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


def _load_system_prompt(
    template: str | None = None,
    template_vars: dict[str, Any] | None = None,
) -> str:
    """Load and render system prompt.

    Args:
        template: Template string. If None, loads from prompts/main.md.
        template_vars: Variables to pass to Jinja2 template.

    Returns:
        Rendered system prompt string, or empty string if template is empty/not found.
    """
    if template is None:
        prompt_path = Path(__file__).parent / "prompts" / "main.md"
        if not prompt_path.exists():
            return ""
        template = prompt_path.read_text()

    if not template.strip():
        return ""

    # Always render with Jinja2 to support default values in templates
    env = jinja2.Environment(autoescape=False)  # noqa: S701
    jinja_template = env.from_string(template)
    return jinja_template.render(**(template_vars or {}))


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
    need_user_approve_mcps: Sequence[str] | None = None,
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
    unified_subagents: bool = False,
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
        need_user_approve_mcps: MCP servers requiring user approval for all tools.

        tools: Sequence of BaseTool classes to include in the toolset.
        toolsets: Additional AbstractToolset instances to include.

        subagent_configs: Sequence of SubagentConfig for custom subagents.
        include_builtin_subagents: If True, include builtin subagents from presets/.
        unified_subagents: If True, create a single 'delegate' tool that can call any
            subagent by name. If False (default), create separate tools for each subagent.
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
        system_prompt: Custom system prompt or Jinja2 template string. If None, loads from
            prompts/main.md. When used with system_prompt_template_vars, the string is
            rendered as a Jinja2 template, supporting conditionals and default values.
        system_prompt_template_vars: Variables for Jinja2 template rendering. Works with
            both custom system_prompt strings and the default template file.
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

        With templated system prompt::

            runtime = create_agent(
                "openai:gpt-4",
                system_prompt="You are a {{ role }}. {{ extra_instructions | default('') }}",
                system_prompt_template_vars={"role": "helpful assistant"},
            )
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
        need_user_approve_mcps=list(need_user_approve_mcps) if need_user_approve_mcps else [],
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
                model_cfg=effective_model_cfg,
                unified=unified_subagents,
            )

    all_toolsets.append(core_toolset)

    # Add user-provided toolsets
    if toolsets:
        all_toolsets.extend(toolsets)

    # Add environment toolsets (will be available after env enters)
    all_toolsets.extend(actual_env.toolsets)

    # --- System Prompt ---
    effective_system_prompt = _load_system_prompt(system_prompt, system_prompt_template_vars)

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

    # Attach message bus guard for pending message handling
    attach_message_bus_guard(agent)

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
class RuntimeReadyContext(Generic[AgentDepsT, OutputT]):
    """Context passed to runtime ready hook (after runtime enter, before agent.iter).

    This hook is called after the runtime (env, ctx, agent) has been entered but
    before agent.iter() starts. Use it to:
    - Initialize resources that depend on the environment being ready
    - Emit custom events to the output stream
    - Modify context state before agent execution
    - Modify user_prompt or deferred_tool_results to control agent input

    Attributes:
        runtime: The AgentRuntime containing env, ctx, and agent.
        agent_info: Metadata about the main agent.
        output_queue: Queue for emitting custom StreamEvent to the output stream.
        user_prompt: The user prompt to send to the agent. Can be modified by hook.
        deferred_tool_results: Results from deferred tool calls. Can be modified by hook.
    """

    runtime: AgentRuntime[AgentDepsT, OutputT]
    agent_info: AgentInfo
    output_queue: asyncio.Queue[StreamEvent]
    user_prompt: str | Sequence[UserContent] | None
    deferred_tool_results: DeferredToolResults | None


@dataclass
class AgentStartContext(Generic[AgentDepsT, OutputT]):
    """Context passed to agent start hook (after agent.iter starts, before first node).

    This hook is called after agent.iter() has started and the run object is available,
    but before any nodes are processed. Use it to:
    - Access the run object for initial state inspection
    - Log agent start with run metadata
    - Emit custom events at agent start

    Attributes:
        runtime: The AgentRuntime containing env, ctx, and agent.
        agent_info: Metadata about the main agent.
        output_queue: Queue for emitting custom StreamEvent to the output stream.
        run: The AgentRun instance from agent.iter().
    """

    runtime: AgentRuntime[AgentDepsT, OutputT]
    agent_info: AgentInfo
    output_queue: asyncio.Queue[StreamEvent]
    run: AgentRun[AgentDepsT, OutputT]


@dataclass
class AgentCompleteContext(Generic[AgentDepsT, OutputT]):
    """Context passed to agent complete hook (after all nodes processed, before agent.iter exits).

    This hook is called after all nodes have been processed but before the agent.iter()
    context manager exits. Use it to:
    - Access the final result and usage statistics
    - Log agent completion with full run data
    - Emit custom completion events

    Attributes:
        runtime: The AgentRuntime containing env, ctx, and agent.
        agent_info: Metadata about the main agent.
        output_queue: Queue for emitting custom StreamEvent to the output stream.
        run: The AgentRun instance with result available.
    """

    runtime: AgentRuntime[AgentDepsT, OutputT]
    agent_info: AgentInfo
    output_queue: asyncio.Queue[StreamEvent]
    run: AgentRun[AgentDepsT, OutputT]


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


# User prompt type alias
UserPromptT = str | Sequence[UserContent]

# Hook type aliases
RuntimeReadyHook = Callable[[RuntimeReadyContext[AgentDepsT, OutputT]], Awaitable[None]]
AgentStartHook = Callable[[AgentStartContext[AgentDepsT, OutputT]], Awaitable[None]]
AgentCompleteHook = Callable[[AgentCompleteContext[AgentDepsT, OutputT]], Awaitable[None]]
NodeHook = Callable[[NodeHookContext[AgentDepsT, OutputT]], Awaitable[None]]
EventHook = Callable[[EventHookContext[AgentDepsT, OutputT]], Awaitable[None]]
UserPromptFactory = Callable[[AgentRuntime[AgentDepsT, OutputT]], Awaitable[UserPromptT]]


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
    user_prompt: UserPromptT | None = None,
    *,
    user_prompt_factory: UserPromptFactory[AgentDepsT, OutputT] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    usage_limits: UsageLimits | None = None,
    # Hooks
    on_runtime_ready: RuntimeReadyHook[AgentDepsT, OutputT] | None = None,
    on_agent_start: AgentStartHook[AgentDepsT, OutputT] | None = None,
    on_agent_complete: AgentCompleteHook[AgentDepsT, OutputT] | None = None,
    pre_node_hook: NodeHook[AgentDepsT, OutputT] | None = None,
    post_node_hook: NodeHook[AgentDepsT, OutputT] | None = None,
    pre_event_hook: EventHook[AgentDepsT, OutputT] | None = None,
    post_event_hook: EventHook[AgentDepsT, OutputT] | None = None,
    metadata: RunContextMetadata | None = None,
    # Error handling
    raise_on_error: bool = True,
    # Lifecycle events
    emit_lifecycle_events: bool = True,
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
        user_prompt_factory: Async callable that receives AgentRuntime and returns
            a user prompt. Called after runtime enters, before on_runtime_ready.
            Use when prompt generation requires runtime resources (e.g., reading files).
            If both user_prompt and user_prompt_factory are provided, factory takes precedence.
        message_history: Optional conversation history.
        deferred_tool_results: Results from deferred tool calls.
        on_runtime_ready: Called after runtime enters but before agent.iter() starts.
            Use to initialize resources, emit events, or modify context state.
        on_agent_start: Called after agent.iter() starts, before first node.
            Use to access run object for initial state inspection.
        on_agent_complete: Called after all nodes processed, before agent.iter() exits.
            Use to access final result and usage statistics.
        pre_node_hook: Called before node.stream() starts.
        post_node_hook: Called after node.stream() completes.
        pre_event_hook: Called before each event is yielded.
        post_event_hook: Called after each event is yielded.
        metadata: Optional RunContextMetadata for context management.
        raise_on_error: If True (default), exceptions during streaming are re-raised
            immediately. If False, exceptions are captured in streamer.exception
            and can be checked after iteration via raise_if_exception().
        emit_lifecycle_events: If True (default), emit built-in lifecycle events
            (AgentExecutionStartEvent, LoopStartEvent, NodeStartEvent, etc.) to the
            stream. Set to False to disable these events for cleaner output or
            when implementing custom event handling via hooks.

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
    # Validate mutually exclusive parameters
    if user_prompt is not None and user_prompt_factory is not None:
        msg = "Cannot specify both 'user_prompt' and 'user_prompt_factory'. Use one or the other."
        raise UserError(msg)

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

    # Lifecycle tracker for loop counting.
    # loop_index is set at the start of each ModelRequest and used by both
    # ModelRequest and ToolCalls events within the same loop iteration.
    tracker = LifecycleTracker()

    async def emit_lifecycle_event(event: LifecycleEvent) -> None:
        """Emit a lifecycle event if enabled."""
        if emit_lifecycle_events:
            await output_queue.put(
                StreamEvent(
                    agent_id=main_agent_info.agent_id,
                    agent_name=main_agent_info.agent_name,
                    event=event,
                )
            )

    async def handle_model_request_node(
        node: ModelRequestNode[AgentDepsT, OutputT],
        run: AgentRun[AgentDepsT, OutputT],
        node_start_time: float,
    ) -> None:
        """Handle model_request node with lifecycle events.

        Each ModelRequestNode marks the start of a new loop iteration.
        The loop_index is incremented here before processing.
        """
        current_loop = tracker.loop_index
        tracker.loop_index += 1  # Increment for next loop

        await emit_lifecycle_event(
            ModelRequestStartEvent(event_id=ctx.run_id, loop_index=current_loop, message_count=len(run.all_messages()))
        )

        await process_node(node, run)

        await emit_lifecycle_event(
            ModelRequestCompleteEvent(
                event_id=ctx.run_id,
                loop_index=current_loop,
                duration_seconds=time.perf_counter() - node_start_time,
            )
        )

    async def handle_call_tools_node(
        node: CallToolsNode[AgentDepsT, OutputT],
        run: AgentRun[AgentDepsT, OutputT],
        node_start_time: float,
    ) -> None:
        """Handle call_tools node with lifecycle events.

        ToolCalls always follow a ModelRequest, so we use (loop_index - 1)
        to reference the loop that just completed its model request phase.
        """
        current_loop = tracker.loop_index - 1
        await emit_lifecycle_event(ToolCallsStartEvent(event_id=ctx.run_id, loop_index=current_loop))

        await process_node(node, run)

        await emit_lifecycle_event(
            ToolCallsCompleteEvent(
                event_id=ctx.run_id,
                loop_index=current_loop,
                duration_seconds=time.perf_counter() - node_start_time,
            )
        )

    async def process_all_nodes(run: AgentRun[AgentDepsT, OutputT]) -> None:
        """Process all nodes in the agent run with lifecycle events."""
        async for node in run:
            node_start_time = time.perf_counter()

            if Agent.is_user_prompt_node(node) or Agent.is_end_node(node):
                # Skip user_prompt and end nodes - their info is in AgentExecution events
                continue
            elif Agent.is_model_request_node(node):
                await handle_model_request_node(node, run, node_start_time)
            elif Agent.is_call_tools_node(node):
                await handle_call_tools_node(node, run, node_start_time)

    async def run_agent_iteration(
        effective_user_prompt: UserPromptT | None,
        effective_deferred_tool_results: DeferredToolResults | None,
        execution_start_time: float,
    ) -> None:
        """Run the agent iteration with hooks and lifecycle events."""
        await emit_lifecycle_event(
            AgentExecutionStartEvent(
                event_id=ctx.run_id,
                user_prompt=effective_user_prompt,
                deferred_tool_results=effective_deferred_tool_results,
                message_history_count=len(message_history) if message_history else 0,
            )
        )

        async with agent.iter(
            effective_user_prompt,
            deps=ctx,
            usage_limits=usage_limits,
            message_history=message_history,
            deferred_tool_results=effective_deferred_tool_results,
            metadata=cast(dict[str, Any], metadata) if metadata else None,
        ) as run:
            streamer.run = run

            if on_agent_start:
                await on_agent_start(
                    AgentStartContext(runtime=runtime, agent_info=main_agent_info, output_queue=output_queue, run=run)
                )

            await process_all_nodes(run)

            if on_agent_complete:
                await on_agent_complete(
                    AgentCompleteContext(
                        runtime=runtime, agent_info=main_agent_info, output_queue=output_queue, run=run
                    )
                )

            await emit_lifecycle_event(
                AgentExecutionCompleteEvent(
                    event_id=ctx.run_id,
                    total_loops=tracker.loop_index,
                    total_duration_seconds=time.perf_counter() - execution_start_time,
                    final_message_count=len(run.all_messages()),
                )
            )

    async def run_main() -> None:
        """Run the main agent and push events to output_queue."""
        logger.debug("Main agent task started")

        effective_user_prompt = user_prompt
        effective_deferred_tool_results = deferred_tool_results
        execution_start_time = time.perf_counter()

        try:
            async with runtime:
                if user_prompt_factory:
                    effective_user_prompt = await user_prompt_factory(runtime)

                if on_runtime_ready:
                    ready_ctx = RuntimeReadyContext(
                        runtime=runtime,
                        agent_info=main_agent_info,
                        output_queue=output_queue,
                        user_prompt=effective_user_prompt,
                        deferred_tool_results=effective_deferred_tool_results,
                    )
                    await on_runtime_ready(ready_ctx)
                    effective_user_prompt = ready_ctx.user_prompt
                    effective_deferred_tool_results = ready_ctx.deferred_tool_results

                await run_agent_iteration(effective_user_prompt, effective_deferred_tool_results, execution_start_time)

        except BaseException as e:
            if isinstance(e, asyncio.CancelledError):
                logger.debug("Main agent task cancelled")
            else:
                logger.exception("Error in main agent task")
                await emit_lifecycle_event(
                    AgentExecutionFailedEvent(
                        event_id=ctx.run_id,
                        error=str(e),
                        error_type=type(e).__name__,
                        total_loops=tracker.loop_index,
                        total_duration_seconds=time.perf_counter() - execution_start_time,
                    )
                )
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
        # Cancel all running tasks first to ensure clean shutdown
        # This handles both explicit interrupt() calls and external cancellation (e.g., Ctrl+C)
        for task in streamer._tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete and capture any exception
        results = await asyncio.gather(main_task, poll_task, return_exceptions=True)

        # Find first real exception (non-CancelledError)
        exceptions = [r for r in results if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError)]

        if streamer._interrupted:
            streamer.exception = AgentInterrupted("Agent execution was interrupted")
        elif exceptions:
            streamer.exception = exceptions[0]
