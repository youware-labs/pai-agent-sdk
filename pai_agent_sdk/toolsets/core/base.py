"""Toolset implementation with hooks and HITL support.

This module provides:
- Toolset: Container for tools with hooks and HITL support
- HookableToolsetTool: Extended ToolsetTool with hook support
- Hook types and utilities

Base classes (BaseTool, BaseToolset) are in pai_agent_sdk.toolsets.base.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from pydantic_ai import ApprovalRequired, RunContext, Tool, UserError
from pydantic_ai._agent_graph import HistoryProcessor
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import (
    DeferredToolResults,
    ToolApproved,
    ToolDenied,
)
from pydantic_ai.toolsets.abstract import ToolsetTool
from typing_extensions import TypeVar

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.base import (
    BaseTool,
    BaseToolset,
    resolve_instructions,
)
from pai_agent_sdk.utils import get_tool_name_from_id

if TYPE_CHECKING:
    from pydantic_ai import ModelSettings
    from pydantic_ai.models import Model

    from pai_agent_sdk.context import ModelConfig
    from pai_agent_sdk.subagents import SubagentConfig

logger = get_logger(__name__)

AgentDepsT = TypeVar("AgentDepsT", bound=AgentContext, default=AgentContext, contravariant=True)


class UserInteraction(BaseModel):
    """Represents a user's interaction with a deferred tool call."""

    tool_call_id: str = Field(..., description="The ID of the tool interaction.")
    approved: bool = Field(
        ...,
        description="Whether the user approved the previous action. "
        "If false, the 'reason' field may provide additional context. "
        "If true, 'user_input' may contain additional data provided by the user.",
    )
    reason: str | None = Field(None, description="The reason for rejection, if any.")
    user_input: Any = Field(
        None,
        description="Additional user input data. Structure depends on tool implementation.",
    )


CallMetadata = dict[str, Any]
"""Metadata dictionary shared across hooks within a single call_tool invocation."""

PreHookFunc = Callable[[RunContext[AgentDepsT], dict[str, Any], CallMetadata], Awaitable[dict[str, Any]]]
"""Pre-hook function signature: (ctx, tool_args, metadata) -> modified_tool_args"""

PostHookFunc = Callable[[RunContext[AgentDepsT], Any, CallMetadata], Awaitable[Any]]
"""Post-hook function signature: (ctx, result, metadata) -> modified_result"""

GlobalPreHookFunc = Callable[[RunContext[AgentDepsT], str, dict[str, Any], CallMetadata], Awaitable[dict[str, Any]]]
"""Global pre-hook function signature: (ctx, tool_name, tool_args, metadata) -> modified_tool_args"""

GlobalPostHookFunc = Callable[[RunContext[AgentDepsT], str, Any, CallMetadata], Awaitable[Any]]
"""Global post-hook function signature: (ctx, tool_name, result, metadata) -> modified_result"""


class GlobalHooks(BaseModel):
    """Container for global hooks applied to all tools."""

    pre: GlobalPreHookFunc[Any] | None = None
    """Global pre-hook applied before tool-specific pre-hooks."""

    post: GlobalPostHookFunc[Any] | None = None
    """Global post-hook applied after tool-specific post-hooks."""

    model_config = {"arbitrary_types_allowed": True}


@dataclass(kw_only=True)
class HookableToolsetTool(ToolsetTool[AgentDepsT]):
    """Extended ToolsetTool with hook and instruction support."""

    call_func: Callable[[dict[str, Any], RunContext[AgentDepsT]], Awaitable[Any]]
    """The function to call when the tool is invoked."""

    is_async: bool
    """Whether the underlying function is async."""

    timeout: float | None = None
    """Timeout in seconds for tool execution."""

    pre_hook: PreHookFunc[AgentDepsT] | None = None
    """Tool-specific pre-hook."""

    post_hook: PostHookFunc[AgentDepsT] | None = None
    """Tool-specific post-hook."""

    tool_instance: BaseTool | None = None
    """Reference to the BaseTool instance for HITL processing."""


class Toolset(BaseToolset[AgentDepsT]):
    """A toolset that manages tools with hooks and HITL support.

    Example:
        toolset = Toolset(
            tools=[ReadFileTool, WriteFileTool],
            pre_hooks={"read_file": my_pre_hook},
            post_hooks={"write_file": my_post_hook},
            global_hooks=GlobalHooks(pre=global_pre, post=global_post),
        )

        agent = Agent(model="openai:gpt-4", toolsets=[toolset])
    """

    def __init__(
        self,
        tools: Sequence[type[BaseTool]],
        *,
        pre_hooks: dict[str, PreHookFunc[AgentDepsT]] | None = None,
        post_hooks: dict[str, PostHookFunc[AgentDepsT]] | None = None,
        global_hooks: GlobalHooks | None = None,
        max_retries: int = 3,
        timeout: float | None = None,
        toolset_id: str | None = None,
        skip_unavailable: bool = True,
    ) -> None:
        """Initialize the toolset.

        Args:
            tools: Sequence of BaseTool classes to include in this toolset.
            pre_hooks: Dict mapping tool names to pre-hook functions.
            post_hooks: Dict mapping tool names to post-hook functions.
            global_hooks: Global hooks applied to all tools.
            max_retries: Maximum retries for tool execution.
            timeout: Default timeout for tool execution.
            toolset_id: Optional unique ID for the toolset.
            skip_unavailable: If True, skip tools where is_available() returns False in get_tools().
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self._id = toolset_id
        self._skip_unavailable = skip_unavailable

        self.pre_hooks = pre_hooks or {}
        self.post_hooks = post_hooks or {}
        self.global_hooks = global_hooks or GlobalHooks()

        # Store tool classes, instances created lazily in get_tools
        self._tool_classes: dict[str, type[BaseTool]] = {}
        self._tool_instances: dict[str, BaseTool] = {}

        logger.debug(f"Initializing Toolset with {len(tools)} tool classes")
        for tool_cls in tools:
            name = tool_cls.name
            if name in self._tool_classes:
                msg = f"Duplicate tool name: {name!r}"
                raise UserError(msg)
            self._tool_classes[name] = tool_cls
            logger.debug(f"Registered tool class: {name!r}")

        self._pydantic_tools: dict[str, Tool[AgentDepsT]] = {}
        logger.debug(f"Toolset initialized with tools: {list(self._tool_classes.keys())}")

    @property
    def id(self) -> str | None:
        """Get the toolset ID."""
        return self._id

    @property
    def tool_names(self) -> list[str]:
        """Get list of tool names in this toolset."""
        return list(self._tool_classes.keys())

    def _get_tool_instance(self, name: str) -> BaseTool:
        """Get or create a tool instance by name."""
        if name not in self._tool_instances:
            if name not in self._tool_classes:
                msg = f"Tool {name!r} not found in toolset"
                raise UserError(msg)
            self._tool_instances[name] = self._tool_classes[name]()
        return self._tool_instances[name]

    def is_tool_available(
        self,
        tool_name: str,
        ctx: RunContext[AgentDepsT],
    ) -> bool:
        """Check if a tool is available.

        Args:
            tool_name: The name of the tool to check.
            ctx: The run context for checking runtime availability.

        Returns:
            True if the tool exists and is available, False otherwise.
        """
        if tool_name not in self._tool_classes:
            return False
        tool_instance = self._get_tool_instance(tool_name)
        return tool_instance.is_available(ctx)

    def subset(
        self,
        tool_names: list[str] | None = None,
        *,
        inherit_hooks: bool = False,
    ) -> Toolset[AgentDepsT]:
        """Create a subset Toolset with only the specified tools.

        This is useful for creating subagent toolsets that only have access to
        a subset of the parent agent's tools.

        Args:
            tool_names: List of tool names to include. None means all tools.
            inherit_hooks: Whether to inherit pre/post hooks for selected tools.

        Returns:
            A new Toolset instance with the selected tools.

        Example::

            # Create main toolset
            main_toolset = Toolset(tools=[ViewTool, EditTool, ShellTool])

            # Create subset for subagent (only view and edit)
            sub_toolset = main_toolset.subset(["view", "edit"])
        """
        if tool_names is None:
            selected_classes = list(self._tool_classes.values())
            selected_names = set(self._tool_classes.keys())
        else:
            selected_classes = []
            selected_names: set[str] = set()
            for name in tool_names:
                if name in self._tool_classes:
                    selected_classes.append(self._tool_classes[name])
                    selected_names.add(name)
                else:
                    logger.debug(f"Tool {name!r} not found in parent toolset, skipping")

        pre_hooks: dict[str, PreHookFunc[AgentDepsT]] | None = None
        post_hooks: dict[str, PostHookFunc[AgentDepsT]] | None = None
        global_hooks: GlobalHooks | None = None

        if inherit_hooks:
            pre_hooks = {k: v for k, v in self.pre_hooks.items() if k in selected_names}
            post_hooks = {k: v for k, v in self.post_hooks.items() if k in selected_names}
            global_hooks = self.global_hooks

        return Toolset(
            tools=selected_classes,
            pre_hooks=pre_hooks,
            post_hooks=post_hooks,
            global_hooks=global_hooks,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

    def with_subagents(
        self,
        configs: Sequence[SubagentConfig],
        *,
        model: str | Model | None = None,
        model_settings: ModelSettings | dict[str, Any] | str | None = None,
        history_processors: Sequence[HistoryProcessor[AgentContext]] | None = None,
        model_cfg: ModelConfig | None = None,
    ) -> Toolset[AgentDepsT]:
        """Create a new Toolset that includes subagent tools.

        Subagent tools are created from the provided configurations, using this
        toolset as the parent. Each subagent will have access to a subset of
        tools as specified in its configuration.

        Args:
            configs: Sequence of SubagentConfig defining the subagents to create.
            model: Fallback model for subagents with 'inherit' or None model.
            model_settings: Fallback model settings for subagents with 'inherit' or None.
            history_processors: History processors for subagents.
            model_cfg: Fallback ModelConfig for subagents.

        Returns:
            A new Toolset instance with subagent tools added.

        Example::

            from pai_agent_sdk.subagents import SubagentConfig

            # Create toolset with subagents
            config = SubagentConfig(
                name="debugger",
                description="Debug code issues",
                system_prompt="You are a debugging expert...",
                tools=["grep", "view"],
            )
            toolset_with_subs = toolset.with_subagents(
                [config],
                model="anthropic:claude-sonnet-4",
            )
        """
        # Import here to avoid circular dependency
        from pai_agent_sdk.subagents import create_subagent_tool_from_config

        if not configs:
            return self

        subagent_tools = [
            create_subagent_tool_from_config(
                cfg,
                parent_toolset=self,
                model=model,
                model_settings=model_settings,
                history_processors=history_processors,
                model_cfg=model_cfg,
            )
            for cfg in configs
        ]
        all_tools = list(self._tool_classes.values()) + subagent_tools

        return Toolset(
            tools=all_tools,
            pre_hooks=self.pre_hooks,
            post_hooks=self.post_hooks,
            global_hooks=self.global_hooks,
            max_retries=self.max_retries,
            timeout=self.timeout,
            toolset_id=self._id,
        )

    def _create_pydantic_tool(self, name: str, tool_instance: BaseTool) -> Tool[AgentDepsT]:
        """Create a pydantic_ai Tool wrapper for a BaseTool instance."""

        @functools.wraps(tool_instance.call)
        async def _call(ctx: RunContext[AgentDepsT], **kwargs: Any) -> Any:
            return await tool_instance.call(ctx, **kwargs)

        return Tool(
            function=_call,
            name=name,
            description=tool_instance.description,
            max_retries=self.max_retries,
            takes_ctx=True,
        )

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Return all tools in this toolset."""
        logger.debug(f"get_tools called, preparing {len(self._tool_classes)} tools")
        tools: dict[str, ToolsetTool[AgentDepsT]] = {}

        for name in self._tool_classes:
            tool_instance = self._get_tool_instance(name)
            # Check availability at get_tools time (when env is entered)
            if self._skip_unavailable and not tool_instance.is_available(ctx):
                logger.debug(f"Skipping unavailable tool {name!r}")
                continue

            # Get or create pydantic_ai Tool wrapper
            if name not in self._pydantic_tools:
                self._pydantic_tools[name] = self._create_pydantic_tool(name, tool_instance)

            pydantic_tool = self._pydantic_tools[name]
            tool_def = await pydantic_tool.prepare_tool_def(ctx)
            if not tool_def:
                continue

            tools[name] = HookableToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=self.max_retries,
                args_validator=pydantic_tool.function_schema.validator,
                call_func=pydantic_tool.function_schema.call,
                is_async=pydantic_tool.function_schema.is_async,
                timeout=self.timeout,
                pre_hook=self.pre_hooks.get(name),
                post_hook=self.post_hooks.get(name),
                tool_instance=tool_instance,
            )

        return tools

    async def _call_tool_func(
        self,
        args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: HookableToolsetTool[AgentDepsT],
    ) -> Any:
        """Execute the tool function and capture exceptions.

        Subclasses can override this method to customize tool execution,
        e.g., adding timeout handling, retry logic, or custom error handling.

        Args:
            args: The validated tool arguments.
            ctx: The run context.
            tool: The tool to execute.

        Returns:
            The tool result, or an Exception if execution failed.
        """
        try:
            return await tool.call_func(args, ctx)
        except Exception as e:
            return e

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        """Call a tool with hooks.

        Execution order: global_pre -> tool_pre -> execute -> tool_post -> global_post

        A shared metadata dictionary is created at the start of each call_tool invocation
        and passed to all hooks. This allows hooks to share data within a single tool call:
        - Pre-hooks can store data (e.g., start_time, tracing spans)
        - Post-hooks can read that data (e.g., calculate duration, close spans)

        Post-hooks receive the result, which may be an Exception instance if the tool
        execution failed. Hooks can:
        - Log/monitor errors by checking `isinstance(result, Exception)`
        - Transform errors into user-friendly messages
        - Recover from certain errors by returning a fallback value

        If the final result is still an Exception after all hooks, it will be raised.
        """
        logger.debug(f"call_tool: {name!r} with args keys: {list(tool_args.keys())}")

        if not isinstance(tool, HookableToolsetTool):
            msg = f"Expected HookableToolsetTool, got {type(tool)}"
            raise UserError(msg)

        if name in ctx.deps.need_user_approve_tools and not ctx.tool_call_approved:
            approval_metadata = tool.tool_instance.get_approval_metadata() if tool.tool_instance else None
            logger.debug(f"call_tool: {name!r} requires user approval")
            raise ApprovalRequired(metadata=approval_metadata)

        # Create call-scoped metadata dict shared across all hooks
        metadata: CallMetadata = {}
        args = tool_args

        if self.global_hooks.pre:
            logger.debug(f"call_tool: {name!r} executing global pre-hook")
            args = await self.global_hooks.pre(ctx, name, args, metadata)

        if tool.pre_hook:
            logger.debug(f"call_tool: {name!r} executing tool pre-hook")
            args = await tool.pre_hook(ctx, args, metadata)

        logger.debug(f"call_tool: {name!r} executing tool function")
        try:
            result = await self._call_tool_func(args, ctx, tool)
        except Exception as e:
            # Let the post-hook handle the exception
            logger.debug(f"call_tool: {name!r} tool function raised exception: {type(e).__name__}")
            result = e

        if tool.post_hook:
            logger.debug(f"call_tool: {name!r} executing tool post-hook")
            result = await tool.post_hook(ctx, result, metadata)

        if self.global_hooks.post:
            logger.debug(f"call_tool: {name!r} executing global post-hook")
            result = await self.global_hooks.post(ctx, name, result, metadata)

        # Wrap into a non-Exception result so won't break the agentic loop
        if isinstance(result, BaseException):
            logger.debug(f"call_tool: {name!r} raising exception: {type(result).__name__}")
            return f"Error calling tool {name}: {result}"

        logger.debug(f"call_tool: {name!r} completed successfully")
        return result

    async def get_instructions(self, ctx: RunContext[AgentDepsT]) -> str | None:
        """Collect instructions from all tools.

        Returns a combined instruction string or None if no tools have instructions.
        Supports both sync and async get_instruction methods on tools.
        """
        instructions: list[str] = []
        for name in self._tool_classes:
            tool_instance = self._get_tool_instance(name)
            instruction = await resolve_instructions(tool_instance.get_instruction(ctx))
            if instruction:
                instructions.append(f'<tool-instruction name="{tool_instance.name}">{instruction}</tool-instruction>')

        return "\n".join(instructions) if instructions else None

    def _get_tool_impl_by_name(self, name: str) -> BaseTool | None:
        """Get a tool instance by name."""
        if name not in self._tool_classes:
            return None
        return self._get_tool_instance(name)

    async def process_hitl_call(
        self,
        ctx: AgentContext,
        user_interactions: list[UserInteraction] | None,
        message_history: list[ModelMessage],
    ) -> DeferredToolResults | None:
        """Process HITL interactions and return deferred tool results.

        Args:
            ctx: The agent context for processing user input.
            user_interactions: List of user interactions for deferred tool calls.
            message_history: The message history to look up tool names from call IDs.

        Returns:
            DeferredToolResults with approvals, or None if no interactions.
        """
        if not user_interactions:
            return None

        results = DeferredToolResults()
        for interaction in user_interactions:
            if interaction.approved:
                override_args = None
                metadata = None

                if interaction.user_input is not None:
                    tool_name = get_tool_name_from_id(interaction.tool_call_id, message_history)
                    if tool_name and (tool_impl := self._get_tool_impl_by_name(tool_name)):
                        try:
                            process_result = await tool_impl.process_user_input(
                                ctx,
                                user_input=interaction.user_input,
                            )
                            if process_result:
                                override_args = process_result.override_args
                                metadata = process_result.metadata
                        except Exception:
                            logger.exception(f"Failed to process user input for tool '{tool_name}'")
                            results.approvals[interaction.tool_call_id] = ToolDenied(
                                message="Failed to process user input"
                            )
                            continue

                results.approvals[interaction.tool_call_id] = ToolApproved(override_args=override_args)
                if metadata:
                    results.metadata[interaction.tool_call_id] = metadata
                logger.info(f"User approved tool call: {interaction.tool_call_id}")
            else:
                reason = interaction.reason or "User rejected the tool call."
                results.approvals[interaction.tool_call_id] = ToolDenied(message=reason)
                logger.info(f"User rejected tool call: {interaction.tool_call_id}, reason: {reason}")

        return results
