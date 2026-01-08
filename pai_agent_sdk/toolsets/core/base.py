"""Base classes for toolsets and tools.

This module provides the foundational abstractions for building toolsets:
- BaseTool: Abstract base class for individual tools
- Toolset: Container for tools with hooks and HITL support
- InstructableToolset: Protocol for toolsets that provide instructions
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field
from pydantic_ai import RunContext, Tool, UserError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import (
    DeferredToolResults,
    ToolApproved,
    ToolDenied,
)
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from typing_extensions import TypeVar

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.utils import get_tool_name_from_id

logger = get_logger(__name__)

AgentDepsT = TypeVar("AgentDepsT", bound=AgentContext, default=AgentContext, contravariant=True)


@runtime_checkable
class InstructableToolset(Protocol[AgentDepsT]):
    """Protocol for toolsets that provide instructions.

    This enables duck typing for any toolset that has a get_instructions method,
    allowing add_toolset_instructions() to work with both Toolset and BrowserUseToolset.
    """

    def get_instructions(self, ctx: RunContext[AgentDepsT]) -> str | None:
        """Get instructions to inject into the system prompt."""
        ...


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


class UserInputPreprocessResult(BaseModel):
    """Result from processing user input in HITL scenarios."""

    override_args: dict[str, Any] | None = None
    """Override arguments for the tool call."""

    metadata: dict[str, Any] | None = None
    """Additional metadata from user input processing."""


PreHookFunc = Callable[[RunContext[AgentDepsT], dict[str, Any]], Awaitable[dict[str, Any]]]
"""Pre-hook function signature: (ctx, tool_args) -> modified_tool_args"""

PostHookFunc = Callable[[RunContext[AgentDepsT], Any], Awaitable[Any]]
"""Post-hook function signature: (ctx, result) -> modified_result"""

GlobalPreHookFunc = Callable[[RunContext[AgentDepsT], str, dict[str, Any]], Awaitable[dict[str, Any]]]
"""Global pre-hook function signature: (ctx, tool_name, tool_args) -> modified_tool_args"""

GlobalPostHookFunc = Callable[[RunContext[AgentDepsT], str, Any], Awaitable[Any]]
"""Global post-hook function signature: (ctx, tool_name, result) -> modified_result"""


class GlobalHooks(BaseModel):
    """Container for global hooks applied to all tools."""

    pre: GlobalPreHookFunc[Any] | None = None
    """Global pre-hook applied before tool-specific pre-hooks."""

    post: GlobalPostHookFunc[Any] | None = None
    """Global post-hook applied after tool-specific post-hooks."""

    model_config = {"arbitrary_types_allowed": True}


class BaseTool(ABC):
    """Abstract base class for tools.

    Subclasses define name, description as class attributes, implement
    the `call` method, and optionally override `get_instruction()` for
    dynamic instruction generation.

    Example:
        class ReadFileTool(BaseTool):
            name = "read_file"
            description = "Read contents of a file"

            def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
                return "Use this tool to read file contents from the filesystem."

            async def call(self, ctx: RunContext[AgentContext], path: str) -> str:
                return Path(path).read_text()
    """

    name: str
    """The name of the tool, used for invocation."""

    description: str
    """Description of what the tool does, shown to the model."""

    def __init__(self, ctx: AgentContext) -> None:
        """Initialize the tool with the agent context.

        Args:
            ctx: The agent context for this tool instance.
        """
        self.ctx = ctx

    def is_available(self) -> bool:
        """Check if tool is available in current context.

        Override this method to check runtime conditions like model capabilities,
        optional dependencies, or configuration settings.
        Tools that return False will be excluded when skip_unavailable=True.

        Returns:
            True if tool can be used, False otherwise.
        """
        return True

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Get instruction for this tool.

        Override this method to provide dynamic instructions based on context.
        Default implementation returns None (no instruction).

        Args:
            ctx: The run context containing runtime information.

        Returns:
            Instruction text to inject into system prompt, or None.
        """
        return None

    @abstractmethod
    async def call(self, ctx: RunContext[AgentContext], /, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool logic.

        Subclasses should override this method with their specific parameter signature.
        The base signature uses *args/**kwargs to allow any parameter combination.

        Args:
            ctx: The run context containing runtime information.
            *args: Tool-specific positional arguments.
            **kwargs: Tool-specific keyword arguments.

        Returns:
            The tool's result.
        """
        ...

    async def process_user_input(
        self,
        ctx: AgentContext,
        user_input: Any,
    ) -> UserInputPreprocessResult | None:
        """Process user input for HITL scenarios.

        Override this method to handle user-provided input when a tool call
        requires approval. You can use this to:
        - Validate user input
        - Transform user input into tool arguments
        - Store metadata for later use

        Args:
            ctx: The agent context.
            user_input: The user-provided input data.

        Returns:
            A UserInputPreprocessResult with override_args and/or metadata,
            or None if no processing is needed.
        """
        return None

    def get_deferred_metadata(self, ctx: RunContext[AgentContext]) -> dict[str, Any] | None:
        """Get HITL metadata for the tool call, if applicable."""
        # TODO: Get it from ctx.tool_call_metadata when supported: https://github.com/pydantic/pydantic-ai/pull/3811
        if not ctx.tool_call_id:
            return None
        return ctx.deps.deferred_tool_metadata.get(ctx.tool_call_id)


class BaseToolset(AbstractToolset[AgentDepsT], ABC):
    """Base class for toolsets with instruction support."""

    def get_instructions(self, ctx: RunContext[AgentDepsT]) -> str | None:
        """Get instructions to inject into the system prompt.

        Override this method to provide tool-specific instructions.
        """
        return None


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
        ctx = AgentContext()
        toolset = Toolset(
            ctx,
            tools=[ReadFileTool, WriteFileTool],
            pre_hooks={"read_file": my_pre_hook},
            post_hooks={"write_file": my_post_hook},
            global_hooks=GlobalHooks(pre=global_pre, post=global_post),
        )

        agent = Agent(model="openai:gpt-4", toolsets=[toolset])
    """

    def __init__(
        self,
        ctx: AgentContext,
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
            ctx: The agent context, passed to all tool instances.
            tools: Sequence of BaseTool classes to include in this toolset.
            pre_hooks: Dict mapping tool names to pre-hook functions.
            post_hooks: Dict mapping tool names to post-hook functions.
            global_hooks: Global hooks applied to all tools.
            max_retries: Maximum retries for tool execution.
            timeout: Default timeout for tool execution.
            toolset_id: Optional unique ID for the toolset.
            skip_unavailable: If True, skip tools where is_available() returns False.
        """
        self.ctx = ctx
        self.max_retries = max_retries
        self.timeout = timeout
        self._id = toolset_id

        self.pre_hooks = pre_hooks or {}
        self.post_hooks = post_hooks or {}
        self.global_hooks = global_hooks or GlobalHooks()

        self._tool_instances: dict[str, BaseTool] = {}
        self._tool_classes: dict[str, type[BaseTool]] = {}

        for tool_cls in tools:
            tool_instance = tool_cls(ctx)
            name = tool_instance.name

            # Check availability after instantiation
            if skip_unavailable and not tool_instance.is_available():
                logger.info(f"Skipping unavailable tool {name!r}")
                continue

            if name in self._tool_instances:
                msg = f"Duplicate tool name: {name!r}"
                raise UserError(msg)
            self._tool_instances[name] = tool_instance
            self._tool_classes[name] = tool_cls

        self._pydantic_tools: dict[str, Tool[AgentDepsT]] = {}

    @property
    def id(self) -> str | None:
        """Return the toolset ID."""
        return self._id

    @property
    def tool_names(self) -> list[str]:
        """Return list of available tool names in this toolset."""
        return list(self._tool_classes.keys())

    def subset(
        self,
        tool_names: list[str] | None = None,
        *,
        ctx: AgentContext | None = None,
        inherit_hooks: bool = False,
    ) -> Toolset[AgentDepsT]:
        """Create a subset Toolset with only the specified tools.

        This is useful for creating subagent toolsets that only have access to
        a subset of the parent agent's tools.

        Args:
            tool_names: List of tool names to include. None means all tools.
            ctx: Optional new context for the subset. If None, uses self.ctx.
            inherit_hooks: Whether to inherit pre/post hooks for selected tools.

        Returns:
            A new Toolset instance with the selected tools.

        Example::

            # Create main toolset
            main_toolset = Toolset(ctx, tools=[ViewTool, EditTool, ShellTool])

            # Create subset for subagent (only view and edit)
            sub_toolset = main_toolset.subset(["view", "edit"])

            # Create subset with new context
            sub_toolset = main_toolset.subset(["view"], ctx=subagent_ctx)
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
                    logger.warning(f"Tool {name!r} not found in parent toolset, skipping")

        pre_hooks: dict[str, PreHookFunc[AgentDepsT]] | None = None
        post_hooks: dict[str, PostHookFunc[AgentDepsT]] | None = None
        global_hooks: GlobalHooks | None = None

        if inherit_hooks:
            pre_hooks = {k: v for k, v in self.pre_hooks.items() if k in selected_names}
            post_hooks = {k: v for k, v in self.post_hooks.items() if k in selected_names}
            global_hooks = self.global_hooks

        return Toolset(
            ctx=ctx or self.ctx,
            tools=selected_classes,
            pre_hooks=pre_hooks,
            post_hooks=post_hooks,
            global_hooks=global_hooks,
            max_retries=self.max_retries,
            timeout=self.timeout,
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
        tools: dict[str, ToolsetTool[AgentDepsT]] = {}

        for name, tool_instance in self._tool_instances.items():
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

        Post-hooks receive the result, which may be an Exception instance if the tool
        execution failed. Hooks can:
        - Log/monitor errors by checking `isinstance(result, Exception)`
        - Transform errors into user-friendly messages
        - Recover from certain errors by returning a fallback value

        If the final result is still an Exception after all hooks, it will be raised.
        """
        if not isinstance(tool, HookableToolsetTool):
            msg = f"Expected HookableToolsetTool, got {type(tool)}"
            raise UserError(msg)

        args = tool_args

        if self.global_hooks.pre:
            args = await self.global_hooks.pre(ctx, name, args)

        if tool.pre_hook:
            args = await tool.pre_hook(ctx, args)

        result = await self._call_tool_func(args, ctx, tool)

        if tool.post_hook:
            result = await tool.post_hook(ctx, result)

        if self.global_hooks.post:
            result = await self.global_hooks.post(ctx, name, result)

        # Re-raise if result is still an exception after hooks
        if isinstance(result, BaseException):
            raise result

        return result

    def get_instructions(self, ctx: RunContext[AgentDepsT]) -> str | None:
        """Collect instructions from all tools.

        Returns a combined instruction string or None if no tools have instructions.
        """
        instructions: list[str] = []
        for tool_instance in self._tool_instances.values():
            instruction = tool_instance.get_instruction(ctx)
            if instruction:
                instructions.append(f'<tool-instruction name="{tool_instance.name}">{instruction}</tool-instruction>')

        return "\n".join(instructions) if instructions else None

    def _get_tool_impl_by_name(self, name: str) -> BaseTool | None:
        """Get a tool instance by name."""
        return self._tool_instances.get(name)

    async def process_hitl_call(
        self,
        user_interactions: list[UserInteraction] | None,
        message_history: list[ModelMessage],
    ) -> DeferredToolResults | None:
        """Process HITL interactions and return deferred tool results.

        Args:
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
                                self.ctx,
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
                    # TODO: Switch to DeferredToolResults.metadata https://github.com/pydantic/pydantic-ai/pull/3811
                    #     results.metadata[interaction.tool_call_id] = metadata
                    self.ctx.deferred_tool_metadata[interaction.tool_call_id] = metadata
                logger.info(f"User approved tool call: {interaction.tool_call_id}")
            else:
                reason = interaction.reason or "User rejected the tool call."
                results.approvals[interaction.tool_call_id] = ToolDenied(message=reason)
                logger.info(f"User rejected tool call: {interaction.tool_call_id}, reason: {reason}")

        return results
