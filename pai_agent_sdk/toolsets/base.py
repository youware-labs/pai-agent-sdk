"""Base classes for toolsets and tools.

This module provides the foundational abstractions for building toolsets:
- BaseTool: Abstract base class for individual tools
- BaseToolset: Abstract base class for toolsets with instruction support
"""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai.toolsets import AbstractToolset
from typing_extensions import TypeVar

from pai_agent_sdk.context import AgentContext

if TYPE_CHECKING:
    pass

AgentDepsT = TypeVar("AgentDepsT", bound=AgentContext, default=AgentContext, contravariant=True)


class UserInputPreprocessResult(BaseModel):
    """Result from processing user input in HITL scenarios."""

    override_args: dict[str, Any] | None = None
    """Override arguments for the tool call."""

    metadata: dict[str, Any] | None = None
    """Additional metadata from user input processing."""


@runtime_checkable
class InstructableToolset(Protocol[AgentDepsT]):
    """Protocol for toolsets that provide instructions.

    This enables duck typing for any toolset that has a get_instructions method,
    allowing add_toolset_instructions() to work with both Toolset and BrowserUseToolset.

    Supports both sync and async get_instructions methods.

    TODO: Drop it when https://github.com/pydantic/pydantic-ai/pull/3780 merged
    """

    def get_instructions(self, ctx: RunContext[AgentDepsT]) -> str | Awaitable[str | None] | None:
        """Get instructions to inject into the system prompt.

        Can be implemented as either sync or async method.
        """
        ...


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

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available in current context.

        Override this method to check runtime conditions like model capabilities,
        optional dependencies, or configuration settings.
        Tools that return False will be excluded when skip_unavailable=True.

        Args:
            ctx: The run context containing runtime information.

        Returns:
            True if tool can be used, False otherwise.
        """
        return True

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str | Awaitable[str] | None:
        """Get instruction for this tool.

        Override this method to provide dynamic instructions based on context.
        Can be implemented as either sync or async method.
        Default implementation returns None (no instruction).

        Args:
            ctx: The run context containing runtime information.

        Returns:
            Instruction text to inject into system prompt, or None.
        """
        return None

    def get_approval_metadata(self) -> dict[str, Any] | None:
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

    def get_deferred_metadata(self, ctx: RunContext[AgentContext]) -> Any:
        """Get HITL metadata for the tool call, if applicable."""
        return ctx.tool_call_metadata


class BaseToolset(AbstractToolset[AgentDepsT], ABC):
    """Base class for toolsets with instruction support.

    Subclasses can override get_instructions() as either sync or async method.
    The framework will handle both cases automatically.

    Example (sync):
        class MyToolset(BaseToolset):
            def get_instructions(self, ctx: RunContext[AgentContext]) -> str | None:
                return "My instructions"

    Example (async):
        class MyAsyncToolset(BaseToolset):
            async def get_instructions(self, ctx: RunContext[AgentContext]) -> str | None:
                content = await self._load_instructions(ctx)
                return content
    """

    def get_instructions(self, ctx: RunContext[AgentDepsT]) -> str | Awaitable[str | None] | None:
        """Get instructions to inject into the system prompt.

        Override this method to provide tool-specific instructions.
        Can be implemented as either sync or async method.

        Args:
            ctx: The run context containing runtime information.

        Returns:
            Instruction string, awaitable returning string/None, or None.
        """
        return None


async def resolve_instructions(result: str | Awaitable[str | None] | None) -> str | None:
    """Resolve instruction result that may be sync or async.

    Args:
        result: The result from get_instructions(), which may be:
            - str: Return as-is
            - Awaitable[str]: Await and return
            - None: Return None

    Returns:
        The resolved instruction string, or None.
    """
    if result is None:
        return None
    if isinstance(result, str):
        return result
    if inspect.isawaitable(result):
        return await result
    # Fallback for coroutine objects
    if asyncio.iscoroutine(result):
        return await result
    return result
