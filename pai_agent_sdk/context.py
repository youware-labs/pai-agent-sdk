"""Agent context management.

This module provides the AgentContext class for managing session state
during agent execution. AgentContext is designed to be used inside an
Environment context.

Architecture:
    Environment (outer, long-lived)
      - Manages tmp_dir lifecycle
      - Creates and owns file_operator and shell
      - async with environment as env:

        AgentContext (inner, short-lived)
          - Manages session state (run_id, timing, handoff)
          - Receives file_operator, shell as parameters
          - async with AgentContext(file_operator, shell) as ctx:

Example:
    Using AsyncExitStack for flat structure (recommended for dependent contexts):

    ```python
    from contextlib import AsyncExitStack
    from pai_agent_sdk.environment.local import LocalEnvironment
    from pai_agent_sdk.context import AgentContext

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(tmp_base_dir=Path("/tmp"))
        )
        ctx = await stack.enter_async_context(
            AgentContext(file_operator=env.file_operator, shell=env.shell)
        )
        # Handle request
        await ctx.file_operator.read_file("test.txt")
    # Resources cleaned up when stack exits
    ```

    Multiple sessions sharing environment:

    ```python
    async with LocalEnvironment(tmp_base_dir=Path("/tmp")) as env:
        # First session
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
        ) as ctx1:
            await ctx1.file_operator.read_file("test.txt")

        # Second session (reuses same environment)
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
        ) as ctx2:
            ...
    # tmp_dir cleaned up when environment exits
    ```
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, TypedDict, cast
from xml.etree.ElementTree import Element, SubElement, tostring

from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_ai.usage import RunUsage

from pai_agent_sdk.environment.base import FileOperator, Shell
from pai_agent_sdk.utils import get_latest_request_usage

if TYPE_CHECKING:
    from typing import Self


class ModelCapability(str, Enum):
    """Model capabilities that can be used to describe what a model supports."""

    vision = "vision"
    """Model can process and understand images."""

    video_understanding = "video_understanding"
    """Model can process and understand video content."""

    document_understanding = "document_understanding"
    """Model can process and understand documents (PDF, etc.)."""


def _generate_run_id() -> str:
    from uuid import uuid4

    return uuid4().hex


def _xml_to_string(element: Element) -> str:
    """Convert XML element to formatted string."""
    from xml.dom.minidom import parseString

    rough_string = tostring(element, encoding="unicode")
    dom = parseString(rough_string)  # noqa: S318
    # Get pretty-printed XML, skip the XML declaration line
    lines = dom.toprettyxml(indent="  ").split("\n")[1:]
    # Remove empty lines
    return "\n".join(line for line in lines if line.strip())


class ToolConfig(BaseModel):
    """Tool-level configuration for fine-grained control."""

    enable_load_document: bool = False
    """Enable document URL parsing in LoadTool. Default disabled due to poor model support."""

    image_understanding_model: str | None = None
    """Model to use for image understanding. Falls back to AgentSettings.image_understanding_model."""

    image_understanding_model_settings: dict[str, Any] | None = None
    """Model settings for image understanding agent."""

    video_understanding_model: str | None = None
    """Model to use for video understanding. Falls back to AgentSettings.video_understanding_model."""

    video_understanding_model_settings: dict[str, Any] | None = None
    """Model settings for video understanding agent."""


class ModelConfig(BaseModel):
    """Model configuration for context management."""

    context_window: int | None = None
    """Total context window size in tokens."""

    handoff_threshold: float | None = None
    """Handoff threshold for context injection."""

    max_images: int = 20
    """Maximum number of images allowed in message history. Default is 20 (Claude's limit)."""

    max_videos: int = 1
    """Maximum number of videos allowed in message history. Default is 1."""

    support_gif: bool = True
    """Whether the model supports GIF images. If False, GIF images will be filtered out."""

    capabilities: set[ModelCapability] = Field(default_factory=set)
    """Set of capabilities supported by the model."""

    tool_config: ToolConfig = Field(default_factory=ToolConfig)
    """Tool-level configuration for fine-grained control."""

    def has_capability(self, capability: ModelCapability) -> bool:
        """Check if the model has a specific capability."""
        return capability in self.capabilities


class RunContextMetadata(TypedDict, total=False):
    """Metadata for RunContext passed to get_context_instructions.

    This TypedDict defines the expected structure of metadata passed via
    pydantic-ai's Agent metadata parameter. It enables handoff threshold
    warnings when the context window usage exceeds the configured threshold.

    Example:
        Using with Agent and HandoffTool for automatic context management::

            from contextlib import AsyncExitStack
            from pydantic_ai import Agent

            from pai_agent_sdk.context import AgentContext, ModelConfig, RunContextMetadata
            from pai_agent_sdk.environment.local import LocalEnvironment
            from pai_agent_sdk.filters.handoff import process_handoff_message
            from pai_agent_sdk.toolsets.core.base import Toolset
            from pai_agent_sdk.toolsets.core.context.handoff import HandoffTool

            async with AsyncExitStack() as stack:
                env = await stack.enter_async_context(LocalEnvironment())
                ctx = await stack.enter_async_context(
                    AgentContext(
                        file_operator=env.file_operator,
                        shell=env.shell,
                        model_cfg=ModelConfig(
                            context_window=200000,
                            handoff_threshold=0.5,
                        ),
                    )
                )
                toolset = Toolset(ctx, tools=[HandoffTool])
                agent = Agent(
                    'openai:gpt-4',
                    deps_type=AgentContext,
                    toolsets=[toolset],
                    history_processors=[process_handoff_message],
                    # Enable handoff tool via metadata - triggers threshold warning
                    metadata=lambda _: {'enable_handoff_tool': True},
                )
                result = await agent.run('Your prompt here', deps=ctx)
    """

    enable_handoff_tool: bool
    """Whether the handoff tool is enabled for this run."""


class AgentContext(BaseModel):
    """Context for a single agent session.

    AgentContext manages session-specific state including:
    - Run identification (run_id, parent_run_id)
    - Timing (start_at, end_at, elapsed_time)
    - Deferred tool metadata
    - Handoff messages

    The file_operator and shell are provided externally (typically from
    an Environment) and are not managed by AgentContext.

    Example:
        Using AsyncExitStack (recommended for dependent contexts):

        ```python
        from contextlib import AsyncExitStack

        async with AsyncExitStack() as stack:
            env = await stack.enter_async_context(LocalEnvironment())
            ctx = await stack.enter_async_context(
                AgentContext(file_operator=env.file_operator, shell=env.shell)
            )
            await ctx.file_operator.read_file("data.json")
        ```
        ```
    """

    model_config = {"arbitrary_types_allowed": True}

    run_id: str = Field(default_factory=_generate_run_id)
    """Unique identifier for this session run."""

    parent_run_id: str | None = None
    """Parent run_id if this is a subagent context."""

    start_at: datetime | None = None
    """Timestamp when the context was entered."""

    end_at: datetime | None = None
    """Timestamp when the context was exited."""

    deferred_tool_metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Metadata for deferred tool calls, keyed by tool_call_id."""

    handoff_message: str | None = None
    """Rendered handoff message to be injected into new context after handoff."""

    file_operator: FileOperator
    """File operator for file system operations. Provided by Environment."""

    shell: Shell
    """Shell executor for command execution. Provided by Environment."""

    model_cfg: ModelConfig | None = None
    """Model configuration for context management."""

    extra_usage: dict[str, RunUsage] = Field(default_factory=dict)
    """Extra usage from tool calls, keyed by tool_call_id."""

    _agent_name: str = "main"

    @property
    def elapsed_time(self) -> timedelta | None:
        """Return elapsed time since start, or None if not started.

        If session has ended, returns the final duration.
        If session is running, returns the current elapsed time.
        """
        if self.start_at is None:
            return None
        end = self.end_at if self.end_at else datetime.now()
        return end - self.start_at

    async def get_context_instructions(
        self,
        run_context: "RunContext[AgentContext] | None" = None,
    ) -> str:
        """Return runtime context instructions in XML format.

        Provides runtime information about the current session.

        Args:
            runtime_info: Additional runtime information to include.

        Returns:
            XML-formatted string with runtime context and optional system reminders.
        """
        parts: list[str] = []

        # Build runtime-context element
        root = Element("runtime-context")

        # Elapsed time
        elapsed = self.elapsed_time
        elapsed_str = f"{elapsed.total_seconds():.1f}s" if elapsed else "not started"
        SubElement(root, "elapsed-time").text = elapsed_str

        # Model configuration - only context_window
        if self.model_cfg is not None and self.model_cfg.context_window is not None:
            config = SubElement(root, "model-config")
            SubElement(config, "context-window").text = str(self.model_cfg.context_window)

        # Token usage from runtime info
        if (
            run_context
            and (request_usage := get_latest_request_usage(run_context.messages))
            and request_usage.total_tokens is not None
        ):
            usage_elem = SubElement(root, "token-usage")
            SubElement(usage_elem, "total-tokens").text = str(request_usage.total_tokens)

        parts.append(_xml_to_string(root))

        # Build system-reminder element (sibling to runtime-context)
        reminders: list[str] = []

        # Cast metadata to typed dict for type safety
        metadata = cast(RunContextMetadata, run_context.metadata if run_context and run_context.deps else {})

        # Handoff threshold warning
        if (
            metadata.get("enable_handoff_tool", False)
            and self.model_cfg is not None
            and self.model_cfg.context_window is not None
            and self.model_cfg.handoff_threshold is not None
            and run_context
            and (request_usage := get_latest_request_usage(run_context.messages))
        ):
            threshold_tokens = int(self.model_cfg.context_window * self.model_cfg.handoff_threshold)
            if request_usage.total_tokens >= threshold_tokens:
                reminders.append(
                    "IMPORTANT: **You have reached the handoff threshold, please calling the `handoff` tool "
                    "to summarize then continue the task at the appropriate time.**"
                )

        if reminders:
            reminder_root = Element("system-reminder")
            for reminder_text in reminders:
                item = SubElement(reminder_root, "item")
                item.text = reminder_text
            parts.append(_xml_to_string(reminder_root))

        return "\n\n".join(parts)

    @asynccontextmanager
    async def enter_subagent(
        self,
        agent_name: str,
        **override: Any,
    ) -> AsyncGenerator["Self", None]:
        """Create a child context for subagent with independent timing.

        The subagent context inherits all fields but gets:
        - A new run_id
        - parent_run_id set to current run_id
        - Fresh start_at/end_at for independent timing
        - Shared file_operator and shell from parent

        Args:
            agent_name: Name of the subagent.
            **override: Additional fields to override in the subagent context.
                Subclasses can pass extra fields without overriding this method.
        """
        update: dict[str, Any] = {
            "run_id": _generate_run_id(),
            "parent_run_id": self.run_id,
            "start_at": datetime.now(),
            "end_at": None,
            "handoff_message": None,  # Subagents don't inherit handoff state
            **override,
        }
        new_ctx = self.model_copy(update=update)
        new_ctx._agent_name = agent_name
        try:
            yield new_ctx
        finally:
            new_ctx.end_at = datetime.now()

    async def __aenter__(self):
        """Enter the context and start timing."""
        self.start_at = datetime.now()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and record end time."""
        self.end_at = datetime.now()
