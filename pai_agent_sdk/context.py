"""Agent context management.

This module provides the AgentContext class for managing session state
during agent execution. AgentContext holds a reference to an Environment
and derives file_operator/shell/resources from it via properties.

Architecture:
    AgentRuntime (recommended entry point)
      - Created by create_agent() factory function
      - Manages env -> ctx -> agent lifecycle as async context manager
      - stream_agent() handles runtime lifecycle automatically

    Environment (outer, long-lived)
      - Manages tmp_dir lifecycle
      - Creates and owns file_operator and shell
      - async with environment as env:

        AgentContext (inner, short-lived)
          - Manages session state (run_id, timing, handoff)
          - Holds env reference, derives file_operator/shell/resources from it
          - async with AgentContext(env=env) as ctx:

Example:
    Using create_agent and stream_agent (recommended)::

        from pai_agent_sdk.agents.main import create_agent, stream_agent

        # create_agent returns AgentRuntime (not a context manager)
        runtime = create_agent("openai:gpt-4")

        # stream_agent manages runtime lifecycle automatically
        async with stream_agent(runtime, "Hello") as streamer:
            async for event in streamer:
                print(event)

    Using create_agent with manual agent.run::

        runtime = create_agent("openai:gpt-4")
        async with runtime:  # Enter runtime to manage env/ctx/agent
            result = await runtime.agent.run("Hello", deps=runtime.ctx)
            print(result.output)

    Manual Environment and AgentContext setup (advanced)::

        from pai_agent_sdk.environment.local import LocalEnvironment
        from pai_agent_sdk.context import AgentContext

        async with LocalEnvironment() as env:
            async with AgentContext(env=env) as ctx:
                await ctx.file_operator.read_file("test.txt")

    Multiple sessions sharing environment::

        async with LocalEnvironment() as env:
            # First session
            async with AgentContext(env=env) as ctx1:
                await ctx1.file_operator.read_file("test.txt")

            # Second session (reuses same environment)
            async with AgentContext(env=env) as ctx2:
                ...
        # tmp_dir cleaned up when environment exits
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring

from agent_environment import Environment, FileOperator, ResourceRegistry, Shell
from pydantic import BaseModel, Field
from pydantic_ai import (
    DeferredToolRequests,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelSettings,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RunContext,
    ToolCallPartDelta,
    UserContent,
)
from pydantic_ai.messages import HandleResponseEvent as PydanticHandleResponseEvent
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponseStreamEvent,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import TypedDict

from pai_agent_sdk.events import AgentEvent
from pai_agent_sdk.usage import ExtraUsageRecord, InternalUsage
from pai_agent_sdk.utils import get_latest_request_usage

# =============================================================================
# Type Aliases
# =============================================================================

# Hook function type for converting media data to URL.
# Can be sync or async. Returns URL string or None to use default behavior.
MediaToUrlHook = Callable[["RunContext[AgentContext]", bytes, str], "Awaitable[str | None] | str | None"]


if TYPE_CHECKING:
    from typing import Self


# =============================================================================
# Task Management
# =============================================================================


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class Task(BaseModel):
    """A single task with dependencies and metadata.

    Tasks support blocking relationships where a task can block other tasks
    or be blocked by other tasks. When a blocking task is completed, the
    blocked tasks are automatically unblocked.

    Attributes:
        id: Unique task identifier (e.g., "1", "2").
        subject: Task title in imperative form (e.g., "Run tests").
        description: Detailed task description.
        active_form: Present progressive form shown during in_progress (e.g., "Running tests").
        status: Current task status.
        owner: Optional task owner/assignee.
        blocks: List of task IDs that this task blocks.
        blocked_by: List of task IDs that block this task.
        metadata: Additional task metadata.
        created_at: Task creation timestamp.
        updated_at: Last update timestamp.
    """

    id: str
    subject: str
    description: str
    active_form: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    owner: str | None = None
    blocks: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def is_blocked(self) -> bool:
        """Check if task is blocked by any incomplete tasks."""
        return len(self.blocked_by) > 0


class TaskManager(BaseModel):
    """Manager for task lifecycle and dependencies.

    Handles task creation, updates, and automatic dependency resolution.
    When a task is completed, it is automatically removed from the blocked_by
    list of tasks it was blocking.

    Example:
        manager = TaskManager()
        task1 = manager.create("Implement API", "Create REST endpoints")
        task2 = manager.create("Write tests", "Unit tests for API")
        manager.add_blocked_by(task2.id, [task1.id])  # task2 blocked by task1
        manager.update_status(task1.id, TaskStatus.COMPLETED)  # task2 unblocked
    """

    tasks: dict[str, Task] = Field(default_factory=dict)
    """All tasks keyed by task ID."""

    _next_id: int = 1
    """Counter for generating sequential task IDs."""

    def __init__(self, **data: Any) -> None:
        """Initialize TaskManager."""
        super().__init__(**data)
        # Sync _next_id with existing tasks
        if self.tasks:
            max_id = max(int(task_id) for task_id in self.tasks if task_id.isdigit())
            object.__setattr__(self, "_next_id", max_id + 1)

    def _generate_id(self) -> str:
        """Generate next task ID."""
        task_id = str(self._next_id)
        object.__setattr__(self, "_next_id", self._next_id + 1)
        return task_id

    def create(
        self,
        subject: str,
        description: str,
        active_form: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        """Create a new task.

        Args:
            subject: Task title in imperative form.
            description: Detailed task description.
            active_form: Present progressive form for in_progress status.
            metadata: Optional additional metadata.

        Returns:
            The created Task instance.
        """
        task_id = self._generate_id()
        now = datetime.now()
        task = Task(
            id=task_id,
            subject=subject,
            description=description,
            active_form=active_form,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
        self.tasks[task_id] = task
        return task

    def get(self, task_id: str) -> Task | None:
        """Get a task by ID.

        Args:
            task_id: The task ID to look up.

        Returns:
            The Task if found, None otherwise.
        """
        return self.tasks.get(task_id)

    def _add_blocking_relationship(self, task_id: str, blocked_id: str) -> None:
        """Add a blocking relationship: task_id blocks blocked_id."""
        task = self.tasks.get(task_id)
        blocked_task = self.tasks.get(blocked_id)
        if task and blocked_id not in task.blocks:
            task.blocks.append(blocked_id)
        if blocked_task and task_id not in blocked_task.blocked_by:
            blocked_task.blocked_by.append(task_id)
            blocked_task.updated_at = datetime.now()

    def _add_blocked_by_relationship(self, task_id: str, blocker_id: str) -> None:
        """Add a blocked-by relationship: task_id is blocked by blocker_id."""
        task = self.tasks.get(task_id)
        blocker_task = self.tasks.get(blocker_id)
        if task and blocker_id not in task.blocked_by:
            task.blocked_by.append(blocker_id)
        if blocker_task and task_id not in blocker_task.blocks:
            blocker_task.blocks.append(task_id)
            blocker_task.updated_at = datetime.now()

    def _resolve_completion(self, task: Task) -> None:
        """Remove completed task from blocked_by lists of tasks it blocks."""
        for blocked_id in task.blocks:
            blocked_task = self.tasks.get(blocked_id)
            if blocked_task and task.id in blocked_task.blocked_by:
                blocked_task.blocked_by.remove(task.id)
                blocked_task.updated_at = datetime.now()

    def _update_task_fields(
        self,
        task: Task,
        status: TaskStatus | None,
        subject: str | None,
        description: str | None,
        active_form: str | None,
        owner: str | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Update simple task fields."""
        if status is not None:
            task.status = status
        if subject is not None:
            task.subject = subject
        if description is not None:
            task.description = description
        if active_form is not None:
            task.active_form = active_form
        if owner is not None:
            task.owner = owner
        if metadata:
            task.metadata.update(metadata)

    def update(
        self,
        task_id: str,
        *,
        status: TaskStatus | None = None,
        subject: str | None = None,
        description: str | None = None,
        active_form: str | None = None,
        owner: str | None = None,
        add_blocks: list[str] | None = None,
        add_blocked_by: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task | None:
        """Update a task's properties.

        When status changes to COMPLETED, automatically removes this task
        from the blocked_by list of all tasks it was blocking.

        Args:
            task_id: The task ID to update.
            status: New task status.
            subject: New task subject.
            description: New task description.
            active_form: New active form text.
            owner: New task owner.
            add_blocks: Task IDs to add to blocks list.
            add_blocked_by: Task IDs to add to blocked_by list.
            metadata: Metadata to merge into existing metadata.

        Returns:
            The updated Task if found, None otherwise.
        """
        task = self.tasks.get(task_id)
        if task is None:
            return None

        # Track if status changed to completed for dependency resolution
        was_completed = status == TaskStatus.COMPLETED and task.status != TaskStatus.COMPLETED

        # Update simple fields
        self._update_task_fields(task, status, subject, description, active_form, owner, metadata)

        # Update relationships
        for blocked_id in add_blocks or []:
            self._add_blocking_relationship(task_id, blocked_id)
        for blocker_id in add_blocked_by or []:
            self._add_blocked_by_relationship(task_id, blocker_id)

        task.updated_at = datetime.now()

        # Handle completion: remove this task from blocked_by of tasks it blocks
        if was_completed:
            self._resolve_completion(task)

        return task

    def list_all(self) -> list[Task]:
        """Get all tasks sorted by ID.

        Returns:
            List of all tasks sorted by numeric ID.
        """
        return sorted(self.tasks.values(), key=lambda t: int(t.id) if t.id.isdigit() else 0)

    def export_tasks(self) -> dict[str, dict[str, Any]]:
        """Export tasks for serialization.

        Returns:
            Dict of task data keyed by task ID.
        """
        return {task_id: task.model_dump(mode="json") for task_id, task in self.tasks.items()}

    @classmethod
    def from_exported(cls, data: dict[str, dict[str, Any]]) -> TaskManager:
        """Restore TaskManager from exported data.

        Args:
            data: Exported task data from export_tasks().

        Returns:
            Restored TaskManager instance.
        """
        tasks = {task_id: Task.model_validate(task_data) for task_id, task_data in data.items()}
        return cls(tasks=tasks)


# =============================================================================
# Resumable State
# =============================================================================


class ResumableState(BaseModel):
    """Resumable session state for AgentContext.

    This model captures the session state that can be serialized to JSON and
    restored later. It handles the special serialization requirements of
    ModelMessage using ModelMessagesTypeAdapter.

    The subagent_history is stored as serialized dict format (list[dict]) rather
    than ModelMessage objects, making the entire model JSON-serializable.

    Example:
        Saving state to JSON file::

            state = ctx.export_state()
            with open("session.json", "w") as f:
                f.write(state.model_dump_json(indent=2))

        Restoring state from JSON file::

            with open("session.json") as f:
                state = ResumableState.model_validate_json(f.read())
            new_ctx.restore_state(state)
    """

    subagent_history: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    """Serialized subagent history, keyed by agent_id. Values are list[dict] from ModelMessagesTypeAdapter.dump_python()."""

    extra_usages: list[ExtraUsageRecord] = Field(default_factory=list)
    """Extra usage records from tool calls and filters."""

    user_prompts: str | Sequence[UserContent] | None = None
    """User prompts collected during the session."""

    handoff_message: str | None = None
    """Rendered handoff message."""

    deferred_tool_metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Metadata for deferred tool calls."""

    agent_registry: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Serialized agent registry for tracking agent metadata."""

    need_user_approve_tools: list[str] = Field(default_factory=list)
    """List of tool names that require user approval before execution."""

    need_user_approve_mcps: list[str] = Field(default_factory=list)
    """List of MCP server names that require user approval for all tools."""

    auto_load_files: list[str] = Field(default_factory=list)
    """Files to auto-load on next request. Set by handoff/compact, consumed by auto_load_files filter."""

    tasks: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Serialized tasks from TaskManager, keyed by task ID."""

    def to_subagent_history(self) -> dict[str, list[ModelMessage]]:
        """Deserialize subagent_history to ModelMessage objects.

        Returns:
            Dict mapping agent_id to list of ModelMessage objects.
        """
        result: dict[str, list[ModelMessage]] = {}
        for key, messages_data in self.subagent_history.items():
            result[key] = ModelMessagesTypeAdapter.validate_python(messages_data)
        return result

    def restore(self, ctx: AgentContext) -> None:
        """Restore state into an AgentContext.

        This method applies the saved state to the given context.
        Subclasses can override this method to restore additional fields.

        Args:
            ctx: The AgentContext to restore state into.

        Example::

            class MyState(ResumableState):
                custom_field: str = ""

                def restore(self, ctx: "MyContext") -> None:
                    super().restore(ctx)
                    ctx.custom_field = self.custom_field
        """
        ctx.subagent_history = self.to_subagent_history()
        ctx.extra_usages = list(self.extra_usages)
        ctx.user_prompts = self.user_prompts
        ctx.handoff_message = self.handoff_message
        ctx.deferred_tool_metadata = dict(self.deferred_tool_metadata)
        # Restore agent_registry from serialized format
        ctx.agent_registry = {agent_id: AgentInfo(**info) for agent_id, info in self.agent_registry.items()}
        ctx.need_user_approve_tools = list(self.need_user_approve_tools)
        ctx.need_user_approve_mcps = list(self.need_user_approve_mcps)
        ctx.auto_load_files = list(self.auto_load_files)
        # Restore task_manager from serialized tasks
        if self.tasks:
            ctx.task_manager = TaskManager.from_exported(self.tasks)


class ToolIdWrapper:
    """Wrapper for tool call IDs to ensure stable, cross-provider compatible identifiers.

    This class normalizes tool call IDs from different LLM providers by mapping them to
    a consistent format ("pai-{uuid}"). This ensures:
    - Consistent ID format across all providers (OpenAI, Anthropic, Gemini, etc.)
    - Stable ID mapping within a session for proper tool call/result matching
    - Compatibility with all downstream systems expecting standardized IDs

    The wrapper maintains an internal mapping to ensure the same original ID always
    maps to the same normalized ID within a session.
    """

    def __init__(self) -> None:
        self._prefix = "pai-"
        self._tool_call_maps: dict[str, str] = {}

    def clear(self) -> None:
        """Clear the tool call ID mapping.

        This should be called when starting a new session to ensure
        fresh ID mappings.
        """
        self._tool_call_maps.clear()

    def upsert_tool_call_id(self, tool_call_id: str) -> str:
        """Normalize a tool call ID to the standard format.

        If the ID already has the standard prefix, return it unchanged.
        Otherwise, create and cache a new normalized ID.

        Args:
            tool_call_id: The original tool call ID from any provider.

        Returns:
            Normalized tool call ID with "pai-" prefix.
        """
        if tool_call_id.startswith(self._prefix):
            return tool_call_id

        if tool_call_id not in self._tool_call_maps:
            self._tool_call_maps[tool_call_id] = f"{self._prefix}{uuid4().hex}"
        return self._tool_call_maps[tool_call_id]

    def wrap_event(self, event: AgentStreamEvent) -> AgentStreamEvent:
        match event:
            case FunctionToolCallEvent():
                event.part.tool_call_id = self.upsert_tool_call_id(event.tool_call_id)
            case FunctionToolResultEvent():
                event.result.tool_call_id = self.upsert_tool_call_id(event.tool_call_id)
            case PartStartEvent() | PartEndEvent():
                if isinstance(event.part, (ToolCallPart, ToolReturnPart, RetryPromptPart)):
                    event.part.tool_call_id = self.upsert_tool_call_id(event.part.tool_call_id)
            case PartDeltaEvent():
                if isinstance(event.delta, ToolCallPartDelta) and event.delta.tool_call_id:
                    event.delta.tool_call_id = self.upsert_tool_call_id(event.delta.tool_call_id)
        return event

    def wrap_deferred_tool_requests(self, deferred_tool_requests: DeferredToolRequests) -> DeferredToolRequests:
        for call in deferred_tool_requests.calls or []:
            call.tool_call_id = self.upsert_tool_call_id(call.tool_call_id)
        for approval in deferred_tool_requests.approvals or []:
            approval.tool_call_id = self.upsert_tool_call_id(approval.tool_call_id)
        return deferred_tool_requests

    def wrap_messages(
        self,
        _: RunContext[AgentContext],
        message_history: list[ModelMessage],
    ) -> list[ModelMessage]:
        """Normalize all tool call IDs in the message history.

        This method can be used directly as a pydantic-ai history_processor.

        Args:
            _: RunContext (unused, required by history_processor signature).
            message_history: List of messages to process.

        Returns:
            The same message history with normalized tool call IDs.
        """
        for m in message_history:
            for p in m.parts:
                if isinstance(p, (ToolCallPart, ToolReturnPart, RetryPromptPart)):
                    p.tool_call_id = self.upsert_tool_call_id(p.tool_call_id)
        return message_history


# =============================================================================
# Subagent Stream Event Types
# =============================================================================

# Stream event type for the queue
# Includes pydantic-ai events + custom AgentEvent for user-defined events
AgentStreamEvent = ModelResponseStreamEvent | PydanticHandleResponseEvent | AgentEvent


def _create_stream_queue_factory() -> dict[str, asyncio.Queue[AgentStreamEvent]]:
    """Create a defaultdict factory for subagent stream queues."""
    return defaultdict(asyncio.Queue)


# =============================================================================
# Agent Info and Stream Event
# =============================================================================


@dataclass
class AgentInfo:
    """Metadata for a registered agent.

    Used to track agent identity and hierarchy in stream events.

    Attributes:
        agent_id: Unique identifier for the agent (e.g., "main" or 4-char short ID).
        agent_name: Human-readable name (e.g., "main", "search", "reasoning").
        parent_agent_id: ID of the parent agent, None for main agent.
    """

    agent_id: str
    agent_name: str
    parent_agent_id: str | None = None


@dataclass
class StreamEvent:
    """Stream event with agent identification.

    Wraps raw stream events with agent metadata for distinguishing
    events from different agents in a merged stream.

    Attributes:
        agent_id: ID of the agent that produced this event.
        agent_name: Name of the agent.
        event: The underlying stream event (ModelResponseStreamEvent, etc.).
    """

    agent_id: str
    agent_name: str
    event: AgentStreamEvent


# =============================================================================
# Model Capability
# =============================================================================


class ModelCapability(str, Enum):
    """Model capabilities that can be used to describe what a model supports."""

    vision = "vision"
    """Model can process and understand images."""

    video_understanding = "video_understanding"
    """Model can process and understand video content."""

    document_understanding = "document_understanding"
    """Model can process and understand documents (PDF, etc.)."""


def _generate_run_id() -> str:
    return uuid4().hex


def _xml_to_string(element: Element) -> str:
    """Convert XML element to formatted string."""
    rough_string = tostring(element, encoding="unicode")
    dom = parseString(rough_string)  # noqa: S318
    # Get pretty-printed XML, skip the XML declaration line
    lines = dom.toprettyxml(indent="  ").split("\n")[1:]
    # Remove empty lines
    return "\n".join(line for line in lines if line.strip())


class ToolSettings(BaseSettings):
    """Tool-related settings from environment variables.

    API keys for various external services used by tools.
    All settings are loaded from environment variables or .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Web search API keys
    google_search_api_key: str | None = None
    """Google Custom Search API key."""

    google_search_cx: str | None = None
    """Google Custom Search Engine ID."""

    tavily_api_key: str | None = None
    """Tavily API key for web search."""

    # Image search API keys
    pixabay_api_key: str | None = None
    """Pixabay API key for stock image search."""

    rapidapi_api_key: str | None = None
    """RapidAPI key for real-time image search."""

    # Web scraping API key
    firecrawl_api_key: str | None = None
    """Firecrawl API key for web scraping."""


def _get_tool_settings() -> ToolSettings:
    """Get ToolSettings instance.

    This function creates a new ToolSettings instance each time to ensure
    environment variables are read at the time of ToolConfig creation,
    not at module import time.
    """
    return ToolSettings()


class ToolConfig(BaseModel):
    """Tool-level configuration for fine-grained control.

    API keys can be passed directly or loaded from environment variables
    via ToolSettings. See .env.example for available environment variables.

    Note: Environment variables are read when ToolConfig is instantiated,
    not when the module is imported. This allows callers to set environment
    variables after importing the module.

    Extensibility:
        This class supports two extension patterns:

        1. Inheritance (recommended for type safety)::

            class MyToolConfig(ToolConfig):
                my_api_key: str | None = None

            class MyContext(AgentContext):
                tool_config: MyToolConfig = Field(default_factory=MyToolConfig)

        2. Extra attributes (for quick prototyping)::

            config = ToolConfig(my_custom_key="value")
            config.my_custom_key  # Accessible but not type-checked
    """

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    skip_url_verification: bool = True
    """Skip SSRF URL verification. Default enabled (skip). Set to False to enable verification for public-facing environments."""

    enable_load_document: bool = False
    """Enable document URL parsing in LoadTool. Default disabled due to poor model support."""

    image_understanding_model: str | None = None
    """Model to use for image understanding. Falls back to AgentSettings.image_understanding_model."""

    image_understanding_model_settings: ModelSettings | None = None
    """Model settings for image understanding agent."""

    video_understanding_model: str | None = None
    """Model to use for video understanding. Falls back to AgentSettings.video_understanding_model."""

    video_understanding_model_settings: ModelSettings | None = None
    """Model settings for video understanding agent."""

    # Web search API keys
    google_search_api_key: str | None = Field(default_factory=lambda: _get_tool_settings().google_search_api_key)
    """Google Custom Search API key."""

    google_search_cx: str | None = Field(default_factory=lambda: _get_tool_settings().google_search_cx)
    """Google Custom Search Engine ID."""

    tavily_api_key: str | None = Field(default_factory=lambda: _get_tool_settings().tavily_api_key)
    """Tavily API key for web search."""

    # Image search API keys
    pixabay_api_key: str | None = Field(default_factory=lambda: _get_tool_settings().pixabay_api_key)
    """Pixabay API key for stock image search."""

    rapidapi_api_key: str | None = Field(default_factory=lambda: _get_tool_settings().rapidapi_api_key)
    """RapidAPI key for real-time image search."""

    # Web scraping API key
    firecrawl_api_key: str | None = Field(default_factory=lambda: _get_tool_settings().firecrawl_api_key)
    """Firecrawl API key for web scraping."""

    # Media to URL conversion hooks
    image_to_url_hook: MediaToUrlHook | None = None
    """Hook to convert image data to URL.

    Args:
        ctx: RunContext with AgentContext
        image_data: Raw image bytes
        media_type: MIME type (e.g., 'image/png')

    Returns:
        Publicly accessible URL string, or None to use default BinaryContent behavior.
        Can be sync or async function.

    Note:
        The returned URL must be publicly accessible by the LLM provider.
        Empty strings are treated as None (fallback to default behavior).
    """

    video_to_url_hook: MediaToUrlHook | None = None
    """Hook to convert video data to URL.

    Args:
        ctx: RunContext with AgentContext
        video_data: Raw video bytes
        media_type: MIME type (e.g., 'video/mp4')

    Returns:
        Publicly accessible URL string, or None to use default BinaryContent behavior.
        Can be sync or async function.

    Note:
        The returned URL must be publicly accessible by the LLM provider.
        Empty strings are treated as None (fallback to default behavior).
    """


class ModelConfig(BaseModel):
    """Model configuration for context management.

    Extensibility:
        This class supports two extension patterns:

        1. Inheritance (recommended for type safety)::

            class MyModelConfig(ModelConfig):
                custom_threshold: float = 0.8

            class MyContext(AgentContext):
                model_cfg: MyModelConfig = Field(default_factory=MyModelConfig)

        2. Extra attributes (for quick prototyping)::

            config = ModelConfig(custom_field="value")
            config.custom_field  # Accessible but not type-checked
    """

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    context_window: int | None = None
    """Total context window size in tokens."""

    proactive_context_management_threshold: float | None = 0.5
    """Proactive context management threshold. When token usage exceeds this ratio, reminders are triggered."""

    compact_threshold: float = 0.90
    """Compact threshold for auto-compaction. When token usage exceeds this ratio, compact is triggered."""

    max_images: int = 20
    """Maximum number of images allowed in message history. Default is 20 (Claude's limit)."""

    max_videos: int = 1
    """Maximum number of videos allowed in message history. Default is 1."""

    support_gif: bool = True
    """Whether the model supports GIF images. If False, GIF images will be filtered out."""

    capabilities: set[ModelCapability] = Field(default_factory=set)
    """Set of capabilities supported by the model."""

    def has_capability(self, capability: ModelCapability) -> bool:
        """Check if the model has a specific capability."""
        return capability in self.capabilities

    @property
    def has_vision(self) -> bool:
        """Check if the model supports vision (image understanding)."""
        return ModelCapability.vision in self.capabilities

    @property
    def has_video_understanding(self) -> bool:
        """Check if the model supports video understanding."""
        return ModelCapability.video_understanding in self.capabilities

    @property
    def has_document_understanding(self) -> bool:
        """Check if the model supports document understanding."""
        return ModelCapability.document_understanding in self.capabilities


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
                        env=env,
                        model_cfg=ModelConfig(
                            context_window=200000,
                            proactive_context_management_threshold=0.5,
                        ),
                    )
                )
                toolset = Toolset(tools=[HandoffTool])
                agent = Agent(
                    'openai:gpt-4',
                    deps_type=AgentContext,
                    toolsets=[toolset],
                    history_processors=[process_handoff_message],
                    # Set context management tool name - triggers threshold warning
                    metadata=lambda _: {'context_manage_tool': 'handoff'},
                )
                result = await agent.run('Your prompt here', deps=ctx)
    """

    context_manage_tool: str
    """Name of the context management tool to use (e.g., 'handoff')."""


class AgentContext(BaseModel):
    """Context for a single agent session.

    AgentContext manages session-specific state including:
    - Run identification (run_id, parent_run_id)
    - Timing (start_at, end_at, elapsed_time)
    - Deferred tool metadata
    - Handoff messages

    AgentContext holds a reference to an Environment and derives
    file_operator, shell, and resources from it via properties.

    Example:
        Using create_agent and stream_agent (recommended)::

            from pai_agent_sdk.agents.main import create_agent, stream_agent

            runtime = create_agent("openai:gpt-4")
            # stream_agent manages runtime lifecycle automatically
            async with stream_agent(runtime, "Hello") as streamer:
                async for event in streamer:
                    print(event)

        Using create_agent with manual agent.run::

            runtime = create_agent("openai:gpt-4")
            async with runtime:
                result = await runtime.agent.run("Hello", deps=runtime.ctx)

        Manual setup with Environment (advanced)::

            async with LocalEnvironment() as env:
                async with AgentContext(env=env) as ctx:
                    await ctx.file_operator.read_file("data.json")
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

    env: Environment | None = None
    """Environment instance. file_operator/shell/resources are derived from it."""

    model_cfg: ModelConfig = Field(default_factory=ModelConfig)
    """Model configuration for context management."""

    tool_config: ToolConfig = Field(default_factory=ToolConfig)
    """Tool-level configuration for API keys and tool-specific settings."""

    extra_usages: list[ExtraUsageRecord] = Field(default_factory=list)
    """Extra usage records from tool calls and filters."""

    user_prompts: str | Sequence[UserContent] | None = None
    """User prompts collected during the session for compact."""

    tool_id_wrapper: ToolIdWrapper = Field(default_factory=ToolIdWrapper)
    """Tool ID wrapper for normalizing tool call IDs across providers."""

    agent_stream_queues: dict[str, asyncio.Queue[AgentStreamEvent]] = Field(
        default_factory=_create_stream_queue_factory
    )
    """Stream queues for agent events, keyed by run_id(tool_call_id).

    Each queue receives AgentStreamEvent instances during agent execution,
    enabling real-time streaming of agent responses via a sideband channel.
    """

    need_user_approve_tools: list[str] = Field(default_factory=list)
    """List of tool names that require user approval before execution.

    Tools in this list will trigger HITL (Human-in-the-Loop) flow,
    deferring execution until the user explicitly approves.
    """

    need_user_approve_mcps: list[str] = Field(default_factory=list)
    """List of MCP server names that require user approval for all tools.

    When a server name is in this list, all tools from that server will
    trigger the HITL approval flow before execution. The server name
    corresponds to the tool_prefix used when creating the MCPServer.

    Example:
        ctx.need_user_approve_mcps = ["filesystem", "github"]
        # All tools from these servers will require approval
    """

    subagent_history: dict[str, list[ModelMessage]] = Field(default_factory=dict)
    """Subagent history for resuming sessions."""

    agent_registry: dict[str, AgentInfo] = Field(default_factory=dict)
    """Registry of agent metadata, keyed by agent_id.

    Used by stream_agent to track agent identity and hierarchy.
    Populated by enter_subagent when subagents are created.
    """

    auto_load_files: list[str] = Field(default_factory=list)
    """Files to auto-load on next request. Set by handoff/compact tool, consumed by auto_load_files filter."""

    task_manager: TaskManager = Field(default_factory=TaskManager)
    """Task manager for tracking tasks and dependencies within the session."""

    _agent_id: str = "main"
    _entered: bool = False
    _enter_lock: asyncio.Lock = None  # type: ignore[assignment]  # Initialized in __init__
    _stream_queue_enabled: bool = False

    def __init__(self, **data: Any) -> None:
        """Initialize AgentContext."""
        super().__init__(**data)
        object.__setattr__(self, "_enter_lock", asyncio.Lock())

    @property
    def file_operator(self) -> FileOperator | None:
        """File operator for file system operations. Derived from env."""
        if self.env is not None:
            return self.env.file_operator
        return None

    @property
    def shell(self) -> Shell | None:
        """Shell executor for command execution. Derived from env."""
        if self.env is not None:
            return self.env.shell
        return None

    @property
    def resources(self) -> ResourceRegistry | None:
        """Resource registry for runtime resources. Derived from env."""
        if self.env is not None:
            return self.env.resources
        return None

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

    def get_current_time(self) -> datetime:
        """Return current time with timezone information.

        Override this method to provide custom time sources (e.g., NTP, mock for testing).
        Default implementation uses system local time with timezone offset.

        Returns:
            Current datetime with timezone information (ISO 8601 compatible).

        Example:
            Subclass to customize time source::

                class MyContext(AgentContext):
                    def get_current_time(self) -> datetime:
                        return ntp_client.get_time()  # Custom NTP source

            Mock for testing::

                class MockContext(AgentContext):
                    def get_current_time(self) -> datetime:
                        return datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        """
        return datetime.now().astimezone()

    def _build_active_tasks_element(self, parent: Element, detailed: bool) -> None:
        """Build active-tasks XML element and append to parent if tasks exist.

        Args:
            parent: Parent XML element to append to.
            detailed: If True, use detailed format with hints and active_form;
                otherwise use compact format.
        """
        active_tasks = [t for t in self.task_manager.list_all() if t.status != TaskStatus.COMPLETED]
        if not active_tasks:
            return

        tasks_elem = SubElement(parent, "active-tasks")
        if detailed:
            tasks_elem.set("hint", "Update status with task_update tool")

        for task in active_tasks:
            task_elem = SubElement(tasks_elem, "task")
            task_elem.set("id", task.id)
            task_elem.set("status", task.status.value)

            # Only show active (incomplete) blockers
            active_blockers = [
                bid
                for bid in task.blocked_by
                if (blocker := self.task_manager.get(bid)) and blocker.status != TaskStatus.COMPLETED
            ]
            if active_blockers:
                task_elem.set("blocked-by", ",".join(active_blockers))

            if detailed:
                # Detailed format: subject as sub-element, include active_form
                SubElement(task_elem, "subject").text = task.subject
                if task.active_form and task.status == TaskStatus.IN_PROGRESS:
                    SubElement(task_elem, "active-form").text = task.active_form
            else:
                # Compact format: subject as text content
                task_elem.text = task.subject

    async def get_context_instructions(
        self,
        run_context: RunContext[AgentContext] | None = None,
        *,
        is_user_prompt: bool = True,
    ) -> str:
        """Return runtime context instructions in XML format.

        Provides runtime information about the current session.

        Args:
            run_context: Optional RunContext for accessing message history and metadata.
            is_user_prompt: Whether this is a user prompt (True) or a tool response/retry (False).
                Some information (e.g., known subagents) is only included on user prompts
                to reduce noise. Subclasses can override to customize behavior. Default True.

        Returns:
            XML-formatted string with runtime context and optional system reminders.
        """
        parts: list[str] = []

        # Build runtime-context element
        root = Element("runtime-context")

        # Agent ID (useful for subagents to know their identity)
        SubElement(root, "agent-id").text = self._agent_id

        # Current time (ISO 8601 with timezone offset for cross-environment compatibility)
        SubElement(root, "current-time").text = self.get_current_time().isoformat(timespec="seconds")

        # Elapsed time
        elapsed = self.elapsed_time
        elapsed_str = f"{elapsed.total_seconds():.1f}s" if elapsed else "not started"
        SubElement(root, "elapsed-time").text = elapsed_str

        # Model configuration - only context_window
        if self.model_cfg.context_window is not None:
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

        # Known subagents from agent_registry (excluding main agent)
        # Only include on user prompts, not tool responses
        if is_user_prompt:
            known_subagents = {
                agent_id: info for agent_id, info in self.agent_registry.items() if agent_id != self.run_id
            }
            if known_subagents:
                subagents_elem = SubElement(root, "known-subagents")
                subagents_elem.set("hint", "Use subagent_info tool for more details")
                for _agent_id, info in known_subagents.items():
                    agent_elem = SubElement(subagents_elem, "agent")
                    agent_elem.set("id", info.agent_id)
                    agent_elem.set("name", info.agent_name)

        # Active tasks (pending + in_progress)
        self._build_active_tasks_element(root, detailed=is_user_prompt)

        parts.append(_xml_to_string(root))

        # Build system-reminder element (sibling to runtime-context)
        reminders: list[str] = []

        # Cast metadata to typed dict for type safety
        metadata = cast(
            RunContextMetadata,
            run_context.metadata if run_context and run_context.metadata else {},
        )

        # Handoff threshold warning
        if (
            (context_manage_tool := metadata.get("context_manage_tool"))
            and self.model_cfg.context_window is not None
            and self.model_cfg.proactive_context_management_threshold is not None
            and run_context
            and (request_usage := get_latest_request_usage(run_context.messages))
        ):
            threshold_tokens = int(
                self.model_cfg.context_window * self.model_cfg.proactive_context_management_threshold
            )
            if request_usage.total_tokens >= threshold_tokens:
                reminders.append(
                    f"IMPORTANT: **You have reached the handoff threshold, please calling the `{context_manage_tool}` tool "
                    "to summarize then continue the task at the appropriate time.**"
                )

        if reminders:
            reminder_root = Element("system-reminder")
            for reminder_text in reminders:
                item = SubElement(reminder_root, "item")
                item.text = reminder_text
            parts.append(_xml_to_string(reminder_root))

        return "\n\n".join(parts)

    def create_subagent_context(
        self,
        agent_name: str,
        agent_id: str | None = None,
        **override: Any,
    ) -> Self:
        """Create a child context for subagent with independent timing.

        The subagent context inherits all fields but gets:
        - A new run_id (uses agent_id if provided, otherwise generates one)
        - parent_run_id set to current run_id
        - Fresh timing (start_at/end_at managed by __aenter__/__aexit__)
        - Shared file_operator and shell from parent
        - Registers agent info in parent's agent_registry

        Note:
            The returned context should be used with `async with` to properly
            manage timing and resource cleanup::

                sub_ctx = parent.create_subagent_context("search")
                async with sub_ctx:
                    # subagent work here

        Args:
            agent_name: Name of the subagent (e.g., "search", "reasoning").
            agent_id: ID for the subagent. If None, generates a unique ID.
                Can be tool_call_id for correlation with tool calls.
            **override: Additional fields to override in the subagent context.
                Subclasses can pass extra fields without overriding this method.

        Returns:
            A new context instance configured for the subagent.
        """
        # Generate agent_id if not provided
        effective_agent_id = agent_id or _generate_run_id()

        # Register agent info in parent's registry (idempotent for resume)
        if effective_agent_id not in self.agent_registry:
            self.agent_registry[effective_agent_id] = AgentInfo(
                agent_id=effective_agent_id,
                agent_name=agent_name,
                parent_agent_id=self._agent_id,
            )

        update: dict[str, Any] = {
            "run_id": _generate_run_id(),
            "parent_run_id": self.run_id,
            "start_at": None,  # Will be set by __aenter__
            "end_at": None,  # Will be set by __aexit__
            "handoff_message": None,  # Subagents don't inherit handoff state
            "tool_id_wrapper": ToolIdWrapper(),  # Fresh wrapper for subagent
            # env is inherited via model_copy (shares parent's env reference)
            **override,
        }
        new_ctx = self.model_copy(update=update)
        new_ctx._agent_id = effective_agent_id
        # Reset re-entry protection for subagent (independent lifecycle)
        object.__setattr__(new_ctx, "_entered", False)
        object.__setattr__(new_ctx, "_enter_lock", asyncio.Lock())
        return new_ctx

    def get_history_processors(self) -> list:
        """Return a list of history processors for this context.

        Returns a list containing the tool_id_wrapper.wrap_messages method
        which can be used directly with pydantic-ai's history_processors parameter.

        Returns:
            List of history processor functions.

        Example::

            async with AgentContext(...) as ctx:
                agent = Agent(
                    'openai:gpt-4',
                    deps_type=AgentContext,
                    history_processors=ctx.get_history_processors(),
                )
        """
        # Import filters here to avoid circular imports
        from pai_agent_sdk.filters.auto_load_files import process_auto_load_files
        from pai_agent_sdk.filters.capability import filter_by_capability
        from pai_agent_sdk.filters.handoff import process_handoff_message
        from pai_agent_sdk.filters.image import drop_extra_images, drop_extra_videos, drop_gif_images
        from pai_agent_sdk.filters.runtime_instructions import inject_runtime_instructions
        from pai_agent_sdk.filters.tool_args import fix_truncated_tool_args

        def dynamic_tool_id_wrapper(ctx: RunContext[AgentContext], messages: list[ModelMessage]) -> list[ModelMessage]:
            """Dynamically get tool_id_wrapper from current context."""
            return ctx.deps.tool_id_wrapper.wrap_messages(ctx, messages)

        return [
            # handle_model_switch, # Disabled as response.model_name is not the same as ctx.model.model_name
            drop_extra_images,
            drop_gif_images,
            drop_extra_videos,
            fix_truncated_tool_args,
            process_handoff_message,
            process_auto_load_files,
            filter_by_capability,
            inject_runtime_instructions,
            dynamic_tool_id_wrapper,
        ]

    def add_extra_usage(
        self,
        agent: str,
        internal_usage: InternalUsage,
        uuid: str | None = None,
    ) -> None:
        """Add an extra usage record.

        Args:
            agent: Agent name that generated this usage.
            internal_usage: Internal usage record containing model_id and token usage.
            uuid: Unique identifier (defaults to generated UUID if not provided).
        """
        record_uuid = uuid or uuid4().hex
        self.extra_usages.append(
            ExtraUsageRecord(
                uuid=record_uuid,
                agent=agent,
                model_id=internal_usage.model_id,
                usage=internal_usage.usage,
            )
        )

    async def emit_event(self, event: AgentStreamEvent) -> None:
        """Emit a custom event to the sideband stream queue.

        Events are placed in the agent_stream_queues under the current run_id,
        allowing consumers to receive custom notifications alongside pydantic-ai
        stream events.

        This method is a no-op if streaming is not enabled. Streaming is
        automatically enabled when using stream_agent().

        Args:
            event: Any event object (AgentEvent subclass, pydantic-ai events, or custom).

        Example::

            from pai_agent_sdk.events import CompactStartEvent

            await ctx.emit_event(CompactStartEvent(event_id="abc123", message_count=50))
        """
        if not self._stream_queue_enabled:
            return
        await self.agent_stream_queues[self.run_id].put(event)

    async def __aenter__(self):
        """Enter the context and start timing.

        Raises:
            RuntimeError: If the context has already been entered.
        """
        async with self._enter_lock:
            if self._entered:
                raise RuntimeError(
                    "AgentContext has already been entered. "
                    "Each AgentContext instance can only be entered once at a time."
                )
            self._entered = True
        self.start_at = datetime.now()
        self.tool_id_wrapper.clear()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and record end time."""
        self.end_at = datetime.now()
        async with self._enter_lock:
            self._entered = False

    def export_state(self, *, include_subagent: bool = True) -> ResumableState:
        """Export resumable session state.

        Creates a ResumableState containing all session data that can be
        serialized to JSON and restored later.

        Args:
            include_subagent: Whether to include subagent history and registry.
                Defaults to True. Set to False to exclude subagent data,
                which can reduce state size for main agent-only persistence.

        Returns:
            ResumableState instance ready for serialization.

        Example::

            # Save full state including subagent history
            state = ctx.export_state()

            # Save state without subagent data
            state = ctx.export_state(include_subagent=False)

            with open("session.json", "w") as f:
                f.write(state.model_dump_json(indent=2))
        """
        serialized_history: dict[str, list[dict[str, Any]]] = {}
        serialized_registry: dict[str, dict[str, Any]] = {}

        if include_subagent:
            # Serialize subagent_history using ModelMessagesTypeAdapter
            # Use mode='json' to ensure bytes (e.g., BinaryContent.data) are base64-encoded
            for key, messages in self.subagent_history.items():
                serialized_history[key] = ModelMessagesTypeAdapter.dump_python(messages, mode="json")

            # Serialize agent_registry to dict format
            serialized_registry = {
                agent_id: {
                    "agent_id": info.agent_id,
                    "agent_name": info.agent_name,
                    "parent_agent_id": info.parent_agent_id,
                }
                for agent_id, info in self.agent_registry.items()
            }

        return ResumableState(
            subagent_history=serialized_history,
            extra_usages=list(self.extra_usages),
            user_prompts=self.user_prompts,
            handoff_message=self.handoff_message,
            deferred_tool_metadata=dict(self.deferred_tool_metadata),
            agent_registry=serialized_registry,
            need_user_approve_tools=list(self.need_user_approve_tools),
            auto_load_files=list(self.auto_load_files),
            tasks=self.task_manager.export_tasks(),
        )

    def with_state(self, state: ResumableState | None) -> Self:
        """Restore session state from a ResumableState.

        Updates the context with state from a previously exported ResumableState.
        This allows resuming a session after serialization/deserialization.

        If state is None, returns self unchanged for convenient chaining.

        Args:
            state: ResumableState to restore from, or None to skip restoration.

        Returns:
            Self for method chaining.

        Example::\n
            # Load from JSON file and use with async context manager
            with open("session.json") as f:
                state = ResumableState.model_validate_json(f.read())
            async with AgentContext(...).with_state(state) as ctx:
                ...

            # Also works with None for conditional restoration
            async with AgentContext(...).with_state(maybe_state) as ctx:
                ...
        """
        if state is None:
            return self
        state.restore(self)
        return self
