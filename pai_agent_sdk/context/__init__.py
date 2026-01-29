"""Agent context management.

This module provides the AgentContext class and related components for managing
session state during agent execution.

Example:
    Using create_agent and stream_agent (recommended)::

        from pai_agent_sdk.agents.main import create_agent, stream_agent

        runtime = create_agent("openai:gpt-4")
        async with stream_agent(runtime, "Hello") as streamer:
            async for event in streamer:
                print(event)

    Manual Environment and AgentContext setup (advanced)::

        from pai_agent_sdk.environment.local import LocalEnvironment
        from pai_agent_sdk.context import AgentContext

        async with LocalEnvironment() as env:
            async with AgentContext(env=env) as ctx:
                await ctx.file_operator.read_file("test.txt")
"""

from pai_agent_sdk.usage import ExtraUsageRecord

from .agent import (
    AgentContext,
    AgentInfo,
    AgentStreamEvent,
    MediaToUrlHook,
    ModelCapability,
    ModelConfig,
    ModelWrapper,
    ResumableState,
    RunContextMetadata,
    StreamEvent,
    ToolConfig,
    ToolIdWrapper,
    ToolSettings,
)
from .bus import BusMessage, MessageBus
from .tasks import Task, TaskManager, TaskStatus

__all__ = [
    "AgentContext",
    "AgentInfo",
    "AgentStreamEvent",
    "BusMessage",
    "ExtraUsageRecord",
    "MediaToUrlHook",
    "MessageBus",
    "ModelCapability",
    "ModelConfig",
    "ModelWrapper",
    "ResumableState",
    "RunContextMetadata",
    "StreamEvent",
    "Task",
    "TaskManager",
    "TaskStatus",
    "ToolConfig",
    "ToolIdWrapper",
    "ToolSettings",
]
