"""Enhancement tools for agent capabilities.

Tools for thinking, task management, and other enhancements.
"""

from pai_agent_sdk.toolsets.core.base import BaseTool
from pai_agent_sdk.toolsets.core.enhance.task import (
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskUpdateTool,
)
from pai_agent_sdk.toolsets.core.enhance.thinking import ThinkingTool
from pai_agent_sdk.toolsets.core.enhance.todo import TodoItem, TodoReadTool, TodoWriteTool

thinking_tools: list[type[BaseTool]] = [ThinkingTool]
todo_tools: list[type[BaseTool]] = [TodoReadTool, TodoWriteTool]
task_tools: list[type[BaseTool]] = [
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskUpdateTool,
]

tools: list[type[BaseTool]] = [
    # ThinkingTool,  # Disable by default via interleaved thinking
    # TodoReadTool,
    # TodoWriteTool, # Prefer task tools over individual todo tools
    TaskCreateTool,
    TaskGetTool,
    TaskUpdateTool,
    TaskListTool,
]

__all__ = [
    "TaskCreateTool",
    "TaskGetTool",
    "TaskListTool",
    "TaskUpdateTool",
    "ThinkingTool",
    "TodoItem",
    "TodoReadTool",
    "TodoWriteTool",
    "tools",
]
