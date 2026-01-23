"""Filters for message history processing.

This module provides history processors for pydantic-ai agents.
"""

from pai_agent_sdk.filters.auto_load_files import process_auto_load_files
from pai_agent_sdk.filters.environment_instructions import create_environment_instructions_filter
from pai_agent_sdk.filters.handoff import process_handoff_message
from pai_agent_sdk.filters.image import drop_extra_images, drop_extra_videos, drop_gif_images
from pai_agent_sdk.filters.system_prompt import create_system_prompt_filter
from pai_agent_sdk.filters.tool_args import fix_truncated_tool_args

__all__ = [
    "create_environment_instructions_filter",
    "create_system_prompt_filter",
    "drop_extra_images",
    "drop_extra_videos",
    "drop_gif_images",
    "fix_truncated_tool_args",
    "process_auto_load_files",
    "process_handoff_message",
]
