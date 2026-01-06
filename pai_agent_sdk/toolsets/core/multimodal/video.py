"""Video reading tool for models without native video understanding support.

This tool allows processing videos when the model does not support
native video understanding capabilities.
"""

from typing import Annotated
from uuid import uuid4

from pydantic import Field
from pydantic_ai import RunContext

from pai_agent_sdk.agents.video_understanding import get_video_description
from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.base import BaseTool


class ReadVideoTool(BaseTool):
    """Tool for reading and analyzing videos.

    Use this tool when the model does not support native video understanding
    but needs to process video content from URLs.
    """

    name = "read_video"
    description = "Read and analyze a video from a URL. Use when native video understanding is unavailable."

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """No instruction needed for this tool."""
        return None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        url: Annotated[
            str,
            Field(description="The URL of the video to read and analyze."),
        ],
    ) -> str:
        """Read and analyze a video from a URL.

        Args:
            ctx: The run context containing the agent context.
            url: The URL of the video to analyze.

        Returns:
            A text description or analysis of the video content in XML format.
        """
        agent_ctx = ctx.deps

        # Get model and settings from tool_config if available
        model = None
        model_settings = None
        if agent_ctx.model_cfg and agent_ctx.model_cfg.tool_config:
            tool_config = agent_ctx.model_cfg.tool_config
            model = tool_config.video_understanding_model
            model_settings = tool_config.video_understanding_model_settings

        description, usage = await get_video_description(
            video_url=url,
            model=model,
            model_settings=model_settings,
        )

        if uuid := ctx.tool_call_id or uuid4().hex:
            agent_ctx.extra_usage[uuid] = usage

        return description
