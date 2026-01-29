"""Video reading tool for models without native video understanding support.

This tool allows processing videos when the model does not support
native video understanding capabilities.
"""

from typing import Annotated

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

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return not ctx.deps.model_cfg.has_video_understanding

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
        if agent_ctx.tool_config:
            tool_config = agent_ctx.tool_config
            model = tool_config.video_understanding_model
            model_settings = tool_config.video_understanding_model_settings

        description, internal_usage = await get_video_description(
            video_url=url,
            model=model,
            model_settings=model_settings,
            model_wrapper=agent_ctx.model_wrapper,
            wrapper_context=agent_ctx.get_wrapper_context(),
        )

        # Store usage in extra_usages with tool_call_id
        if ctx.tool_call_id:
            agent_ctx.add_extra_usage(agent="video_understanding", internal_usage=internal_usage, uuid=ctx.tool_call_id)

        return description
