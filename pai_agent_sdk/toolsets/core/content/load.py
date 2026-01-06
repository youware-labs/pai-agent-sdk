"""Load tool for loading content from HTTP/HTTPS URLs.

This tool allows the agent to load content from web URLs and returns
appropriate content types based on model capabilities.

Note:
    Only HTTP and HTTPS URLs are supported. Other protocols will be rejected.

Example::

    from pai_agent_sdk.context import AgentContext, ModelCapability, ModelConfig
    from pai_agent_sdk.toolsets.core.base import Toolset
    from pai_agent_sdk.toolsets.core.content.load import LoadTool

    async with AgentContext(
        file_operator=env.file_operator,
        shell=env.shell,
        model_cfg=ModelConfig(
            capabilities={ModelCapability.vision},
        ),
    ) as ctx:
        toolset = Toolset(ctx, tools=[LoadTool])
        # Agent can now load URLs and get appropriate content types
"""

from pathlib import Path
from typing import Annotated

from jinja2 import Template
from pydantic import Field
from pydantic_ai import DocumentUrl, ImageUrl, RunContext, VideoUrl
from pydantic_ai.messages import AudioUrl

from pai_agent_sdk.context import AgentContext, ModelCapability
from pai_agent_sdk.toolsets.core.base import BaseTool
from pai_agent_sdk.toolsets.core.content._url_helper import (
    ContentCategory,
    detect_content_category,
    is_valid_http_url,
)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_instruction_template() -> Template:
    """Load load instruction template from prompts/load.md."""
    prompt_file = _PROMPTS_DIR / "load.md"
    return Template(prompt_file.read_text())


_INSTRUCTION_TEMPLATE = _load_instruction_template()


class LoadTool(BaseTool):
    """Tool for loading content from HTTP/HTTPS URLs.

    This tool validates that the URL uses HTTP or HTTPS protocol,
    detects the content type, and returns an appropriate URL type
    based on the model's capabilities.
    """

    name = "load"
    description = "Load content from HTTP/HTTPS URL (images, videos, audio, text, documents)."

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str:
        """Generate dynamic instruction based on tool config."""
        enable_load_document = ctx.deps.model_cfg is not None and ctx.deps.model_cfg.tool_config.enable_load_document
        return _INSTRUCTION_TEMPLATE.render(enable_load_document=enable_load_document)

    async def call(  # noqa: C901
        self,
        ctx: RunContext[AgentContext],
        url: Annotated[
            str,
            Field(description="The HTTP or HTTPS URL to load content from."),
        ],
    ) -> str | ImageUrl | VideoUrl | AudioUrl | DocumentUrl:
        """Load content from a URL.

        Args:
            ctx: The run context containing the agent context.
            url: The URL to load content from.

        Returns:
            The content in an appropriate format based on content type and model capabilities.
            Returns a text message if the model lacks required capabilities or URL is invalid.
        """
        # Validate URL protocol
        if not is_valid_http_url(url):
            return (
                f"Error: Only HTTP and HTTPS URLs are supported. The provided URL '{url}' uses an unsupported protocol."
            )

        # Detect content type
        category = await detect_content_category(url)

        # Get model capabilities
        model_cfg = ctx.deps.model_cfg
        has_vision = model_cfg is not None and model_cfg.has_capability(ModelCapability.vision)
        has_video = model_cfg is not None and model_cfg.has_capability(ModelCapability.video_understanding)
        has_document = model_cfg is not None and model_cfg.has_capability(ModelCapability.document_understanding)
        enable_load_document = model_cfg is not None and model_cfg.tool_config.enable_load_document

        # Return appropriate content type based on category and capabilities
        if category == ContentCategory.image:
            if not has_vision:
                return (
                    f"The URL '{url}' points to an image, but the current model does not support vision capability. "
                    "Use the `read_image` tool instead to analyze this image."
                )
            return ImageUrl(url=url)

        if category == ContentCategory.video:
            if not has_video:
                return (
                    f"The URL '{url}' points to a video, but the current model does not support video understanding. "
                    "Use the `read_video` tool instead to analyze this video."
                )
            return VideoUrl(url=url)

        if category == ContentCategory.audio:
            # Audio doesn't require special capability check for now
            return AudioUrl(url=url)

        if category == ContentCategory.document:
            if not enable_load_document:
                return (
                    f"Document parsing disabled for URL '{url}'. "
                    "Use download tool to save locally, then use pdf_convert."
                )
            if not has_document:
                return (
                    f"The URL '{url}' points to a document, but the current model does not support "
                    "document understanding. Cannot display document content."
                )
            return DocumentUrl(url=url)

        # For text or unknown content, return as document URL
        # The model can then process the text content
        if category == ContentCategory.text or category == ContentCategory.unknown:
            # Text content doesn't require special capability
            return DocumentUrl(url=url)

        return DocumentUrl(url=url)
