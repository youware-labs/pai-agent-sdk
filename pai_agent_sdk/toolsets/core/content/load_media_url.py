"""Load media URL tool for loading multimedia content from HTTP/HTTPS URLs.

This tool allows the agent to load multimedia content (images, videos, audio,
documents) from web URLs and returns appropriate content types based on model
capabilities.

Note:
    Only HTTP and HTTPS URLs are supported. Other protocols will be rejected.

Example::

    from pai_agent_sdk.context import AgentContext, ModelCapability, ModelConfig
    from pai_agent_sdk.toolsets.core.base import Toolset
    from pai_agent_sdk.toolsets.core.content.load_media_url import LoadMediaUrlTool

    async with AgentContext(env=env) as ctx:
        toolset = Toolset(tools=[LoadMediaUrlTool])
        # Agent can now load multimedia URLs and get appropriate content types
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
    """Load load_media_url instruction template from prompts/load_media_url.md."""
    prompt_file = _PROMPTS_DIR / "load_media_url.md"
    return Template(prompt_file.read_text())


_INSTRUCTION_TEMPLATE = _load_instruction_template()


class LoadMediaUrlTool(BaseTool):
    """Tool for loading multimedia content from HTTP/HTTPS URLs.

    This tool validates that the URL uses HTTP or HTTPS protocol,
    detects the content type (image, video, audio, document), and returns
    an appropriate URL type based on the model's capabilities.
    """

    name = "load_media_url"
    description = "Load multimedia content directly from HTTP/HTTPS URL (images, videos, audio). e.g. https://example.com/image.png, https://example.com/video.mp4, https://youtube.com/watch?v=abc123"

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available based on model capabilities.

        LoadTool requires at least one of: vision, video_understanding, or document_understanding.
        If none are available, the tool should be disabled.
        """
        model_cfg = ctx.deps.model_cfg
        if model_cfg is None:
            return False

        has_vision = model_cfg.has_capability(ModelCapability.vision)
        has_video = model_cfg.has_capability(ModelCapability.video_understanding)
        has_document = model_cfg.has_capability(ModelCapability.document_understanding)
        enable_load_document = ctx.deps.tool_config.enable_load_document

        # Available if any capability is present
        return has_vision or has_video or (has_document and enable_load_document)

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str:
        """Generate dynamic instruction based on model capabilities."""
        model_cfg = ctx.deps.model_cfg
        has_vision = model_cfg is not None and model_cfg.has_capability(ModelCapability.vision)
        has_video = model_cfg is not None and model_cfg.has_capability(ModelCapability.video_understanding)
        has_document = model_cfg is not None and model_cfg.has_capability(ModelCapability.document_understanding)
        enable_load_document = ctx.deps.tool_config.enable_load_document

        return _INSTRUCTION_TEMPLATE.render(
            has_vision=has_vision,
            has_video=has_video,
            has_document=has_document,
            enable_load_document=enable_load_document,
        )

    async def call(
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
        enable_load_document = ctx.deps.tool_config.enable_load_document

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

        return f"Unknown content category: {category}, try again with a different URL or use `fetch` tool to download."
