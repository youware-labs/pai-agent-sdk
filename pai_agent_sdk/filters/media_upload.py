"""Media upload filter for message history.

This module provides a history processor that uploads binary media content
(images and videos) to external storage and replaces them with URLs.

This filter should run AFTER image processing filters (compression, resize, etc.)
to upload the processed media rather than raw data.

Example::

    from pai_agent_sdk.filters.media_upload import create_media_upload_filter
    from pai_agent_sdk.media import S3MediaUploader, S3MediaConfig

    # Create uploader
    config = S3MediaConfig(bucket="my-bucket", url_mode="cdn", cdn_base_url="https://cdn.example.com")
    uploader = S3MediaUploader(config)

    # Create filter
    media_upload_filter = create_media_upload_filter(uploader)

    # Use in agent
    agent = Agent(
        'openai:gpt-4',
        deps_type=AgentContext,
        history_processors=[
            drop_extra_images,  # First: limit images
            media_upload_filter,  # Then: upload remaining to S3
        ],
    )
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    UserContent,
    UserPromptPart,
    VideoUrl,
)

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.context import AgentContext, ModelCapability
from pai_agent_sdk.media import MediaUploader

if TYPE_CHECKING:
    from pydantic_ai.tools import RunContext

__all__ = ["create_media_upload_filter"]

logger = get_logger(__name__)


async def _upload_media_item(
    uploader: MediaUploader,
    item: BinaryContent,
    is_video: bool,
) -> UserContent:
    """Upload a single media item and return URL content or original on failure."""
    try:
        url = await uploader.upload(item.data, item.media_type)
        if is_video:
            logger.debug("Uploaded video to %s", url)
            return VideoUrl(url=url)
        else:
            logger.debug("Uploaded image to %s", url)
            return ImageUrl(url=url)
    except Exception:
        media_type = "video" if is_video else "image"
        logger.exception("Failed to upload %s, keeping binary content", media_type)
        return item


async def _process_content_list(
    uploader: MediaUploader,
    content_list: list[UserContent],
    should_upload_images: bool,
    should_upload_videos: bool,
) -> bool:
    """Process content list, uploading media items in place. Returns True if modified."""
    modified = False
    for i, item in enumerate(content_list):
        if not isinstance(item, BinaryContent):
            continue

        if should_upload_images and item.media_type.startswith("image/"):
            content_list[i] = await _upload_media_item(uploader, item, is_video=False)
            modified = True
        elif should_upload_videos and item.media_type.startswith("video/"):
            content_list[i] = await _upload_media_item(uploader, item, is_video=True)
            modified = True

    return modified


def create_media_upload_filter(
    uploader: MediaUploader,
    *,
    upload_images: bool = True,
    upload_videos: bool = True,
) -> Callable[[RunContext[AgentContext], list[ModelMessage]], Awaitable[list[ModelMessage]]]:
    """Create a filter that uploads binary media to external storage and replaces with URLs.

    This filter checks model capabilities before uploading:
    - Only uploads images if model has IMAGE_URL capability
    - Only uploads videos if model has VIDEO_URL capability

    If upload fails, the original binary content is preserved.

    Args:
        uploader: Media uploader implementing the MediaUploader protocol.
        upload_images: Whether to upload images (default: True).
        upload_videos: Whether to upload videos (default: True).

    Returns:
        An async history processor function compatible with pydantic-ai.

    Example:
        from pai_agent_sdk.media import S3MediaUploader, S3MediaConfig

        config = S3MediaConfig(bucket="my-bucket")
        uploader = S3MediaUploader(config)
        filter_fn = create_media_upload_filter(uploader)

        agent = Agent(
            'openai:gpt-4',
            deps_type=AgentContext,
            history_processors=[filter_fn],
        )
    """

    async def filter_fn(
        ctx: RunContext[AgentContext],
        message_history: list[ModelMessage],
    ) -> list[ModelMessage]:
        """Upload binary media content and replace with URLs."""
        model_cfg = ctx.deps.model_cfg
        caps = model_cfg.capabilities if model_cfg else set()

        # Check what we should upload based on capabilities
        should_upload_images = upload_images and ModelCapability.image_url in caps
        should_upload_videos = upload_videos and ModelCapability.video_url in caps

        if not should_upload_images and not should_upload_videos:
            return message_history

        # Process messages
        for message in message_history:
            if not isinstance(message, ModelRequest):
                continue

            for part in message.parts:
                if not isinstance(part, UserPromptPart):
                    continue

                content = part.content
                if isinstance(content, str):
                    continue

                # Convert to list for modification
                content_list: list[UserContent] = list(content) if isinstance(content, Sequence) else [content]
                modified = await _process_content_list(
                    uploader, content_list, should_upload_images, should_upload_videos
                )

                if modified:
                    part.content = content_list

        return message_history

    return filter_fn
