"""Image and video content filter for message history.

This module provides history processors that limit the number of images and videos
in message history and validates image content using Pillow.

Example::

    from contextlib import AsyncExitStack
    from pydantic_ai import Agent

    from pai_agent_sdk.context import AgentContext, ModelCapability, ModelConfig
    from pai_agent_sdk.environment.local import LocalEnvironment
    from pai_agent_sdk.filters.image import drop_extra_images, drop_extra_videos

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(LocalEnvironment())
        ctx = await stack.enter_async_context(
            AgentContext(
                file_operator=env.file_operator,
                shell=env.shell,
                model_cfg=ModelConfig(
                    max_images=20,  # Limit to 20 images (default)
                    max_videos=1,   # Limit to 1 video (default)
                    capabilities={ModelCapability.vision},
                ),
            )
        )
        agent = Agent(
            'openai:gpt-4',
            deps_type=AgentContext,
            history_processors=[drop_extra_images, drop_extra_videos],
        )
        result = await agent.run('Describe these images', deps=ctx)
"""

import io
from collections.abc import Sequence

from PIL import Image
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    UserContent,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.tools import RunContext

from pai_agent_sdk._logger import logger
from pai_agent_sdk.context import AgentContext


def _is_image_content(item: UserContent) -> bool:
    """Check if content item is an image."""
    if isinstance(item, ImageUrl):
        return True
    return isinstance(item, BinaryContent) and item.media_type.startswith("image/")


def _is_video_content(item: UserContent) -> bool:
    """Check if content item is a video."""
    if isinstance(item, VideoUrl):
        return True
    return isinstance(item, BinaryContent) and item.media_type.startswith("video/")


def _validate_image(content: BinaryContent) -> bool:
    """Validate image content using Pillow.

    Args:
        content: Binary content to validate.

    Returns:
        True if the image is valid, False otherwise.
    """
    try:
        image = Image.open(io.BytesIO(content.data))
        image.verify()
        return True
    except Exception:
        return False


def drop_extra_images(
    ctx: RunContext[AgentContext],
    message_history: list[ModelMessage],
) -> list[ModelMessage]:
    """Drop extra image content from message history and validate images.

    This is a pydantic-ai history_processor that:
    1. Limits the number of images to max_images (configured in ModelConfig)
    2. Validates images using Pillow and replaces broken images with text messages
    3. Keeps the most recent images (processes from newest to oldest)

    Args:
        ctx: Runtime context containing AgentContext with model configuration.
        message_history: List of messages to process.

    Returns:
        The modified message history with extra images dropped or replaced.

    Example:
        agent = Agent(
            'openai:gpt-4',
            deps_type=AgentContext,
            history_processors=[drop_extra_images],
        )
    """
    model_cfg = ctx.deps.model_cfg
    max_images = model_cfg.max_images if model_cfg else 20
    current_image_count = 0

    # Reverse iterate message history to keep the latest images
    for i in range(len(message_history) - 1, -1, -1):
        message = message_history[i]
        if not isinstance(message, ModelRequest):
            continue

        # Reverse iterate parts to keep the latest images
        for j in range(len(message.parts) - 1, -1, -1):
            part = message.parts[j]
            if not isinstance(part, UserPromptPart):
                continue

            content = part.content
            if isinstance(content, str):
                continue

            # Convert to list for in-place modification
            content_list: list[UserContent] = list(content) if isinstance(content, Sequence) else [content]

            # Reverse iterate content to keep the latest images
            for k in range(len(content_list) - 1, -1, -1):
                item = content_list[k]
                if not _is_image_content(item):
                    continue

                # Validate image using Pillow
                if isinstance(item, BinaryContent) and not _validate_image(item):
                    logger.info(f"Removing broken image at position {k}")
                    content_list[k] = (
                        "<system-reminder>This image content has been removed "
                        "because the image is broken or corrupted.</system-reminder>"
                    )
                    continue

                current_image_count += 1
                if current_image_count <= max_images:
                    continue

                # Drop the extra image
                logger.info(f"Dropping extra image content: {current_image_count} > {max_images}")
                content_list[k] = (
                    f"<system-reminder>This image content has been dropped "
                    f"as it exceeds the maximum allowed images (max_images={max_images}).</system-reminder>"
                )

            # Update the content
            part.content = content_list

    return message_history


def drop_gif_images(
    ctx: RunContext[AgentContext],
    message_history: list[ModelMessage],
) -> list[ModelMessage]:
    """Drop GIF images from message history when model doesn't support them.

    This is a pydantic-ai history_processor that removes GIF images when
    support_gif is False in ModelConfig. GIF images are replaced with a
    system reminder text.

    Args:
        ctx: Runtime context containing AgentContext with model configuration.
        message_history: List of messages to process.

    Returns:
        The modified message history with GIF images removed if not supported.

    Example:
        agent = Agent(
            'openai:gpt-4',
            deps_type=AgentContext,
            history_processors=[drop_gif_images],
        )
    """
    model_cfg = ctx.deps.model_cfg
    support_gif = model_cfg.support_gif if model_cfg else True

    if support_gif:
        return message_history

    for message in message_history:
        if not isinstance(message, ModelRequest):
            continue

        for part in message.parts:
            if not isinstance(part, UserPromptPart):
                continue

            content = part.content
            if isinstance(content, str):
                continue

            # Convert to list for in-place modification
            content_list: list[UserContent] = list(content) if isinstance(content, Sequence) else [content]

            new_content: list[UserContent] = []
            for item in content_list:
                if isinstance(item, BinaryContent) and item.media_type == "image/gif":
                    logger.info("Dropping GIF image as model does not support GIF")
                    new_content.append(
                        "<system-reminder>This GIF image has been removed "
                        "because the model does not support GIF images.</system-reminder>"
                    )
                else:
                    new_content.append(item)

            part.content = new_content

    return message_history


def drop_extra_videos(
    ctx: RunContext[AgentContext],
    message_history: list[ModelMessage],
) -> list[ModelMessage]:
    """Drop extra video content from message history.

    This is a pydantic-ai history_processor that limits the number of videos
    to max_videos (configured in ModelConfig). Older videos (appearing earlier
    in message history) are dropped first.

    Args:
        ctx: Runtime context containing AgentContext with model configuration.
        message_history: List of messages to process.

    Returns:
        The modified message history with excess videos replaced by system reminders.

    Example:
        agent = Agent(
            'openai:gpt-4',
            deps_type=AgentContext,
            history_processors=[drop_extra_videos],
        )
    """
    model_cfg = ctx.deps.model_cfg
    max_videos = model_cfg.max_videos if model_cfg else 1
    current_video_count = 0

    # Reverse iterate message history to keep the latest videos
    for i in range(len(message_history) - 1, -1, -1):
        message = message_history[i]
        if not isinstance(message, ModelRequest):
            continue

        # Reverse iterate parts to keep the latest videos
        for j in range(len(message.parts) - 1, -1, -1):
            part = message.parts[j]
            if not isinstance(part, UserPromptPart):
                continue

            content = part.content
            if isinstance(content, str):
                continue

            # Convert to list for in-place modification
            content_list: list[UserContent] = list(content) if isinstance(content, Sequence) else [content]

            # Reverse iterate content to keep the latest videos
            for k in range(len(content_list) - 1, -1, -1):
                item = content_list[k]
                if not _is_video_content(item):
                    continue

                current_video_count += 1
                if current_video_count <= max_videos:
                    continue

                # Drop the extra video
                logger.info(f"Dropping extra video content: {current_video_count} > {max_videos}")
                content_list[k] = (
                    f"<system-reminder>This video content has been dropped "
                    f"as it exceeds the maximum allowed videos (max_videos={max_videos}).</system-reminder>"
                )

            # Update the content
            part.content = content_list

    return message_history
