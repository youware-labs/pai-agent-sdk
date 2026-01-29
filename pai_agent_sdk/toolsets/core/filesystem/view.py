"""View tool for reading files.

Supports text files, images, videos, and audio files.
All file operations use the FileOperator abstraction for remote filesystem support.
"""

import asyncio
from functools import cache
from pathlib import Path
from typing import Annotated, Any, cast

from agent_environment import FileOperator
from pydantic import Field
from pydantic_ai import BinaryContent, ImageUrl, RunContext, ToolReturn, VideoUrl

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.base import BaseTool
from pai_agent_sdk.toolsets.core.filesystem._types import (
    ViewMetadata,
    ViewReadingParams,
    ViewSegment,
    ViewTruncationInfo,
)
from pai_agent_sdk.utils import run_in_threadpool

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Image file extensions that can be displayed as BinaryContent
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".webp"})

# Image media types supported for display in the LLM context
SUPPORTED_IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})

# Video file extensions
VIDEO_EXTENSIONS = frozenset({
    ".mp4",
    ".webm",
    ".mov",
    ".avi",
    ".flv",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".3gp",
    ".mkv",
    ".m4v",
    ".ogv",
})

# Media type mapping for common extensions
MEDIA_TYPE_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".ico": "image/x-icon",
}

# Video media type mapping
VIDEO_MEDIA_TYPE_MAP = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".flv": "video/x-flv",
    ".wmv": "video/x-ms-wmv",
    ".mpg": "video/mpeg",
    ".mpeg": "video/mpeg",
    ".3gp": "video/3gpp",
    ".mkv": "video/x-matroska",
    ".m4v": "video/x-m4v",
    ".ogv": "video/ogg",
}


@cache
def _load_instruction() -> str:
    """Load view instruction from prompts/view.md."""
    prompt_file = _PROMPTS_DIR / "view.md"
    return prompt_file.read_text()


class ViewTool(BaseTool):
    """Tool for reading files from the filesystem.

    Supports text files, images, and videos.
    All operations use FileOperator abstraction for remote filesystem support.
    """

    name = "view"
    description = (
        "Read files from local filesystem. Supports text, images (PNG/JPEG/WebP), and videos (MP4/WebM/MOV). "
        "For PDF files, use `pdf_convert` tool instead."
    )

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available (requires file_operator)."""
        if ctx.deps.file_operator is None:
            logger.debug("ViewTool unavailable: file_operator is not configured")
            return False
        return True

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Load instruction from prompts/view.md."""
        return _load_instruction()

    # --- Path and type utilities (string-based, no Path dependency) ---

    def _get_extension(self, file_path: str) -> str:
        """Extract file extension from path string."""
        idx = file_path.rfind(".")
        if idx == -1:
            return ""
        # Handle cases like "/path/to/.hidden" where dot is part of filename
        last_sep = max(file_path.rfind("/"), file_path.rfind("\\"))
        if idx < last_sep:
            return ""
        return file_path[idx:].lower()

    def _is_image_file(self, file_path: str) -> bool:
        """Check if a file is a displayable image based on extension."""
        return self._get_extension(file_path) in IMAGE_EXTENSIONS

    def _is_video_file(self, file_path: str) -> bool:
        """Check if a file is a video based on extension."""
        return self._get_extension(file_path) in VIDEO_EXTENSIONS

    def _get_media_type(self, file_path: str) -> str:
        """Get media type from file extension."""
        ext = self._get_extension(file_path)
        return MEDIA_TYPE_MAP.get(ext, "application/octet-stream")

    def _get_video_media_type(self, file_path: str) -> str:
        """Get video media type from file extension."""
        ext = self._get_extension(file_path)
        return VIDEO_MEDIA_TYPE_MAP.get(ext, "video/mp4")

    # --- File reading methods (async, using FileOperator) ---

    async def _read_image(self, file_operator: FileOperator, file_path: str) -> BinaryContent:
        """Read image file and return BinaryContent."""
        content = await file_operator.read_bytes(file_path)
        media_type = self._get_media_type(file_path)

        # Normalize unsupported media types
        if media_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
            media_type = "image/png"

        return BinaryContent(data=content, media_type=media_type)

    async def _read_image_with_fallback(
        self,
        file_operator: FileOperator,
        file_path: str,
        ctx: RunContext[AgentContext],
    ) -> str | ToolReturn:
        """Read image file, falling back to description if vision not supported."""
        # Read image data and determine media type
        image_data = await file_operator.read_bytes(file_path)
        media_type = self._get_media_type(file_path)

        # Normalize unsupported media types
        if media_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
            media_type = "image/png"

        # Try to convert to URL using hook
        image_url: str | None = None
        if ctx.deps.tool_config and ctx.deps.tool_config.image_to_url_hook:
            try:
                hook = ctx.deps.tool_config.image_to_url_hook
                if asyncio.iscoroutinefunction(hook):
                    result = await hook(ctx, image_data, media_type)
                else:
                    # Run sync hook in threadpool to avoid blocking event loop
                    result = await run_in_threadpool(hook, ctx, image_data, media_type)
                # Treat empty strings as None
                result = cast("str | None", result)
                image_url = result if result and result.strip() else None
            except Exception:
                logger.warning("image_to_url_hook failed, falling back to data", exc_info=True)

        # Check if current model supports vision
        has_vision = ctx.deps.model_cfg.has_vision

        if has_vision:
            # Return image content directly
            if image_url:
                return ToolReturn(
                    return_value="The image is attached in the user message.",
                    content=[ImageUrl(url=image_url)],
                )
            else:
                return ToolReturn(
                    return_value="The image is attached in the user message.",
                    content=[BinaryContent(data=image_data, media_type=media_type)],
                )
        else:
            # Fall back to image understanding agent
            try:
                from pai_agent_sdk.agents.image_understanding import get_image_description

                # Get model and settings from tool_config if available
                model = None
                model_settings = None
                if ctx.deps.tool_config:
                    tool_config = ctx.deps.tool_config
                    model = tool_config.image_understanding_model
                    model_settings = tool_config.image_understanding_model_settings

                # Use image understanding to describe (prefer URL if available)
                description, internal_usage = await get_image_description(
                    image_url=image_url,
                    image_data=None if image_url else image_data,
                    media_type=media_type,
                    model=model,
                    model_settings=model_settings,
                    model_wrapper=ctx.deps.model_wrapper,
                    wrapper_context=ctx.deps.get_wrapper_context(),
                )

                # Store usage in extra_usages
                if ctx.tool_call_id:
                    ctx.deps.add_extra_usage(
                        agent="image_understanding", internal_usage=internal_usage, uuid=ctx.tool_call_id
                    )

                return f"Image description (via image analysis):\n{description}"
            except Exception as e:
                logger.warning(f"Failed to analyze image with image understanding: {e}")
                return f"Image file: {file_path}. Model does not support vision and fallback analysis failed."

    async def _read_video_with_fallback(
        self,
        file_operator: FileOperator,
        file_path: str,
        ctx: RunContext[AgentContext],
    ) -> str | ToolReturn:
        """Read video file, falling back to video understanding agent if not supported."""
        # Read video data and determine media type
        video_data = await file_operator.read_bytes(file_path)
        media_type = self._get_video_media_type(file_path)

        # Try to convert to URL using hook
        video_url: str | None = None
        if ctx.deps.tool_config and ctx.deps.tool_config.video_to_url_hook:
            try:
                hook = ctx.deps.tool_config.video_to_url_hook
                if asyncio.iscoroutinefunction(hook):
                    result = await hook(ctx, video_data, media_type)
                else:
                    # Run sync hook in threadpool to avoid blocking event loop
                    result = await run_in_threadpool(hook, ctx, video_data, media_type)
                # Treat empty strings as None
                result = cast("str | None", result)
                video_url = result if result and result.strip() else None
            except Exception:
                logger.warning("video_to_url_hook failed, falling back to data", exc_info=True)

        # Check if current model supports video understanding
        has_video = ctx.deps.model_cfg.has_video_understanding

        if has_video:
            # Return video content directly
            if video_url:
                return ToolReturn(
                    return_value="The video is attached in the user message.",
                    content=[VideoUrl(url=video_url)],
                )
            else:
                return ToolReturn(
                    return_value="The video is attached in the user message.",
                    content=[BinaryContent(data=video_data, media_type=media_type)],
                )
        else:
            # Fall back to video understanding agent
            try:
                from pai_agent_sdk.agents.video_understanding import get_video_description

                # Get model and settings from tool_config if available
                model = None
                model_settings = None
                if ctx.deps.tool_config:
                    tool_config = ctx.deps.tool_config
                    model = tool_config.video_understanding_model
                    model_settings = tool_config.video_understanding_model_settings

                # Use video understanding to describe video (prefer URL if available)
                description, internal_usage = await get_video_description(
                    video_url=video_url,
                    video_data=None if video_url else video_data,
                    media_type=media_type,
                    model=model,
                    model_settings=model_settings,
                )

                # Store usage in extra_usages with tool_call_id
                if ctx.tool_call_id:
                    ctx.deps.add_extra_usage(
                        agent="video_understanding", internal_usage=internal_usage, uuid=ctx.tool_call_id
                    )

                return f"Video description (via video understanding agent):\n{description}"
            except Exception as e:
                logger.warning(f"Failed to analyze video with video understanding: {e}")
                return (
                    f"Video file: {file_path}. Model does not support video understanding and fallback analysis failed."
                )

    async def _read_text_file(
        self,
        file_operator: FileOperator,
        file_path: str,
        line_offset: int | None,
        line_limit: int,
        max_line_length: int,
    ) -> str | dict[str, Any]:
        """Read text file with pagination and truncation support."""
        lines_truncated = False
        content_truncated = False

        # Read file content
        full_content = await file_operator.read_file(file_path)
        all_lines = full_content.splitlines(keepends=True)

        # Get file stats
        stat = await file_operator.stat(file_path)

        total_lines = len(all_lines)
        total_chars = len(full_content)
        file_size = stat["size"]

        if line_offset is not None and line_offset > 0:
            all_lines = all_lines[line_offset:]
            has_offset = True
        else:
            has_offset = False
            line_offset = 0

        if len(all_lines) > line_limit:
            all_lines = all_lines[:line_limit]
            has_line_limit = True
        else:
            has_line_limit = False

        processed_lines = []
        for line in all_lines:
            if len(line) > max_line_length:
                line = line[:max_line_length] + "... (line truncated)\n"
                lines_truncated = True
            processed_lines.append(line)

        content = "".join(processed_lines)

        if len(content) > 60000:
            content = content[:60000] + "\n... (content truncated)"
            content_truncated = True

        needs_metadata = has_offset or has_line_limit or lines_truncated or content_truncated

        if not needs_metadata:
            return content
        else:
            start_line = line_offset + 1
            actual_lines_read = len(processed_lines)
            end_line = start_line + actual_lines_read - 1 if actual_lines_read > 0 else start_line

            # Extract filename from path
            last_sep = max(file_path.rfind("/"), file_path.rfind("\\"))
            filename = file_path[last_sep + 1 :] if last_sep != -1 else file_path

            return {
                "content": content,
                "metadata": ViewMetadata(
                    file_path=filename,
                    total_lines=total_lines,
                    total_characters=total_chars,
                    file_size_bytes=file_size,
                    current_segment=ViewSegment(
                        start_line=start_line,
                        end_line=end_line,
                        lines_to_show=actual_lines_read,
                        has_more_content=end_line < total_lines,
                    ),
                    reading_parameters=ViewReadingParams(
                        line_offset=line_offset if has_offset else None,
                        line_limit=line_limit,
                    ),
                    truncation_info=ViewTruncationInfo(
                        lines_truncated=lines_truncated,
                        content_truncated=content_truncated,
                        max_line_length=max_line_length,
                    ),
                ),
                "system": "Increase the `line_limit` and `max_line_length` if you need more context.",
            }

    # --- Main entry point ---

    async def call(
        self,
        ctx: RunContext[AgentContext],
        file_path: Annotated[
            str,
            Field(description="Relative path to the file to read"),
        ],
        line_offset: Annotated[
            int | None,
            Field(
                description="Line number to start reading from (0-indexed)",
                default=None,
            ),
        ] = None,
        line_limit: Annotated[
            int,
            Field(
                description="Maximum number of lines to read (default: 300)",
                default=300,
            ),
        ] = 300,
        max_line_length: Annotated[
            int,
            Field(
                description="Maximum length of each line before truncation",
                default=2000,
            ),
        ] = 2000,
    ) -> str | dict[str, Any] | ToolReturn:
        """Read a file from the filesystem."""
        file_operator = cast(FileOperator, ctx.deps.file_operator)

        if not await file_operator.exists(file_path):
            return f"Error: File not found: {file_path}"

        if await file_operator.is_dir(file_path):
            return f"Error: Path is a directory, not a file: {file_path}"

        if self._is_image_file(file_path):
            return await self._read_image_with_fallback(file_operator, file_path, ctx)

        if self._is_video_file(file_path):
            return await self._read_video_with_fallback(file_operator, file_path, ctx)

        return await self._read_text_file(file_operator, file_path, line_offset, line_limit, max_line_length)


__all__ = ["ViewTool"]
