"""Video understanding agent for analyzing video content.

This module provides a video understanding agent that can analyze both
screen recordings and general video content, returning structured descriptions.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from inspect import isawaitable
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, ModelSettings, VideoUrl
from pydantic_ai.models import Model

from pai_agent_sdk._config import AgentSettings
from pai_agent_sdk._logger import logger
from pai_agent_sdk.agents.models import infer_model
from pai_agent_sdk.usage import InternalUsage


class VideoError(Exception):
    """Base exception for video processing errors."""

    pass


class VideoSizeError(VideoError):
    """Raised when video content exceeds the maximum allowed size."""

    def __init__(self, size: int, max_size: int):
        self.size = size
        self.max_size = max_size
        size_mb = size / (1024 * 1024)
        max_mb = max_size / (1024 * 1024)
        super().__init__(f"Video size ({size_mb:.2f}MB) exceeds maximum allowed size ({max_mb:.0f}MB)")


class VideoInputError(VideoError):
    """Raised when video input is invalid."""

    def __init__(self, message: str):
        super().__init__(message)


class VideoAnalysisError(VideoError):
    """Raised when video analysis fails."""

    def __init__(self, message: str, cause: Exception | None = None):
        self.cause = cause
        super().__init__(message)


# Default maximum video content size for base64 encoding (10MB)
DEFAULT_MAX_VIDEO_SIZE = 10 * 1024 * 1024

# Mapping of file extensions to video media types
VIDEO_MEDIA_TYPES: dict[str, str] = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".m4v": "video/x-m4v",
    ".ogv": "video/ogg",
    ".wmv": "video/x-ms-wmv",
    ".flv": "video/x-flv",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".3gp": "video/3gpp",
    ".3g2": "video/3gpp2",
}

AGENT_NAME = "video-understanding"

DEFAULT_VIDEO_ANALYSIS_INSTRUCTION = """Watch this video carefully and describe everything you observe in as much detail as possible.

Include:
- What is happening in the video from start to end
- All visual elements, scenes, objects, people, UI components, text, etc.
- Any audio content: speech (transcribe it), music, sound effects
- The context, purpose, or intent behind what's shown
- Any notable details, transitions, or key moments

Be thorough and comprehensive. The more detail, the better.
"""


# =============================================================================
# Utilities
# =============================================================================


def guess_media_type(source: str | Path) -> str:
    """Guess video media type from URL or file path.

    Args:
        source: URL string or Path object

    Returns:
        Media type string, defaults to 'video/mp4' if unknown
    """
    if isinstance(source, Path):
        ext = source.suffix.lower()
    else:
        parsed = urlparse(source)
        path = parsed.path
        ext = Path(path).suffix.lower() if path else ""

    return VIDEO_MEDIA_TYPES.get(ext, "video/mp4")


def build_video_content(
    video_url: str | None = None,
    video_data: bytes | None = None,
    media_type: str | None = None,
    max_size: int = DEFAULT_MAX_VIDEO_SIZE,
) -> VideoUrl | BinaryContent:
    """Build video input content for AI model consumption.

    Args:
        video_url: URL of the video (mutually exclusive with video_data)
        video_data: Raw video bytes (mutually exclusive with video_url)
        media_type: Optional media type override
        max_size: Maximum allowed size for video_data in bytes

    Returns:
        VideoUrl for remote videos or BinaryContent for binary data

    Raises:
        VideoInputError: If neither or both video_url and video_data are provided
        VideoSizeError: If video_data exceeds max_size
    """
    if not video_url and not video_data:
        raise VideoInputError("Either video_url or video_data must be provided")

    if video_url and video_data:
        raise VideoInputError("Both video_url and video_data cannot be provided")

    if video_url:
        logger.debug(f"Building video content from URL: {video_url}")
        return VideoUrl(url=video_url, media_type=media_type or guess_media_type(video_url))

    # video_data is guaranteed to be not None here due to the checks above
    video_bytes = video_data  # type narrowing for pyright
    if video_bytes is None:
        raise VideoInputError("video_data is required when video_url is not provided")

    if len(video_bytes) > max_size:
        raise VideoSizeError(len(video_bytes), max_size)

    logger.debug(f"Building video content from binary data: {len(video_bytes)} bytes")
    return BinaryContent(
        data=video_bytes,
        media_type=media_type or "video/mp4",
    )


def _load_system_prompt() -> str:
    """Load system prompt from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / "video_understanding.md"
    if prompt_path.exists():
        return prompt_path.read_text()
    return ""


# =============================================================================
# Models
# =============================================================================


class VideoDescription(BaseModel):
    """Minimal constraint - let the model freely describe the video."""

    description: str = Field(description="Detailed, comprehensive description of everything in the video")


def get_video_understanding_agent(
    model: str | Model | None = None,
    model_settings: ModelSettings | None = None,
) -> Agent[None, VideoDescription]:
    """Create a video understanding agent.

    This is an async function to allow for future async initialization needs.

    Args:
        model: Model string or Model instance. If None, uses config setting.
        model_settings: Optional model settings dict.

    Returns:
        Agent configured for video understanding.

    Raises:
        ValueError: If no model is specified and config has no default.
    """
    if model is None:
        settings = AgentSettings()
        if settings.video_understanding_model:
            model = settings.video_understanding_model
        else:
            raise ValueError("No model specified. Provide model parameter or set PAI_AGENT_VIDEO_UNDERSTANDING_MODEL.")

    model_instance = infer_model(model) if isinstance(model, str) else model

    system_prompt = _load_system_prompt()

    return Agent[None, VideoDescription](
        model_instance,
        output_type=VideoDescription,
        system_prompt=system_prompt,
        model_settings=model_settings,
        retries=3,
        output_retries=3,
    )


async def get_video_description(
    video_url: str | None = None,
    video_data: bytes | None = None,
    media_type: str | None = None,
    instruction: str | None = None,
    model: str | Model | None = None,
    model_settings: ModelSettings | None = None,
    max_video_size: int = DEFAULT_MAX_VIDEO_SIZE,
    model_wrapper: Callable[[Model, str, dict[str, Any]], Model | Awaitable[Model]] | None = None,
    wrapper_context: dict[str, Any] | None = None,
) -> tuple[str, InternalUsage]:
    """Analyze a video and get a structured description.

    Args:
        video_url: URL of the video to analyze.
        video_data: Raw video bytes to analyze.
        media_type: Optional media type override.
        instruction: Custom instruction for analysis. If None, uses default.
        model: Model string or Model instance.
        model_settings: Optional model settings dict.
        max_video_size: Maximum allowed size for video_data in bytes.
        model_wrapper: Optional wrapper for model instrumentation.
        wrapper_context: Context dict passed to model_wrapper (e.g., from ctx.get_wrapper_context()).

    Returns:
        Tuple of (description string, InternalUsage with model_id and usage).

    Raises:
        VideoInputError: If video input is invalid.
        VideoSizeError: If video exceeds size limit.
        VideoAnalysisError: If analysis fails.
    """

    video_content = build_video_content(
        video_url=video_url,
        video_data=video_data,
        media_type=media_type,
        max_size=max_video_size,
    )

    agent = get_video_understanding_agent(model=model, model_settings=model_settings)

    # Apply model wrapper if configured
    if model_wrapper is not None:
        effective_context = wrapper_context or {}
        wrapped = model_wrapper(cast(Model, agent.model), AGENT_NAME, effective_context)
        agent.model = await wrapped if isawaitable(wrapped) else wrapped

    try:
        result = await agent.run(
            [
                instruction or DEFAULT_VIDEO_ANALYSIS_INSTRUCTION,
                video_content,
            ],
        )
    except Exception as e:
        logger.error(f"Error analyzing video: {e}")
        raise VideoAnalysisError(f"Failed to analyze video: {e}", cause=e) from e

    # Get model_id from agent's model
    model_id = cast(Model, agent.model).model_name

    return result.output.description, InternalUsage(model_id=model_id, usage=result.usage())
