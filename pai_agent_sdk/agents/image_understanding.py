"""Image understanding agent for analyzing image content.

This module provides an image understanding agent that can analyze images
and return structured descriptions including visual elements, text content,
and design style analysis.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from inspect import isawaitable
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, ImageUrl, ModelSettings
from pydantic_ai.models import Model

from pai_agent_sdk._config import AgentSettings
from pai_agent_sdk._logger import logger
from pai_agent_sdk.agents.models import infer_model
from pai_agent_sdk.usage import InternalUsage

# =============================================================================
# Exceptions
# =============================================================================


class ImageError(Exception):
    """Base exception for image processing errors."""

    pass


class ImageSizeError(ImageError):
    """Raised when image content exceeds the maximum allowed size."""

    def __init__(self, size: int, max_size: int):
        self.size = size
        self.max_size = max_size
        size_mb = size / (1024 * 1024)
        max_mb = max_size / (1024 * 1024)
        super().__init__(f"Image size ({size_mb:.2f}MB) exceeds maximum allowed size ({max_mb:.0f}MB)")


class ImageInputError(ImageError):
    """Raised when image input is invalid."""

    def __init__(self, message: str):
        super().__init__(message)


class ImageAnalysisError(ImageError):
    """Raised when image analysis fails."""

    def __init__(self, message: str, cause: Exception | None = None):
        self.cause = cause
        super().__init__(message)


# =============================================================================
# Constants
# =============================================================================

# Default maximum image content size for base64 encoding (10MB)
DEFAULT_MAX_IMAGE_SIZE = 10 * 1024 * 1024

# Mapping of file extensions to image media types
IMAGE_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}

AGENT_NAME = "image-understanding"

DEFAULT_IMAGE_ANALYSIS_INSTRUCTION = """Look at this image carefully and describe everything you observe in as much detail as possible.

Include:
- All visual elements, objects, people, UI components, text, etc.
- Extract ALL visible text (OCR)
- Design style if applicable (colors, typography, layout)
- For web/UI designs, provide CSS code snippets for key visual styles
- The context, purpose, or intent behind what's shown
- Any notable details or observations

Be thorough and comprehensive. The more detail, the better.
"""


# =============================================================================
# Utilities
# =============================================================================


def guess_media_type(source: str | Path) -> str:
    """Guess image media type from URL or file path.

    Args:
        source: URL string or Path object

    Returns:
        Media type string, defaults to 'image/png' if unknown
    """
    if isinstance(source, Path):
        ext = source.suffix.lower()
    else:
        parsed = urlparse(source)
        path = parsed.path
        ext = Path(path).suffix.lower() if path else ""

    return IMAGE_MEDIA_TYPES.get(ext, "image/png")


def build_image_content(
    image_url: str | None = None,
    image_data: bytes | None = None,
    media_type: str | None = None,
    max_size: int = DEFAULT_MAX_IMAGE_SIZE,
) -> ImageUrl | BinaryContent:
    """Build image input content for AI model consumption.

    Args:
        image_url: URL of the image (mutually exclusive with image_data)
        image_data: Raw image bytes (mutually exclusive with image_url)
        media_type: Optional media type override
        max_size: Maximum allowed size for image_data in bytes

    Returns:
        ImageUrl for remote images or BinaryContent for binary data

    Raises:
        ImageInputError: If neither or both image_url and image_data are provided
        ImageSizeError: If image_data exceeds max_size
    """
    if not image_url and not image_data:
        raise ImageInputError("Either image_url or image_data must be provided")

    if image_url and image_data:
        raise ImageInputError("Both image_url and image_data cannot be provided")

    if image_url:
        logger.debug(f"Building image content from URL: {image_url}")
        return ImageUrl(url=image_url, media_type=media_type or guess_media_type(image_url))

    # image_data is guaranteed to be not None here due to the checks above
    image_bytes = image_data
    if image_bytes is None:
        raise ImageInputError("image_data is required when image_url is not provided")

    if len(image_bytes) > max_size:
        raise ImageSizeError(len(image_bytes), max_size)

    logger.debug(f"Building image content from binary data: {len(image_bytes)} bytes")
    return BinaryContent(
        data=image_bytes,
        media_type=media_type or "image/png",
    )


def _load_system_prompt() -> str:
    """Load system prompt from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / "image_understanding.md"
    if prompt_path.exists():
        return prompt_path.read_text()
    return ""


# =============================================================================
# Models
# =============================================================================


class ImageDescription(BaseModel):
    """Minimal constraint - let the model freely describe the image."""

    description: str = Field(description="Detailed, comprehensive description of everything in the image")


# =============================================================================
# Agent Factory and API
# =============================================================================


def get_image_understanding_agent(
    model: str | Model | None = None,
    model_settings: ModelSettings | None = None,
) -> Agent[None, ImageDescription]:
    """Create an image understanding agent.

    Args:
        model: Model string or Model instance. If None, uses config setting.
        model_settings: Optional model settings dict.

    Returns:
        Agent configured for image understanding.

    Raises:
        ValueError: If no model is specified and config has no default.
    """
    if model is None:
        settings = AgentSettings()
        if settings.image_understanding_model:
            model = settings.image_understanding_model
        else:
            raise ValueError("No model specified. Provide model parameter or set PAI_AGENT_IMAGE_UNDERSTANDING_MODEL.")

    model_instance = infer_model(model) if isinstance(model, str) else model

    system_prompt = _load_system_prompt()

    return Agent[None, ImageDescription](
        model_instance,
        output_type=ImageDescription,
        system_prompt=system_prompt,
        model_settings=model_settings,
        retries=3,
        output_retries=3,
    )


async def get_image_description(
    image_url: str | None = None,
    image_data: bytes | None = None,
    media_type: str | None = None,
    instruction: str | None = None,
    model: str | Model | None = None,
    model_settings: ModelSettings | None = None,
    max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
    model_wrapper: Callable[[Model, str, dict[str, Any]], Model | Awaitable[Model]] | None = None,
    wrapper_metadata: dict[str, Any] | None = None,
) -> tuple[str, InternalUsage]:
    """Analyze an image and get a structured description.

    Args:
        image_url: URL of the image to analyze.
        image_data: Raw image bytes to analyze.
        media_type: Optional media type override.
        instruction: Custom instruction for analysis. If None, uses default.
        model: Model string or Model instance.
        model_settings: Optional model settings dict.
        max_image_size: Maximum allowed size for image_data in bytes.
        model_wrapper: Optional wrapper for model instrumentation.
        wrapper_metadata: Context dict passed to model_wrapper (e.g., from ctx.get_wrapper_metadata()).

    Returns:
        Tuple of (description string, InternalUsage with model_id and usage).

    Raises:
        ImageInputError: If image input is invalid.
        ImageSizeError: If image exceeds size limit.
        ImageAnalysisError: If analysis fails.
    """

    image_content = build_image_content(
        image_url=image_url,
        image_data=image_data,
        media_type=media_type,
        max_size=max_image_size,
    )

    agent = get_image_understanding_agent(model=model, model_settings=model_settings)

    # Apply model wrapper if configured
    if model_wrapper is not None:
        effective_context = wrapper_metadata or {}
        wrapped = model_wrapper(cast(Model, agent.model), AGENT_NAME, effective_context)
        agent.model = await wrapped if isawaitable(wrapped) else wrapped

    try:
        result = await agent.run(
            [
                instruction or DEFAULT_IMAGE_ANALYSIS_INSTRUCTION,
                image_content,
            ],
        )
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise ImageAnalysisError(f"Failed to analyze image: {e}", cause=e) from e

    # Get model_id from agent's model
    model_id = cast(Model, agent.model).model_name

    return result.output.description, InternalUsage(model_id=model_id, usage=result.usage())
