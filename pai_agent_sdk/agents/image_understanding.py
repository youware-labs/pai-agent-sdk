"""Image understanding agent for analyzing image content.

This module provides an image understanding agent that can analyze images
and return structured descriptions including visual elements, text content,
and design style analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse
from xml.etree.ElementTree import Element, SubElement, tostring

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, ImageUrl
from pydantic_ai.usage import RunUsage

from pai_agent_sdk._config import AgentSettings
from pai_agent_sdk._logger import logger
from pai_agent_sdk.agents.models import Model, infer_model

if TYPE_CHECKING:
    pass


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

DEFAULT_IMAGE_ANALYSIS_INSTRUCTION = """<instruction>
  <task>Analyze this image and provide a structured description.</task>

  <focus-areas>
    <area>Identify all visual elements and their relationships</area>
    <area>Extract ALL visible text (OCR)</area>
    <area>Analyze design style if applicable (colors, typography, layout)</area>
    <area>Provide CSS reference for web/UI designs</area>
  </focus-areas>
</instruction>"""


# =============================================================================
# Utilities
# =============================================================================


def _xml_to_string(element: Element) -> str:
    """Convert XML element to formatted string."""
    from xml.dom.minidom import parseString

    rough_string = tostring(element, encoding="unicode")
    dom = parseString(rough_string)  # noqa: S318
    lines = dom.toprettyxml(indent="  ").split("\n")[1:]
    return "\n".join(line for line in lines if line.strip())


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
        content = prompt_path.read_text()
        # Extract XML content from markdown code block
        if "```xml" in content:
            start = content.find("```xml") + 6
            end = content.find("```", start)
            return content[start:end].strip()
        return content
    return ""


# =============================================================================
# Models
# =============================================================================


class ImageDescription(BaseModel):
    """Structured image description for downstream AI agents.

    This model provides comprehensive image content analysis including
    visual elements, text content, and optional design style analysis.
    """

    # Required fields
    summary: str = Field(description="Brief description of the image (2-4 sentences)")
    visual_elements: list[str] = Field(description="Key visual elements: objects, people, UI components, shapes, etc.")
    text_content: list[str] = Field(
        description="All visible text in the image (OCR): labels, buttons, titles, captions"
    )

    # Optional fields - agent decides based on image content
    style_analysis: str | None = Field(
        default=None,
        description="Design style analysis: colors, typography, layout, visual effects, aesthetic",
    )
    css_reference: str | None = Field(
        default=None,
        description="CSS code snippet for web/UI designs: color variables, fonts, effects",
    )
    context: str | None = Field(
        default=None,
        description="Inferred purpose or source of the image",
    )
    key_observations: list[str] | None = Field(
        default=None,
        description="Notable details, issues, or important elements",
    )

    def to_xml(self) -> str:
        """Serialize the image description to XML format."""
        root = Element("image-description")

        # Required fields
        SubElement(root, "summary").text = self.summary

        visual_elem = SubElement(root, "visual-elements")
        for item in self.visual_elements:
            SubElement(visual_elem, "element").text = item

        text_elem = SubElement(root, "text-content")
        for item in self.text_content:
            SubElement(text_elem, "text").text = item

        # Optional fields
        if self.style_analysis:
            SubElement(root, "style-analysis").text = self.style_analysis

        if self.css_reference:
            SubElement(root, "css-reference").text = self.css_reference

        if self.context:
            SubElement(root, "context").text = self.context

        if self.key_observations:
            obs_elem = SubElement(root, "key-observations")
            for obs in self.key_observations:
                SubElement(obs_elem, "observation").text = obs

        return _xml_to_string(root)

    def __repr__(self) -> str:
        return self.to_xml()

    def __str__(self) -> str:
        return self.to_xml()


# =============================================================================
# Agent Factory and API
# =============================================================================


def get_image_understanding_agent(
    model: str | Model | None = None,
    model_settings: dict[str, Any] | None = None,
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

    return Agent[None, ImageDescription](  # pyright: ignore[reportCallIssue]
        model_instance,
        output_type=ImageDescription,
        system_prompt=system_prompt,
        model_settings=model_settings,  # pyright: ignore[reportArgumentType]
        retries=3,
        output_retries=3,
    )


async def get_image_description(
    image_url: str | None = None,
    image_data: bytes | None = None,
    media_type: str | None = None,
    instruction: str | None = None,
    model: str | Model | None = None,
    model_settings: dict[str, Any] | None = None,
    max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
) -> tuple[str, RunUsage]:
    """Analyze an image and get a structured description.

    Args:
        image_url: URL of the image to analyze.
        image_data: Raw image bytes to analyze.
        media_type: Optional media type override.
        instruction: Custom instruction for analysis. If None, uses default.
        model: Model string or Model instance.
        model_settings: Optional model settings dict.
        max_image_size: Maximum allowed size for image_data in bytes.

    Returns:
        Tuple of (XML description string, RunUsage).

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

    return result.output.to_xml(), result.usage()
