"""Video understanding agent for analyzing video content.

This module provides a video understanding agent that can analyze both
screen recordings and general video content, returning structured descriptions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse
from xml.etree.ElementTree import Element, SubElement, tostring

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, BinaryContent, VideoUrl

from pai_agent_sdk._config import AgentSettings
from pai_agent_sdk._logger import logger
from pai_agent_sdk.agents.models import Model, infer_model

if TYPE_CHECKING:
    from typing import Self

from pydantic_ai.usage import RunUsage


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

DEFAULT_VIDEO_ANALYSIS_INSTRUCTION = """<instruction>
  <task>Analyze this video and provide a structured description based on its content type.</task>

  <for-screen-recordings>
    <focus>Capture the complete sequence of user actions</focus>
    <focus>Note the applications, interfaces, and UI elements involved</focus>
    <focus>Describe each step in chronological order</focus>
  </for-screen-recordings>

  <for-general-content>
    <focus>Provide a comprehensive summary of the content</focus>
    <focus>Identify key visual elements, scenes, and themes</focus>
    <focus>Note any important context or information presented</focus>
  </for-general-content>
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


class VideoDescription(BaseModel):
    """Unified video description for downstream AI agents.

    This model provides comprehensive video content analysis that works for both
    screen recordings and general video content. The model determines the video
    type and fills in the appropriate fields.
    """

    # Video classification
    video_type: Literal["screen_recording", "general"] = Field(
        description="Type of video: 'screen_recording' for UI/software demos, 'general' for other content"
    )

    # Common fields (always required)
    summary: str = Field(description="Brief description of what the video shows (2-4 sentences)")
    visual_elements: list[str] = Field(
        description="Key visual elements: UI components, objects, people, locations, etc."
    )
    text_content: list[str] = Field(
        description="All visible text in the video: UI labels, buttons, overlays, captions, titles"
    )

    # Screen recording specific (fill when video_type='screen_recording')
    operation_sequence: list[str] | None = Field(
        default=None,
        description="Step-by-step user actions in chronological order",
    )
    application_context: str | None = Field(
        default=None,
        description="Application name, browser, OS, or environment being used",
    )

    # General video specific (fill when video_type='general')
    scenes: list[str] | None = Field(
        default=None,
        description="Distinct scenes or segments in the video",
    )
    themes: list[str] | None = Field(
        default=None,
        description="Identified themes, topics, or messages",
    )

    # Audio content (fill when audio is present)
    audio_transcription: str | None = Field(
        default=None,
        description="Transcription of all spoken words in the video",
    )
    audio_description: str | None = Field(
        default=None,
        description="Description of non-speech audio: music, sound effects, ambient sounds",
    )

    # Meta
    user_intent: str | None = Field(
        default=None,
        description="Inferred goal or purpose of the video/user actions",
    )
    key_observations: list[str] | None = Field(
        default=None,
        description="Notable details: errors, warnings, loading states, unusual behaviors",
    )

    @model_validator(mode="after")
    def validate_screen_recording_fields(self) -> Self:
        """Ensure screen_recording type has required fields for reproducibility."""
        if self.video_type == "screen_recording":
            if not self.operation_sequence:
                raise ValueError(
                    "operation_sequence is required for screen_recording videos "
                    "to ensure interaction traces remain complete"
                )
            if not self.application_context:
                raise ValueError(
                    "application_context is required for screen_recording videos "
                    "to identify the application environment"
                )
        return self

    def to_xml(self) -> str:
        """Serialize the video description to XML format."""
        root = Element("video-description")
        root.set("type", self.video_type)

        self._build_common_elements(root)
        self._build_type_specific_elements(root)
        self._build_audio_elements(root)
        self._build_meta_elements(root)

        return _xml_to_string(root)

    def _build_common_elements(self, root: Element) -> None:
        """Build common XML elements (summary, visual_elements, text_content)."""
        SubElement(root, "summary").text = self.summary

        visual_elem = SubElement(root, "visual-elements")
        for item in self.visual_elements:
            SubElement(visual_elem, "element").text = item

        text_elem = SubElement(root, "text-content")
        for item in self.text_content:
            SubElement(text_elem, "text").text = item

    def _build_type_specific_elements(self, root: Element) -> None:
        """Build video type specific XML elements."""
        if self.video_type == "screen_recording":
            if self.application_context:
                SubElement(root, "application-context").text = self.application_context
            if self.operation_sequence:
                ops_elem = SubElement(root, "operation-sequence")
                for i, step in enumerate(self.operation_sequence, 1):
                    step_elem = SubElement(ops_elem, "step")
                    step_elem.set("index", str(i))
                    step_elem.text = step
        elif self.video_type == "general":
            if self.scenes:
                scenes_elem = SubElement(root, "scenes")
                for i, scene in enumerate(self.scenes, 1):
                    scene_elem = SubElement(scenes_elem, "scene")
                    scene_elem.set("index", str(i))
                    scene_elem.text = scene
            if self.themes:
                themes_elem = SubElement(root, "themes")
                for theme in self.themes:
                    SubElement(themes_elem, "theme").text = theme

    def _build_audio_elements(self, root: Element) -> None:
        """Build audio-related XML elements."""
        if self.audio_transcription:
            SubElement(root, "audio-transcription").text = self.audio_transcription
        if self.audio_description:
            SubElement(root, "audio-description").text = self.audio_description

    def _build_meta_elements(self, root: Element) -> None:
        """Build meta XML elements (user_intent, key_observations)."""
        if self.user_intent:
            SubElement(root, "user-intent").text = self.user_intent
        if self.key_observations:
            obs_elem = SubElement(root, "key-observations")
            for obs in self.key_observations:
                SubElement(obs_elem, "observation").text = obs

    def __repr__(self) -> str:
        return self.to_xml()

    def __str__(self) -> str:
        return self.to_xml()


def get_video_understanding_agent(
    model: str | Model | None = None,
    model_settings: dict[str, Any] | None = None,
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

    return Agent[None, VideoDescription](  # pyright: ignore[reportCallIssue]
        model_instance,
        output_type=VideoDescription,
        system_prompt=system_prompt,
        model_settings=model_settings,  # pyright: ignore[reportArgumentType]
        retries=3,
        output_retries=3,
    )


async def get_video_description(
    video_url: str | None = None,
    video_data: bytes | None = None,
    media_type: str | None = None,
    instruction: str | None = None,
    model: str | Model | None = None,
    model_settings: dict[str, Any] | None = None,
    max_video_size: int = DEFAULT_MAX_VIDEO_SIZE,
) -> tuple[str, RunUsage]:
    """Analyze a video and get a structured description.

    Args:
        video_url: URL of the video to analyze.
        video_data: Raw video bytes to analyze.
        media_type: Optional media type override.
        instruction: Custom instruction for analysis. If None, uses default.
        model: Model string or Model instance.
        model_settings: Optional model settings dict.
        max_video_size: Maximum allowed size for video_data in bytes.

    Returns:
        Tuple of (XML description string, Usage).

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

    return result.output.to_xml(), result.usage()
