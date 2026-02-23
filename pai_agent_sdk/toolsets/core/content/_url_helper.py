"""URL helper utilities for the load_media_url tool.

This module provides helper functions to validate URLs and detect content types
for the LoadMediaUrlTool.
"""

from enum import StrEnum
from urllib.parse import urlparse

import httpx


class ContentCategory(StrEnum):
    """Categories of content that can be loaded from a URL."""

    image = "image"
    """Image content (JPEG, PNG, GIF, WebP, etc.)."""

    video = "video"
    """Video content (MP4, WebM, etc.)."""

    audio = "audio"
    """Audio content (MP3, WAV, etc.)."""

    document = "document"
    """Document content (PDF)."""

    text = "text"
    """Text content (HTML, JSON, plain text, etc.)."""

    unknown = "unknown"
    """Unknown content type."""


# MIME type to content category mapping
_MIME_CATEGORY_MAP: dict[str, ContentCategory] = {
    # Images
    "image/jpeg": ContentCategory.image,
    "image/png": ContentCategory.image,
    "image/gif": ContentCategory.image,
    "image/webp": ContentCategory.image,
    "image/svg+xml": ContentCategory.image,
    "image/bmp": ContentCategory.image,
    "image/tiff": ContentCategory.image,
    # Videos
    "video/mp4": ContentCategory.video,
    "video/webm": ContentCategory.video,
    "video/ogg": ContentCategory.video,
    "video/quicktime": ContentCategory.video,
    "video/x-msvideo": ContentCategory.video,
    # Audio
    "audio/mpeg": ContentCategory.audio,
    "audio/wav": ContentCategory.audio,
    "audio/ogg": ContentCategory.audio,
    "audio/webm": ContentCategory.audio,
    "audio/flac": ContentCategory.audio,
    # Documents
    "application/pdf": ContentCategory.document,
    # Text
    "text/plain": ContentCategory.text,
    "text/html": ContentCategory.text,
    "text/css": ContentCategory.text,
    "text/javascript": ContentCategory.text,
    "application/json": ContentCategory.text,
    "application/xml": ContentCategory.text,
    "text/xml": ContentCategory.text,
    "text/markdown": ContentCategory.text,
}

# File extension to content category mapping (fallback)
_EXTENSION_CATEGORY_MAP: dict[str, ContentCategory] = {
    # Images
    ".jpg": ContentCategory.image,
    ".jpeg": ContentCategory.image,
    ".png": ContentCategory.image,
    ".gif": ContentCategory.image,
    ".webp": ContentCategory.image,
    ".svg": ContentCategory.image,
    ".bmp": ContentCategory.image,
    ".tiff": ContentCategory.image,
    ".tif": ContentCategory.image,
    # Videos
    ".mp4": ContentCategory.video,
    ".webm": ContentCategory.video,
    ".ogv": ContentCategory.video,
    ".mov": ContentCategory.video,
    ".avi": ContentCategory.video,
    # Audio
    ".mp3": ContentCategory.audio,
    ".wav": ContentCategory.audio,
    ".ogg": ContentCategory.audio,
    ".flac": ContentCategory.audio,
    ".m4a": ContentCategory.audio,
    # Documents
    ".pdf": ContentCategory.document,
    # Text
    ".txt": ContentCategory.text,
    ".html": ContentCategory.text,
    ".htm": ContentCategory.text,
    ".css": ContentCategory.text,
    ".js": ContentCategory.text,
    ".json": ContentCategory.text,
    ".xml": ContentCategory.text,
    ".md": ContentCategory.text,
    ".csv": ContentCategory.text,
}


def is_valid_http_url(url: str) -> bool:
    """Check if the URL is a valid HTTP or HTTPS URL.

    Args:
        url: The URL to validate.

    Returns:
        True if the URL is valid HTTP/HTTPS, False otherwise.
    """
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def get_category_from_mime_type(mime_type: str) -> ContentCategory:
    """Get content category from MIME type.

    Args:
        mime_type: The MIME type string (e.g., "image/png").

    Returns:
        The corresponding ContentCategory.
    """
    # Normalize: take only the main type (ignore parameters like charset)
    main_type = mime_type.split(";")[0].strip().lower()

    # Direct match
    if main_type in _MIME_CATEGORY_MAP:
        return _MIME_CATEGORY_MAP[main_type]

    # Fallback to prefix matching
    if main_type.startswith("image/"):
        return ContentCategory.image
    if main_type.startswith("video/"):
        return ContentCategory.video
    if main_type.startswith("audio/"):
        return ContentCategory.audio
    if main_type.startswith("text/"):
        return ContentCategory.text

    return ContentCategory.unknown


def get_category_from_extension(url: str) -> ContentCategory:
    """Get content category from URL file extension.

    Args:
        url: The URL to extract extension from.

    Returns:
        The corresponding ContentCategory.
    """
    try:
        parsed = urlparse(url)
        path = parsed.path.lower()
        for ext, category in _EXTENSION_CATEGORY_MAP.items():
            if path.endswith(ext):
                return category
    except Exception:  # noqa: S110
        pass
    return ContentCategory.unknown


async def detect_content_category(url: str, timeout: float = 10.0) -> ContentCategory:
    """Detect content category from a URL using HTTP HEAD request.

    This function first tries to get the Content-Type from an HTTP HEAD request.
    If that fails or returns unknown, it falls back to extension-based detection.

    Args:
        url: The URL to detect content type for.
        timeout: Request timeout in seconds.

    Returns:
        The detected ContentCategory.
    """
    # First try HEAD request to get Content-Type
    try:
        async with httpx.AsyncClient() as client:
            response = await client.head(url, timeout=timeout, follow_redirects=True)
            content_type = response.headers.get("content-type", "")
            if content_type:
                category = get_category_from_mime_type(content_type)
                if category != ContentCategory.unknown:
                    return category
    except Exception:  # noqa: S110
        pass

    # Fallback to extension-based detection
    return get_category_from_extension(url)
