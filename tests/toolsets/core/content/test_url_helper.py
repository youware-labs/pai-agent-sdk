"""Tests for URL helper utilities."""

from pai_agent_sdk.toolsets.core.content._url_helper import (
    ContentCategory,
    get_category_from_extension,
    get_category_from_mime_type,
    is_valid_http_url,
)


class TestIsValidHttpUrl:
    """Tests for is_valid_http_url function."""

    def test_valid_http_url(self) -> None:
        """Should accept valid HTTP URLs."""
        assert is_valid_http_url("http://example.com") is True
        assert is_valid_http_url("http://example.com/path") is True
        assert is_valid_http_url("http://example.com/path?query=1") is True

    def test_valid_https_url(self) -> None:
        """Should accept valid HTTPS URLs."""
        assert is_valid_http_url("https://example.com") is True
        assert is_valid_http_url("https://example.com/path") is True
        assert is_valid_http_url("https://example.com/image.png") is True

    def test_invalid_protocol(self) -> None:
        """Should reject non-HTTP/HTTPS protocols."""
        assert is_valid_http_url("ftp://example.com") is False
        assert is_valid_http_url("file:///path/to/file") is False
        assert is_valid_http_url("data:image/png;base64,abc") is False
        assert is_valid_http_url("javascript:alert(1)") is False

    def test_invalid_url(self) -> None:
        """Should reject invalid URLs."""
        assert is_valid_http_url("not-a-url") is False
        assert is_valid_http_url("") is False
        assert is_valid_http_url("http://") is False


class TestGetCategoryFromMimeType:
    """Tests for get_category_from_mime_type function."""

    def test_image_types(self) -> None:
        """Should detect image MIME types."""
        assert get_category_from_mime_type("image/png") == ContentCategory.image
        assert get_category_from_mime_type("image/jpeg") == ContentCategory.image
        assert get_category_from_mime_type("image/gif") == ContentCategory.image
        assert get_category_from_mime_type("image/webp") == ContentCategory.image

    def test_video_types(self) -> None:
        """Should detect video MIME types."""
        assert get_category_from_mime_type("video/mp4") == ContentCategory.video
        assert get_category_from_mime_type("video/webm") == ContentCategory.video

    def test_audio_types(self) -> None:
        """Should detect audio MIME types."""
        assert get_category_from_mime_type("audio/mpeg") == ContentCategory.audio
        assert get_category_from_mime_type("audio/wav") == ContentCategory.audio

    def test_document_types(self) -> None:
        """Should detect document MIME types."""
        assert get_category_from_mime_type("application/pdf") == ContentCategory.document

    def test_text_types(self) -> None:
        """Should detect text MIME types."""
        assert get_category_from_mime_type("text/plain") == ContentCategory.text
        assert get_category_from_mime_type("text/html") == ContentCategory.text
        assert get_category_from_mime_type("application/json") == ContentCategory.text

    def test_mime_with_charset(self) -> None:
        """Should handle MIME types with charset parameter."""
        assert get_category_from_mime_type("text/html; charset=utf-8") == ContentCategory.text
        assert get_category_from_mime_type("application/json; charset=utf-8") == ContentCategory.text

    def test_unknown_types(self) -> None:
        """Should return unknown for unrecognized types."""
        assert get_category_from_mime_type("application/octet-stream") == ContentCategory.unknown

    def test_fallback_to_prefix(self) -> None:
        """Should fallback to prefix matching for unknown subtypes."""
        assert get_category_from_mime_type("image/x-custom") == ContentCategory.image
        assert get_category_from_mime_type("video/x-custom") == ContentCategory.video
        assert get_category_from_mime_type("audio/x-custom") == ContentCategory.audio
        assert get_category_from_mime_type("text/x-custom") == ContentCategory.text


class TestGetCategoryFromExtension:
    """Tests for get_category_from_extension function."""

    def test_image_extensions(self) -> None:
        """Should detect image file extensions."""
        assert get_category_from_extension("https://example.com/image.png") == ContentCategory.image
        assert get_category_from_extension("https://example.com/image.jpg") == ContentCategory.image
        assert get_category_from_extension("https://example.com/image.jpeg") == ContentCategory.image
        assert get_category_from_extension("https://example.com/image.gif") == ContentCategory.image
        assert get_category_from_extension("https://example.com/image.webp") == ContentCategory.image

    def test_video_extensions(self) -> None:
        """Should detect video file extensions."""
        assert get_category_from_extension("https://example.com/video.mp4") == ContentCategory.video
        assert get_category_from_extension("https://example.com/video.webm") == ContentCategory.video
        assert get_category_from_extension("https://example.com/video.mov") == ContentCategory.video

    def test_audio_extensions(self) -> None:
        """Should detect audio file extensions."""
        assert get_category_from_extension("https://example.com/audio.mp3") == ContentCategory.audio
        assert get_category_from_extension("https://example.com/audio.wav") == ContentCategory.audio

    def test_document_extensions(self) -> None:
        """Should detect document file extensions."""
        assert get_category_from_extension("https://example.com/doc.pdf") == ContentCategory.document

    def test_text_extensions(self) -> None:
        """Should detect text file extensions."""
        assert get_category_from_extension("https://example.com/file.txt") == ContentCategory.text
        assert get_category_from_extension("https://example.com/page.html") == ContentCategory.text
        assert get_category_from_extension("https://example.com/data.json") == ContentCategory.text

    def test_case_insensitive(self) -> None:
        """Should be case insensitive."""
        assert get_category_from_extension("https://example.com/IMAGE.PNG") == ContentCategory.image
        assert get_category_from_extension("https://example.com/VIDEO.MP4") == ContentCategory.video

    def test_unknown_extensions(self) -> None:
        """Should return unknown for unrecognized extensions."""
        assert get_category_from_extension("https://example.com/file.xyz") == ContentCategory.unknown
        assert get_category_from_extension("https://example.com/noext") == ContentCategory.unknown


async def test_detect_content_category_fallback() -> None:
    """Should fallback to extension when HEAD request fails."""
    from pai_agent_sdk.toolsets.core.content._url_helper import detect_content_category

    # Use a non-existent URL that will fail HEAD request
    result = await detect_content_category("https://nonexistent-domain-12345.com/image.png", timeout=1.0)
    # Should fallback to extension-based detection
    assert result == ContentCategory.image
