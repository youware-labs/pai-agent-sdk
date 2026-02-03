"""Tests for media upload filter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelRequest,
    UserPromptPart,
    VideoUrl,
)

from pai_agent_sdk.context import AgentContext, ModelCapability, ModelConfig
from pai_agent_sdk.filters.media_upload import create_media_upload_filter
from pai_agent_sdk.media import MediaUploader


class MockUploader:
    """Mock uploader implementing MediaUploader protocol."""

    def __init__(self, base_url: str = "https://cdn.example.com"):
        self.base_url = base_url
        self.upload_count = 0

    async def upload(self, data: bytes, media_type: str) -> str:
        self.upload_count += 1
        ext = media_type.split("/")[-1]
        return f"{self.base_url}/uploaded_{self.upload_count}.{ext}"


def test_mock_uploader_implements_protocol():
    """Test that MockUploader implements MediaUploader protocol."""
    uploader = MockUploader()
    assert isinstance(uploader, MediaUploader)


@pytest.fixture
def mock_ctx_with_image_url():
    """Create mock context with IMAGE_URL capability."""
    ctx = MagicMock()
    ctx.deps = MagicMock(spec=AgentContext)
    ctx.deps.model_cfg = ModelConfig(capabilities={ModelCapability.image_url})
    return ctx


@pytest.fixture
def mock_ctx_with_video_url():
    """Create mock context with VIDEO_URL capability."""
    ctx = MagicMock()
    ctx.deps = MagicMock(spec=AgentContext)
    ctx.deps.model_cfg = ModelConfig(capabilities={ModelCapability.video_url})
    return ctx


@pytest.fixture
def mock_ctx_with_both():
    """Create mock context with both IMAGE_URL and VIDEO_URL capabilities."""
    ctx = MagicMock()
    ctx.deps = MagicMock(spec=AgentContext)
    ctx.deps.model_cfg = ModelConfig(capabilities={ModelCapability.image_url, ModelCapability.video_url})
    return ctx


@pytest.fixture
def mock_ctx_without_url_caps():
    """Create mock context without URL capabilities."""
    ctx = MagicMock()
    ctx.deps = MagicMock(spec=AgentContext)
    ctx.deps.model_cfg = ModelConfig(capabilities={ModelCapability.vision})
    return ctx


@pytest.mark.asyncio
async def test_upload_image_with_capability(mock_ctx_with_image_url):
    """Test that images are uploaded when IMAGE_URL capability is present."""
    uploader = MockUploader()
    filter_fn = create_media_upload_filter(uploader)

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        BinaryContent(data=b"image_data", media_type="image/png"),
                        "Describe this image",
                    ]
                )
            ]
        )
    ]

    result = await filter_fn(mock_ctx_with_image_url, messages)

    assert uploader.upload_count == 1
    part = result[0].parts[0]
    assert isinstance(part, UserPromptPart)
    content = part.content
    assert isinstance(content[0], ImageUrl)
    assert content[0].url == "https://cdn.example.com/uploaded_1.png"
    assert content[1] == "Describe this image"


@pytest.mark.asyncio
async def test_upload_video_with_capability(mock_ctx_with_video_url):
    """Test that videos are uploaded when VIDEO_URL capability is present."""
    uploader = MockUploader()
    filter_fn = create_media_upload_filter(uploader)

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        BinaryContent(data=b"video_data", media_type="video/mp4"),
                    ]
                )
            ]
        )
    ]

    result = await filter_fn(mock_ctx_with_video_url, messages)

    assert uploader.upload_count == 1
    part = result[0].parts[0]
    assert isinstance(part, UserPromptPart)
    content = part.content
    assert isinstance(content[0], VideoUrl)
    assert content[0].url == "https://cdn.example.com/uploaded_1.mp4"


@pytest.mark.asyncio
async def test_no_upload_without_capability(mock_ctx_without_url_caps):
    """Test that media is not uploaded without URL capabilities."""
    uploader = MockUploader()
    filter_fn = create_media_upload_filter(uploader)

    original_content = BinaryContent(data=b"image_data", media_type="image/png")
    messages = [ModelRequest(parts=[UserPromptPart(content=[original_content])])]

    result = await filter_fn(mock_ctx_without_url_caps, messages)

    assert uploader.upload_count == 0
    part = result[0].parts[0]
    assert isinstance(part, UserPromptPart)
    # Content should be unchanged
    assert part.content[0] is original_content


@pytest.mark.asyncio
async def test_upload_both_image_and_video(mock_ctx_with_both):
    """Test uploading both images and videos."""
    uploader = MockUploader()
    filter_fn = create_media_upload_filter(uploader)

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        BinaryContent(data=b"image_data", media_type="image/png"),
                        BinaryContent(data=b"video_data", media_type="video/mp4"),
                    ]
                )
            ]
        )
    ]

    result = await filter_fn(mock_ctx_with_both, messages)

    assert uploader.upload_count == 2
    part = result[0].parts[0]
    assert isinstance(part, UserPromptPart)
    content = part.content
    assert isinstance(content[0], ImageUrl)
    assert isinstance(content[1], VideoUrl)


@pytest.mark.asyncio
async def test_upload_images_disabled(mock_ctx_with_both):
    """Test that upload_images=False disables image upload."""
    uploader = MockUploader()
    filter_fn = create_media_upload_filter(uploader, upload_images=False)

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        BinaryContent(data=b"image_data", media_type="image/png"),
                        BinaryContent(data=b"video_data", media_type="video/mp4"),
                    ]
                )
            ]
        )
    ]

    result = await filter_fn(mock_ctx_with_both, messages)

    assert uploader.upload_count == 1  # Only video uploaded
    part = result[0].parts[0]
    content = part.content
    assert isinstance(content[0], BinaryContent)  # Image unchanged
    assert isinstance(content[1], VideoUrl)  # Video uploaded


@pytest.mark.asyncio
async def test_upload_videos_disabled(mock_ctx_with_both):
    """Test that upload_videos=False disables video upload."""
    uploader = MockUploader()
    filter_fn = create_media_upload_filter(uploader, upload_videos=False)

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        BinaryContent(data=b"image_data", media_type="image/png"),
                        BinaryContent(data=b"video_data", media_type="video/mp4"),
                    ]
                )
            ]
        )
    ]

    result = await filter_fn(mock_ctx_with_both, messages)

    assert uploader.upload_count == 1  # Only image uploaded
    part = result[0].parts[0]
    content = part.content
    assert isinstance(content[0], ImageUrl)  # Image uploaded
    assert isinstance(content[1], BinaryContent)  # Video unchanged


@pytest.mark.asyncio
async def test_upload_failure_keeps_binary(mock_ctx_with_image_url):
    """Test that upload failure preserves original binary content."""
    uploader = MagicMock()
    uploader.upload = AsyncMock(side_effect=Exception("Upload failed"))

    filter_fn = create_media_upload_filter(uploader)

    original_content = BinaryContent(data=b"image_data", media_type="image/png")
    messages = [ModelRequest(parts=[UserPromptPart(content=[original_content])])]

    result = await filter_fn(mock_ctx_with_image_url, messages)

    # Content should be unchanged after failure
    part = result[0].parts[0]
    assert isinstance(part, UserPromptPart)
    assert part.content[0] is original_content


@pytest.mark.asyncio
async def test_skip_already_url_content(mock_ctx_with_image_url):
    """Test that ImageUrl content is not processed."""
    uploader = MockUploader()
    filter_fn = create_media_upload_filter(uploader)

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        ImageUrl(url="https://existing.com/image.png"),
                    ]
                )
            ]
        )
    ]

    result = await filter_fn(mock_ctx_with_image_url, messages)

    assert uploader.upload_count == 0
    part = result[0].parts[0]
    content = part.content
    assert isinstance(content[0], ImageUrl)
    assert content[0].url == "https://existing.com/image.png"
