"""Tests for pai_agent_sdk.filters.image module."""

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
    VideoUrl,
)

from pai_agent_sdk.context import AgentContext, ModelConfig
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.filters.image import drop_extra_images, drop_extra_videos, drop_gif_images


def _create_valid_image_bytes() -> bytes:
    """Create a valid PNG image bytes."""
    img = Image.new("RGB", (100, 100), color="red")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _create_broken_image_bytes() -> bytes:
    """Create broken image bytes."""
    return b"not a valid image"


# Tests for drop_extra_images


async def test_drop_extra_images_no_model_config(tmp_path: Path) -> None:
    """Should use default max_images=20 when no model config is set."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            image_url = ImageUrl(url="https://example.com/image.png")
            request = ModelRequest(parts=[UserPromptPart(content=[image_url])])
            history = [request]

            result = drop_extra_images(mock_ctx, history)

            assert result == history
            # Image should be kept (within default limit of 20)
            content = request.parts[0].content  # type: ignore[union-attr]
            assert len(content) == 1
            assert content[0] == image_url


async def test_drop_extra_images_within_limit(tmp_path: Path) -> None:
    """Should keep all images when within the limit."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(max_images=5),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            images = [ImageUrl(url=f"https://example.com/image{i}.png") for i in range(3)]
            request = ModelRequest(parts=[UserPromptPart(content=images)])
            history = [request]

            drop_extra_images(mock_ctx, history)

            content = request.parts[0].content  # type: ignore[union-attr]
            assert len(content) == 3
            for i, item in enumerate(content):
                assert isinstance(item, ImageUrl)
                assert item.url == f"https://example.com/image{i}.png"


async def test_drop_extra_images_exceeds_limit(tmp_path: Path) -> None:
    """Should drop extra images when exceeding the limit."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(max_images=2),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            # Create 4 images - should keep latest 2
            images = [ImageUrl(url=f"https://example.com/image{i}.png") for i in range(4)]
            request = ModelRequest(parts=[UserPromptPart(content=images)])
            history = [request]

            drop_extra_images(mock_ctx, history)

            content = request.parts[0].content  # type: ignore[union-attr]
            assert len(content) == 4
            # First 2 images should be dropped (replaced with system reminder)
            assert isinstance(content[0], str)
            assert "max_images=2" in content[0]
            assert isinstance(content[1], str)
            assert "max_images=2" in content[1]
            # Last 2 images should be kept
            assert isinstance(content[2], ImageUrl)
            assert isinstance(content[3], ImageUrl)


async def test_drop_extra_images_keeps_latest_across_messages(tmp_path: Path) -> None:
    """Should keep the most recent images across multiple messages."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(max_images=2),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            # Old message with 2 images
            old_images = [ImageUrl(url=f"https://example.com/old{i}.png") for i in range(2)]
            old_request = ModelRequest(parts=[UserPromptPart(content=old_images)])

            # New message with 2 images
            new_images = [ImageUrl(url=f"https://example.com/new{i}.png") for i in range(2)]
            new_request = ModelRequest(parts=[UserPromptPart(content=new_images)])

            history = [old_request, new_request]

            drop_extra_images(mock_ctx, history)

            # Old images should be dropped
            old_content = old_request.parts[0].content  # type: ignore[union-attr]
            assert isinstance(old_content[0], str)
            assert isinstance(old_content[1], str)

            # New images should be kept
            new_content = new_request.parts[0].content  # type: ignore[union-attr]
            assert isinstance(new_content[0], ImageUrl)
            assert isinstance(new_content[1], ImageUrl)


async def test_drop_extra_images_validates_binary_content(tmp_path: Path) -> None:
    """Should validate and remove broken binary image content."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(max_images=10),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            valid_image = BinaryContent(data=_create_valid_image_bytes(), media_type="image/png")
            broken_image = BinaryContent(data=_create_broken_image_bytes(), media_type="image/png")
            request = ModelRequest(parts=[UserPromptPart(content=[valid_image, broken_image])])
            history = [request]

            drop_extra_images(mock_ctx, history)

            content = request.parts[0].content  # type: ignore[union-attr]
            assert len(content) == 2
            # Valid image should be kept
            assert isinstance(content[0], BinaryContent)
            # Broken image should be replaced with system reminder
            assert isinstance(content[1], str)
            assert "broken or corrupted" in content[1]


async def test_drop_extra_images_skips_model_response(tmp_path: Path) -> None:
    """Should skip ModelResponse messages."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(max_images=1),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            response = ModelResponse(parts=[TextPart(content="Response")])
            image_url = ImageUrl(url="https://example.com/image.png")
            request = ModelRequest(parts=[UserPromptPart(content=[image_url])])
            history = [response, request]

            drop_extra_images(mock_ctx, history)

            # ModelResponse should be unchanged
            assert history[0] == response
            # Request image should be kept
            content = request.parts[0].content  # type: ignore[union-attr]
            assert isinstance(content[0], ImageUrl)


async def test_drop_extra_images_preserves_string_content(tmp_path: Path) -> None:
    """Should preserve string content without modification."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(max_images=1),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            request = ModelRequest(parts=[UserPromptPart(content="Just text")])
            history = [request]

            drop_extra_images(mock_ctx, history)

            content = request.parts[0].content  # type: ignore[union-attr]
            assert content == "Just text"


# Tests for drop_extra_videos


async def test_drop_extra_videos_no_model_config(tmp_path: Path) -> None:
    """Should use default max_videos=1 when no model config is set."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            video_url = VideoUrl(url="https://example.com/video.mp4")
            request = ModelRequest(parts=[UserPromptPart(content=[video_url])])
            history = [request]

            result = drop_extra_videos(mock_ctx, history)

            assert result == history
            # Video should be kept (within default limit of 1)
            content = request.parts[0].content  # type: ignore[union-attr]
            assert len(content) == 1
            assert content[0] == video_url


async def test_drop_extra_videos_within_limit(tmp_path: Path) -> None:
    """Should keep all videos when within the limit."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(max_videos=3),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            videos = [VideoUrl(url=f"https://example.com/video{i}.mp4") for i in range(2)]
            request = ModelRequest(parts=[UserPromptPart(content=videos)])
            history = [request]

            drop_extra_videos(mock_ctx, history)

            content = request.parts[0].content  # type: ignore[union-attr]
            assert len(content) == 2
            for item in content:
                assert isinstance(item, VideoUrl)


async def test_drop_extra_videos_exceeds_limit(tmp_path: Path) -> None:
    """Should drop extra videos when exceeding the limit."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(max_videos=1),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            # Create 3 videos - should keep latest 1
            videos = [VideoUrl(url=f"https://example.com/video{i}.mp4") for i in range(3)]
            request = ModelRequest(parts=[UserPromptPart(content=videos)])
            history = [request]

            drop_extra_videos(mock_ctx, history)

            content = request.parts[0].content  # type: ignore[union-attr]
            assert len(content) == 3
            # First 2 videos should be dropped (replaced with system reminder)
            assert isinstance(content[0], str)
            assert "max_videos=1" in content[0]
            assert isinstance(content[1], str)
            assert "max_videos=1" in content[1]
            # Last video should be kept
            assert isinstance(content[2], VideoUrl)


async def test_drop_extra_videos_keeps_latest_across_messages(tmp_path: Path) -> None:
    """Should keep the most recent videos across multiple messages."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(max_videos=1),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            # Old message with 1 video
            old_video = VideoUrl(url="https://example.com/old.mp4")
            old_request = ModelRequest(parts=[UserPromptPart(content=[old_video])])

            # New message with 1 video
            new_video = VideoUrl(url="https://example.com/new.mp4")
            new_request = ModelRequest(parts=[UserPromptPart(content=[new_video])])

            history = [old_request, new_request]

            drop_extra_videos(mock_ctx, history)

            # Old video should be dropped
            old_content = old_request.parts[0].content  # type: ignore[union-attr]
            assert isinstance(old_content[0], str)
            assert "max_videos=1" in old_content[0]

            # New video should be kept
            new_content = new_request.parts[0].content  # type: ignore[union-attr]
            assert isinstance(new_content[0], VideoUrl)


async def test_drop_extra_videos_handles_binary_video_content(tmp_path: Path) -> None:
    """Should handle binary video content."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(max_videos=1),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            binary_video = BinaryContent(data=b"video data", media_type="video/mp4")
            video_url = VideoUrl(url="https://example.com/video.mp4")
            request = ModelRequest(parts=[UserPromptPart(content=[binary_video, video_url])])
            history = [request]

            drop_extra_videos(mock_ctx, history)

            content = request.parts[0].content  # type: ignore[union-attr]
            assert len(content) == 2
            # First video should be dropped
            assert isinstance(content[0], str)
            assert "max_videos=1" in content[0]
            # Latest video should be kept
            assert isinstance(content[1], VideoUrl)


async def test_drop_extra_videos_skips_model_response(tmp_path: Path) -> None:
    """Should skip ModelResponse messages."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(max_videos=1),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            response = ModelResponse(parts=[TextPart(content="Response")])
            video_url = VideoUrl(url="https://example.com/video.mp4")
            request = ModelRequest(parts=[UserPromptPart(content=[video_url])])
            history = [response, request]

            drop_extra_videos(mock_ctx, history)

            # ModelResponse should be unchanged
            assert history[0] == response
            # Request video should be kept
            content = request.parts[0].content  # type: ignore[union-attr]
            assert isinstance(content[0], VideoUrl)


async def test_drop_extra_videos_preserves_string_content(tmp_path: Path) -> None:
    """Should preserve string content without modification."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(max_videos=1),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            request = ModelRequest(parts=[UserPromptPart(content="Just text")])
            history = [request]

            drop_extra_videos(mock_ctx, history)

            content = request.parts[0].content  # type: ignore[union-attr]
            assert content == "Just text"


# Tests for drop_gif_images


async def test_drop_gif_images_support_gif_true(tmp_path: Path) -> None:
    """Should keep GIF images when support_gif is True (default)."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(support_gif=True),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            gif_content = BinaryContent(data=b"GIF89a...", media_type="image/gif")
            request = ModelRequest(parts=[UserPromptPart(content=[gif_content])])
            history = [request]

            result = drop_gif_images(mock_ctx, history)

            assert result == history
            content = request.parts[0].content  # type: ignore[union-attr]
            assert len(content) == 1
            assert isinstance(content[0], BinaryContent)
            assert content[0].media_type == "image/gif"


async def test_drop_gif_images_support_gif_false(tmp_path: Path) -> None:
    """Should drop GIF images when support_gif is False."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(support_gif=False),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            gif_content = BinaryContent(data=b"GIF89a...", media_type="image/gif")
            request = ModelRequest(parts=[UserPromptPart(content=[gif_content])])
            history = [request]

            drop_gif_images(mock_ctx, history)

            content = request.parts[0].content  # type: ignore[union-attr]
            assert len(content) == 1
            assert isinstance(content[0], str)
            assert "GIF image has been removed" in content[0]


async def test_drop_gif_images_keeps_other_images(tmp_path: Path) -> None:
    """Should keep non-GIF images when support_gif is False."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(support_gif=False),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            png_content = BinaryContent(data=_create_valid_image_bytes(), media_type="image/png")
            gif_content = BinaryContent(data=b"GIF89a...", media_type="image/gif")
            jpeg_content = BinaryContent(data=b"jpeg data", media_type="image/jpeg")
            request = ModelRequest(parts=[UserPromptPart(content=[png_content, gif_content, jpeg_content])])
            history = [request]

            drop_gif_images(mock_ctx, history)

            content = request.parts[0].content  # type: ignore[union-attr]
            assert len(content) == 3
            # PNG should be kept
            assert isinstance(content[0], BinaryContent)
            assert content[0].media_type == "image/png"
            # GIF should be dropped
            assert isinstance(content[1], str)
            assert "GIF image has been removed" in content[1]
            # JPEG should be kept
            assert isinstance(content[2], BinaryContent)
            assert content[2].media_type == "image/jpeg"


async def test_drop_gif_images_no_model_config(tmp_path: Path) -> None:
    """Should keep GIF images when no model config is set (default support_gif=True)."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            gif_content = BinaryContent(data=b"GIF89a...", media_type="image/gif")
            request = ModelRequest(parts=[UserPromptPart(content=[gif_content])])
            history = [request]

            result = drop_gif_images(mock_ctx, history)

            assert result == history
            content = request.parts[0].content  # type: ignore[union-attr]
            assert len(content) == 1
            assert isinstance(content[0], BinaryContent)


async def test_drop_gif_images_skips_model_response(tmp_path: Path) -> None:
    """Should skip ModelResponse messages."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(support_gif=False),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            response = ModelResponse(parts=[TextPart(content="Response")])
            gif_content = BinaryContent(data=b"GIF89a...", media_type="image/gif")
            request = ModelRequest(parts=[UserPromptPart(content=[gif_content])])
            history = [response, request]

            drop_gif_images(mock_ctx, history)

            # ModelResponse should be unchanged
            assert history[0] == response
            # Request GIF should be dropped
            content = request.parts[0].content  # type: ignore[union-attr]
            assert isinstance(content[0], str)
            assert "GIF image has been removed" in content[0]


async def test_drop_gif_images_preserves_string_content(tmp_path: Path) -> None:
    """Should preserve string content without modification."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            model_cfg=ModelConfig(support_gif=False),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            request = ModelRequest(parts=[UserPromptPart(content="Just text")])
            history = [request]

            drop_gif_images(mock_ctx, history)

            content = request.parts[0].content  # type: ignore[union-attr]
            assert content == "Just text"
