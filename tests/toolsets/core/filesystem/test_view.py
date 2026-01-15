"""Tests for pai_agent_sdk.toolsets.core.filesystem.view module."""

from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

from inline_snapshot import snapshot
from pydantic_ai import BinaryContent, RunContext, ToolReturn

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.toolsets.core.filesystem.view import (
    IMAGE_EXTENSIONS,
    MEDIA_TYPE_MAP,
    SUPPORTED_IMAGE_MEDIA_TYPES,
    VIDEO_EXTENSIONS,
    ViewTool,
)


def test_view_tool_attributes(agent_context: AgentContext) -> None:
    """Should have correct name and description."""
    assert ViewTool.name == "view"
    assert "Read files" in ViewTool.description
    tool = ViewTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    instruction = tool.get_instruction(mock_run_ctx)
    assert instruction is not None


def test_view_tool_initialization(agent_context: AgentContext) -> None:
    """Should initialize with context."""
    tool = ViewTool()
    assert tool.name == "view"


def test_view_tool_is_available(agent_context: AgentContext, mock_run_ctx) -> None:
    """Should be available by default."""
    tool = ViewTool()
    assert tool.is_available(mock_run_ctx) is True


def test_is_image_file(agent_context: AgentContext) -> None:
    """Should correctly identify image files."""
    tool = ViewTool()
    for ext in IMAGE_EXTENSIONS:
        assert tool._is_image_file(f"test{ext}") is True
        assert tool._is_image_file(f"test{ext.upper()}") is True
    assert tool._is_image_file("test.txt") is False
    assert tool._is_image_file("test.py") is False


def test_is_video_file(agent_context: AgentContext) -> None:
    """Should correctly identify video files."""
    tool = ViewTool()
    for ext in VIDEO_EXTENSIONS:
        assert tool._is_video_file(f"test{ext}") is True
        assert tool._is_video_file(f"test{ext.upper()}") is True
    assert tool._is_video_file("test.txt") is False
    assert tool._is_video_file("test.png") is False


def test_get_media_type(agent_context: AgentContext) -> None:
    """Should return correct media type for extensions."""
    tool = ViewTool()
    for ext, expected in MEDIA_TYPE_MAP.items():
        assert tool._get_media_type(f"test{ext}") == expected
    assert tool._get_media_type("test.unknown") == "application/octet-stream"


async def test_view_text_file_simple(tmp_path: Path) -> None:
    """Should read text file and return content string when no truncation."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ViewTool()

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!\nLine 2\nLine 3")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="test.txt")
        assert result == "Hello, World!\nLine 2\nLine 3"


async def test_view_text_file_with_offset(tmp_path: Path) -> None:
    """Should read text file with line offset and return metadata."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ViewTool()

        # Create test file with multiple lines
        lines = [f"Line {i}" for i in range(10)]
        test_file = tmp_path / "test.txt"
        test_file.write_text("\n".join(lines))

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="test.txt", line_offset=5)
        assert isinstance(result, dict)
        assert "content" in result
        assert "metadata" in result
        assert "Line 5" in result["content"]
        assert result["metadata"]["current_segment"]["start_line"] == 6


async def test_view_text_file_with_limit(tmp_path: Path) -> None:
    """Should truncate content when exceeding line limit."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ViewTool()

        # Create file with many lines
        lines = [f"Line {i}" for i in range(100)]
        test_file = tmp_path / "test.txt"
        test_file.write_text("\n".join(lines))

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="test.txt", line_limit=10)
        assert isinstance(result, dict)
        assert result["metadata"]["current_segment"]["lines_to_show"] == 10
        assert result["metadata"]["current_segment"]["has_more_content"] is True


async def test_view_text_file_line_truncation(tmp_path: Path) -> None:
    """Should truncate long lines."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ViewTool()

        # Create file with long line
        long_line = "A" * 3000
        test_file = tmp_path / "test.txt"
        test_file.write_text(long_line)

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="test.txt", max_line_length=100)
        assert isinstance(result, dict)
        assert "(line truncated)" in result["content"]
        assert result["metadata"]["truncation_info"]["lines_truncated"] is True


async def test_view_file_not_found(tmp_path: Path) -> None:
    """Should return error message when file not found."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ViewTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="nonexistent.txt")
        assert result == snapshot("Error: File not found: nonexistent.txt")


async def test_view_directory_error(tmp_path: Path) -> None:
    """Should return error when path is a directory."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ViewTool()

        # Create a directory
        test_dir = tmp_path / "testdir"
        test_dir.mkdir()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="testdir")
        assert result == snapshot("Error: Path is a directory, not a file: testdir")


async def test_view_image_file(tmp_path: Path) -> None:
    """Should return ToolReturn with BinaryContent for image files."""
    from pai_agent_sdk.context import ModelCapability, ModelConfig

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        # Create context with vision capability
        ctx = await stack.enter_async_context(
            AgentContext(
                env=env,
                model_cfg=ModelConfig(capabilities={ModelCapability.vision}),
            )
        )
        tool = ViewTool()

        # Create a minimal PNG file (1x1 transparent pixel)
        png_data = bytes([
            0x89,
            0x50,
            0x4E,
            0x47,
            0x0D,
            0x0A,
            0x1A,
            0x0A,
            0x00,
            0x00,
            0x00,
            0x0D,
            0x49,
            0x48,
            0x44,
            0x52,
            0x00,
            0x00,
            0x00,
            0x01,
            0x00,
            0x00,
            0x00,
            0x01,
            0x08,
            0x06,
            0x00,
            0x00,
            0x00,
            0x1F,
            0x15,
            0xC4,
            0x89,
            0x00,
            0x00,
            0x00,
            0x0A,
            0x49,
            0x44,
            0x41,
            0x54,
            0x78,
            0x9C,
            0x63,
            0x00,
            0x01,
            0x00,
            0x00,
            0x05,
            0x00,
            0x01,
            0x0D,
            0x0A,
            0x2D,
            0xB4,
            0x00,
            0x00,
            0x00,
            0x00,
            0x49,
            0x45,
            0x4E,
            0x44,
            0xAE,
            0x42,
            0x60,
            0x82,
        ])
        test_file = tmp_path / "test.png"
        test_file.write_bytes(png_data)

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="test.png")
        assert isinstance(result, ToolReturn)
        assert "image is attached" in result.return_value
        assert len(result.content) == 1
        assert isinstance(result.content[0], BinaryContent)
        assert result.content[0].media_type == "image/png"


async def test_view_video_file_with_video_model(tmp_path: Path) -> None:
    """Should return video content when model supports video."""
    from pai_agent_sdk.context import ModelCapability, ModelConfig

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        # Create context with video_understanding capability
        ctx = await stack.enter_async_context(
            AgentContext(
                env=env,
                model_cfg=ModelConfig(capabilities={ModelCapability.video_understanding}),
            )
        )
        tool = ViewTool()

        # Create a minimal video file
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake video data")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx
        mock_run_ctx.tool_call_id = "test-id"

        result = await tool.call(mock_run_ctx, file_path="test.mp4")

        assert isinstance(result, ToolReturn)
        assert "video is attached" in result.return_value
        assert len(result.content) == 1
        assert result.content[0].media_type == "video/mp4"


async def test_view_video_file_fallback_to_image_understanding(tmp_path: Path) -> None:
    """Should fallback to image understanding when model doesn't support video."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ViewTool()

        # Create a minimal video file
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake video data")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx
        mock_run_ctx.tool_call_id = "test-id"

        # Mock _read_video_with_fallback to simulate fallback behavior
        async def mock_fallback(path, file_path, run_ctx):
            return "Video description (via image analysis):\nThis video shows a test scene."

        with patch.object(tool, "_read_video_with_fallback", side_effect=mock_fallback):
            result = await tool.call(mock_run_ctx, file_path="test.mp4")
            assert "Video description" in result
            assert "test scene" in result


async def test_view_video_fallback_failure(tmp_path: Path) -> None:
    """Should return error message when video fallback fails."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = ViewTool()

        # Create a minimal video file
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake video data")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx
        mock_run_ctx.tool_call_id = "test-id"

        # Mock _read_video_with_fallback to simulate fallback failure
        async def mock_fallback_failure(path, file_path, run_ctx):
            return f"Video file: {file_path}. Model does not support video understanding and fallback analysis failed."

        with patch.object(tool, "_read_video_with_fallback", side_effect=mock_fallback_failure):
            result = await tool.call(mock_run_ctx, file_path="test.mp4")
            assert "does not support video understanding" in result


async def test_view_webm_video(tmp_path: Path) -> None:
    """Should handle webm video with correct media type."""
    from pai_agent_sdk.context import ModelCapability, ModelConfig

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        # Create context with video_understanding capability
        ctx = await stack.enter_async_context(
            AgentContext(
                env=env,
                model_cfg=ModelConfig(capabilities={ModelCapability.video_understanding}),
            )
        )
        tool = ViewTool()

        test_file = tmp_path / "test.webm"
        test_file.write_bytes(b"fake webm data")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx
        mock_run_ctx.tool_call_id = "test-id"

        result = await tool.call(mock_run_ctx, file_path="test.webm")

        assert isinstance(result, ToolReturn)
        assert result.content[0].media_type == "video/webm"


def test_supported_image_media_types() -> None:
    """Should have expected supported image media types."""
    assert "image/png" in SUPPORTED_IMAGE_MEDIA_TYPES
    assert "image/jpeg" in SUPPORTED_IMAGE_MEDIA_TYPES
    assert "image/webp" in SUPPORTED_IMAGE_MEDIA_TYPES
