"""Tests for pai_agent_sdk.filters.auto_load_files module."""

from pathlib import Path
from unittest.mock import MagicMock

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.filters.auto_load_files import process_auto_load_files


async def test_no_auto_load_files(tmp_path: Path) -> None:
    """Should return unchanged history when no auto_load_files is set."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            request = ModelRequest(parts=[UserPromptPart(content="Hello")])
            history = [request]

            result = await process_auto_load_files(mock_ctx, history)

            assert result == history
            assert len(request.parts) == 1


async def test_auto_load_files_injects_content(tmp_path: Path) -> None:
    """Should inject file content into last request when auto_load_files is set."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello World")

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            ctx.auto_load_files = ["test.txt"]

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            request = ModelRequest(parts=[UserPromptPart(content="Continue")])
            history = [request]

            result = await process_auto_load_files(mock_ctx, history)

            # Should return same history object
            assert result == history

            # Should have 2 parts now: original + auto-loaded
            assert len(request.parts) == 2
            assert isinstance(request.parts[1], UserPromptPart)
            assert "<auto-loaded-files>" in request.parts[1].content
            assert "test.txt" in request.parts[1].content
            assert "Hello World" in request.parts[1].content

            # auto_load_files should be cleared
            assert ctx.auto_load_files == []


async def test_auto_load_files_multiple_files(tmp_path: Path) -> None:
    """Should inject multiple files."""
    # Create test files
    (tmp_path / "file1.txt").write_text("Content 1")
    (tmp_path / "file2.txt").write_text("Content 2")

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            ctx.auto_load_files = ["file1.txt", "file2.txt"]

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            request = ModelRequest(parts=[UserPromptPart(content="Continue")])
            history = [request]

            await process_auto_load_files(mock_ctx, history)

            assert len(request.parts) == 2
            content = request.parts[1].content
            assert "file1.txt" in content
            assert "Content 1" in content
            assert "file2.txt" in content
            assert "Content 2" in content


async def test_auto_load_files_handles_missing_file(tmp_path: Path) -> None:
    """Should handle missing files gracefully."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            ctx.auto_load_files = ["nonexistent.txt"]

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            request = ModelRequest(parts=[UserPromptPart(content="Continue")])
            history = [request]

            await process_auto_load_files(mock_ctx, history)

            assert len(request.parts) == 2
            content = request.parts[1].content
            assert "nonexistent.txt" in content
            assert "Failed to load" in content


async def test_auto_load_files_skips_tool_return(tmp_path: Path) -> None:
    """Should skip injection when last request is tool return."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Content")

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            ctx.auto_load_files = ["test.txt"]

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            # Last request is a tool return
            user_request = ModelRequest(parts=[UserPromptPart(content="Do something")])
            response = ModelResponse(parts=[TextPart(content="Response")])
            tool_return = ModelRequest(parts=[ToolReturnPart(tool_call_id="tc1", tool_name="tool", content="result")])

            history = [user_request, response, tool_return]

            await process_auto_load_files(mock_ctx, history)

            # Should not inject into tool_return
            assert len(tool_return.parts) == 1

            # auto_load_files should NOT be cleared (waiting for user input)
            assert ctx.auto_load_files == ["test.txt"]


async def test_auto_load_files_no_file_operator(tmp_path: Path) -> None:
    """Should return unchanged history when no file_operator available."""
    async with AgentContext() as ctx:
        ctx.auto_load_files = ["test.txt"]

        mock_ctx = MagicMock()
        mock_ctx.deps = ctx

        request = ModelRequest(parts=[UserPromptPart(content="Hello")])
        history = [request]

        result = await process_auto_load_files(mock_ctx, history)

        assert result == history
        assert len(request.parts) == 1


async def test_auto_load_files_empty_history(tmp_path: Path) -> None:
    """Should return unchanged empty history."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Content")

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            ctx.auto_load_files = ["test.txt"]

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            result = await process_auto_load_files(mock_ctx, [])

            assert result == []
            # auto_load_files should NOT be cleared since no injection happened
            assert ctx.auto_load_files == ["test.txt"]
