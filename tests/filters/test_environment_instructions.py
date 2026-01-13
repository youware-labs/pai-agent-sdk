"""Tests for pai_agent_sdk.filters.environment_instructions module."""

from pathlib import Path
from unittest.mock import MagicMock

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolReturnPart, UserPromptPart

from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.filters.environment_instructions import create_environment_instructions_filter


async def test_create_environment_instructions_filter_returns_callable(tmp_path: Path) -> None:
    """Factory should return a callable history processor."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        filter_func = create_environment_instructions_filter(env)
        assert callable(filter_func)


async def test_inject_environment_instructions_empty_history(tmp_path: Path) -> None:
    """Should return unchanged history when no ModelRequest found."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        filter_func = create_environment_instructions_filter(env)
        mock_ctx = MagicMock()

        result = await filter_func(mock_ctx, [])
        assert result == []


async def test_inject_environment_instructions_appends_to_last_request(tmp_path: Path) -> None:
    """Should append environment instructions to the last ModelRequest."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        filter_func = create_environment_instructions_filter(env)
        mock_ctx = MagicMock()

        # Create message history with a ModelRequest
        request = ModelRequest(parts=[UserPromptPart(content="Hello")])
        history = [request]

        result = await filter_func(mock_ctx, history)

        assert result == history
        # Should have added a part
        assert len(request.parts) == 2
        assert isinstance(request.parts[1], UserPromptPart)
        # Should contain file system or shell instructions
        assert "<file-system>" in request.parts[1].content or "<shell" in request.parts[1].content


async def test_inject_environment_instructions_finds_last_request(tmp_path: Path) -> None:
    """Should find and modify the last ModelRequest in history."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        filter_func = create_environment_instructions_filter(env)
        mock_ctx = MagicMock()

        # Create history with multiple messages
        request1 = ModelRequest(parts=[UserPromptPart(content="First")])
        response = ModelResponse(parts=[TextPart(content="Response")])
        request2 = ModelRequest(parts=[UserPromptPart(content="Second")])
        history = [request1, response, request2]

        await filter_func(mock_ctx, history)

        # Only the last request should be modified
        assert len(request1.parts) == 1
        assert len(request2.parts) == 2


async def test_inject_environment_instructions_only_model_response(tmp_path: Path) -> None:
    """Should return unchanged history when only ModelResponse found."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        filter_func = create_environment_instructions_filter(env)
        mock_ctx = MagicMock()

        response = ModelResponse(parts=[TextPart(content="Response")])
        history = [response]

        result = await filter_func(mock_ctx, history)

        assert result == history
        assert len(response.parts) == 1


async def test_inject_environment_instructions_skips_tool_response(tmp_path: Path) -> None:
    """Should skip injection when last_request contains ToolReturnPart."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        filter_func = create_environment_instructions_filter(env)
        mock_ctx = MagicMock()

        # Create a ModelRequest with ToolReturnPart (tool response)
        tool_return = ToolReturnPart(
            tool_name="test_tool",
            content="tool result",
            tool_call_id="call_123",
        )
        request = ModelRequest(parts=[tool_return])
        history = [request]

        result = await filter_func(mock_ctx, history)

        # Should not inject environment instructions
        assert result == history
        assert len(request.parts) == 1
        assert isinstance(request.parts[0], ToolReturnPart)
