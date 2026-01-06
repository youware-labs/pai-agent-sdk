"""Tests for pai_agent_sdk.filters.tool_args module."""

from pathlib import Path
from unittest.mock import MagicMock

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.filters.tool_args import fix_truncated_tool_args


async def test_fix_truncated_tool_args_valid_json(tmp_path: Path) -> None:
    """Should keep valid JSON tool args unchanged."""
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

            tool_call = ToolCallPart(
                tool_name="test_tool",
                args='{"key": "value"}',
                tool_call_id="call_123",
            )
            response = ModelResponse(parts=[tool_call], model_name="test-model")
            history = [response]

            await fix_truncated_tool_args(mock_ctx, history)

            # Valid JSON should be unchanged
            assert tool_call.args == '{"key": "value"}'


async def test_fix_truncated_tool_args_invalid_json(tmp_path: Path) -> None:
    """Should replace invalid JSON tool args with error placeholder."""
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

            tool_call = ToolCallPart(
                tool_name="test_tool",
                args='{"key": "truncated...',  # Invalid JSON
                tool_call_id="call_123",
            )
            response = ModelResponse(parts=[tool_call], model_name="test-model")
            history = [response]

            await fix_truncated_tool_args(mock_ctx, history)

            # Invalid JSON should be replaced with error dict
            assert isinstance(tool_call.args, dict)
            assert "system" in tool_call.args
            assert "not a valid JSON" in tool_call.args["system"]


async def test_fix_truncated_tool_args_skips_model_request(tmp_path: Path) -> None:
    """Should skip ModelRequest messages."""
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

            request = ModelRequest(parts=[UserPromptPart(content="Hello")])
            history = [request]

            result = await fix_truncated_tool_args(mock_ctx, history)

            assert result == history
            assert request.parts[0].content == "Hello"  # type: ignore[union-attr]


async def test_fix_truncated_tool_args_dict_args_unchanged(tmp_path: Path) -> None:
    """Should keep dict tool args unchanged (not string)."""
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

            original_args = {"key": "value"}
            tool_call = ToolCallPart(
                tool_name="test_tool",
                args=original_args,
                tool_call_id="call_123",
            )
            response = ModelResponse(parts=[tool_call], model_name="test-model")
            history = [response]

            await fix_truncated_tool_args(mock_ctx, history)

            # Dict args should be unchanged
            assert tool_call.args == original_args


async def test_fix_truncated_tool_args_empty_string(tmp_path: Path) -> None:
    """Should handle empty string args as invalid JSON."""
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

            tool_call = ToolCallPart(
                tool_name="test_tool",
                args="",  # Empty string - invalid JSON
                tool_call_id="call_123",
            )
            response = ModelResponse(parts=[tool_call], model_name="test-model")
            history = [response]

            await fix_truncated_tool_args(mock_ctx, history)

            # Empty string should be replaced with error dict
            assert isinstance(tool_call.args, dict)
            assert "system" in tool_call.args


async def test_fix_truncated_tool_args_multiple_tool_calls(tmp_path: Path) -> None:
    """Should handle multiple tool calls in single response."""
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

            valid_call = ToolCallPart(
                tool_name="valid_tool",
                args='{"valid": true}',
                tool_call_id="call_1",
            )
            invalid_call = ToolCallPart(
                tool_name="invalid_tool",
                args='{"truncated',
                tool_call_id="call_2",
            )
            response = ModelResponse(
                parts=[valid_call, invalid_call],
                model_name="test-model",
            )
            history = [response]

            await fix_truncated_tool_args(mock_ctx, history)

            # Valid should be unchanged
            assert valid_call.args == '{"valid": true}'
            # Invalid should be replaced
            assert isinstance(invalid_call.args, dict)
            assert "system" in invalid_call.args


async def test_fix_truncated_tool_args_mixed_parts(tmp_path: Path) -> None:
    """Should only process ToolCallPart, skip other parts."""
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

            text_part = TextPart(content="Some response text")
            tool_call = ToolCallPart(
                tool_name="test_tool",
                args='{"valid": true}',
                tool_call_id="call_123",
            )
            response = ModelResponse(
                parts=[text_part, tool_call],
                model_name="test-model",
            )
            history = [response]

            await fix_truncated_tool_args(mock_ctx, history)

            # TextPart should be unchanged
            assert text_part.content == "Some response text"
            # Valid tool call should be unchanged
            assert tool_call.args == '{"valid": true}'
