"""Tests for pai_agent_sdk.filters.handoff module."""

from pathlib import Path
from unittest.mock import MagicMock

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.events import HandoffCompleteEvent, HandoffStartEvent
from pai_agent_sdk.filters.handoff import process_handoff_message


async def test_process_handoff_no_handoff_message(tmp_path: Path) -> None:
    """Should return unchanged history when no handoff message is set."""
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

            result = await process_handoff_message(mock_ctx, history)

            assert result == history
            assert len(request.parts) == 1


async def test_process_handoff_with_handoff_message(tmp_path: Path) -> None:
    """Should inject handoff summary and include virtual tool call to prevent repeat handoff."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            ctx.handoff_message = "Previous context summary here"

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            request = ModelRequest(parts=[UserPromptPart(content="Continue task")])
            history = [request]

            result = await process_handoff_message(mock_ctx, history)

            # Should return 3 messages: user request, virtual tool call, virtual tool return
            assert len(result) == 3

            # First message: user request with handoff content
            first_msg = result[0]
            assert isinstance(first_msg, ModelRequest)
            # 4 parts: prefix, original, suffix, handoff
            assert len(first_msg.parts) == 4
            assert isinstance(first_msg.parts[0], UserPromptPart)
            assert "previous-user-request" in first_msg.parts[0].content
            assert isinstance(first_msg.parts[3], UserPromptPart)
            assert "Previous context summary here" in first_msg.parts[3].content
            assert "system-reminder" in first_msg.parts[3].content

            # Second message: virtual handoff tool call
            second_msg = result[1]
            assert isinstance(second_msg, ModelResponse)
            assert len(second_msg.parts) == 1
            assert isinstance(second_msg.parts[0], ToolCallPart)
            assert second_msg.parts[0].tool_name == "handoff"

            # Third message: virtual tool return
            third_msg = result[2]
            assert isinstance(third_msg, ModelRequest)
            assert len(third_msg.parts) == 1
            assert isinstance(third_msg.parts[0], ToolReturnPart)
            assert third_msg.parts[0].tool_name == "handoff"
            assert "acknowledged" in third_msg.parts[0].content.lower()

            # Tool call IDs should match
            assert second_msg.parts[0].tool_call_id == third_msg.parts[0].tool_call_id

            # Handoff message should be cleared
            assert ctx.handoff_message is None


async def test_process_handoff_empty_history(tmp_path: Path) -> None:
    """Should return unchanged empty history even with handoff message."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            ctx.handoff_message = "Summary"

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            result = await process_handoff_message(mock_ctx, [])

            assert result == []
            # Handoff message should NOT be cleared since no injection happened
            assert ctx.handoff_message == "Summary"


async def test_process_handoff_finds_last_user_request(tmp_path: Path) -> None:
    """Should find the last true user request (not tool return) for injection."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            ctx.handoff_message = "Context summary"

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            # Create history with user request, response, and tool return
            user_request = ModelRequest(parts=[UserPromptPart(content="Do something")])
            response = ModelResponse(parts=[ToolCallPart(tool_call_id="tc1", tool_name="some_tool", args={})])
            tool_return = ModelRequest(
                parts=[ToolReturnPart(tool_call_id="tc1", tool_name="some_tool", content="result")]
            )
            final_user = ModelRequest(parts=[UserPromptPart(content="Next task")])

            history = [user_request, response, tool_return, final_user]

            result = await process_handoff_message(mock_ctx, history)

            # Should return 3 messages: user request, virtual tool call, virtual tool return
            assert len(result) == 3

            # First message: modified user request with handoff content
            assert isinstance(result[0], ModelRequest)
            # 4 parts: prefix, original, suffix, handoff
            assert len(result[0].parts) == 4
            # The second part should be the original user prompt
            assert isinstance(result[0].parts[1], UserPromptPart)
            assert result[0].parts[1].content == "Next task"
            # The last part should be the handoff
            assert isinstance(result[0].parts[3], UserPromptPart)
            assert "Context summary" in result[0].parts[3].content

            # Second message: virtual handoff tool call
            assert isinstance(result[1], ModelResponse)
            assert result[1].parts[0].tool_name == "handoff"

            # Third message: virtual tool return
            assert isinstance(result[2], ModelRequest)
            assert isinstance(result[2].parts[0], ToolReturnPart)


async def test_process_handoff_skips_tool_return_only_request(tmp_path: Path) -> None:
    """Should skip ModelRequest that only contains ToolReturnPart."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            ctx.handoff_message = "Summary"

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            # History with only tool return request at the end
            user_request = ModelRequest(parts=[UserPromptPart(content="Start")])
            response = ModelResponse(parts=[TextPart(content="Response")])
            tool_return = ModelRequest(parts=[ToolReturnPart(tool_call_id="tc1", tool_name="tool", content="result")])

            history = [user_request, response, tool_return]

            result = await process_handoff_message(mock_ctx, history)

            # Should return 3 messages: user request, virtual tool call, virtual tool return
            assert len(result) == 3

            # First message: modified user request
            first_msg = result[0]
            assert isinstance(first_msg, ModelRequest)
            # 4 parts: prefix, original, suffix, handoff
            assert len(first_msg.parts) == 4
            assert isinstance(first_msg.parts[1], UserPromptPart)
            assert first_msg.parts[1].content == "Start"
            assert isinstance(first_msg.parts[3], UserPromptPart)
            assert "Summary" in first_msg.parts[3].content

            # Second and third messages: virtual handoff tool call and return
            assert isinstance(result[1], ModelResponse)
            assert isinstance(result[2], ModelRequest)


async def test_process_handoff_emits_events(tmp_path: Path) -> None:
    """Should emit HandoffStartEvent and HandoffCompleteEvent with handoff content."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            # Enable streaming to capture events
            ctx._stream_queue_enabled = True

            ctx.handoff_message = "Test handoff content"

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            request = ModelRequest(parts=[UserPromptPart(content="Continue")])
            history = [request]

            result = await process_handoff_message(mock_ctx, history)

            # Verify result: 3 messages (user request, virtual tool call, virtual tool return)
            assert len(result) == 3

            # Collect events from queue
            events = []
            while not ctx.agent_stream_queues[ctx.agent_id].empty():
                events.append(await ctx.agent_stream_queues[ctx.agent_id].get())

            # Should have start and complete events
            assert len(events) == 2

            start_event = events[0]
            assert isinstance(start_event, HandoffStartEvent)
            assert start_event.message_count == 1

            complete_event = events[1]
            assert isinstance(complete_event, HandoffCompleteEvent)
            assert complete_event.handoff_content == "Test handoff content"
            assert complete_event.original_message_count == 1
            # Event IDs should match
            assert start_event.event_id == complete_event.event_id


async def test_process_handoff_no_events_when_streaming_disabled(tmp_path: Path) -> None:
    """Should not emit events when streaming is disabled."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            # Streaming is disabled by default
            assert ctx._stream_queue_enabled is False

            ctx.handoff_message = "Test content"

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            request = ModelRequest(parts=[UserPromptPart(content="Continue")])
            history = [request]

            result = await process_handoff_message(mock_ctx, history)

            # Should return 3 messages
            assert len(result) == 3

            # Queue should be empty since streaming is disabled
            assert ctx.agent_stream_queues[ctx._agent_id].empty()
