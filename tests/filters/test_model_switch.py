"""Tests for pai_agent_sdk.filters.model_swtich module."""

from unittest.mock import MagicMock

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)

from pai_agent_sdk.filters.model_swtich import handle_model_switch


def test_handle_model_switch_empty_history() -> None:
    """Should return empty history unchanged."""
    mock_ctx = MagicMock()
    mock_ctx.model.model_name = "gpt-4"

    result = handle_model_switch(mock_ctx, [])

    assert result == []


def test_handle_model_switch_no_previous_response() -> None:
    """Should keep ThinkingParts when no previous ModelResponse exists."""
    mock_ctx = MagicMock()
    mock_ctx.model.model_name = "gpt-4"

    request = ModelRequest(parts=[UserPromptPart(content="Hello")])
    history = [request]

    result = handle_model_switch(mock_ctx, history)

    assert result == history
    assert len(result[0].parts) == 1


def test_handle_model_switch_same_model() -> None:
    """Should keep ThinkingParts when model hasn't changed."""
    mock_ctx = MagicMock()
    mock_ctx.model.model_name = "gpt-4"

    request = ModelRequest(parts=[UserPromptPart(content="Hello")])
    response = ModelResponse(
        parts=[TextPart(content="Hi"), ThinkingPart(content="thinking...")],
        model_name="gpt-4",
    )
    history = [request, response]

    result = handle_model_switch(mock_ctx, history)

    assert result == history
    assert len(result[1].parts) == 2
    assert any(isinstance(part, ThinkingPart) for part in result[1].parts)


def test_handle_model_switch_different_model() -> None:
    """Should remove ThinkingParts when model has changed."""
    mock_ctx = MagicMock()
    mock_ctx.model.model_name = "claude-3"

    request = ModelRequest(parts=[UserPromptPart(content="Hello")])
    response = ModelResponse(
        parts=[TextPart(content="Hi"), ThinkingPart(content="thinking...")],
        model_name="gpt-4",
    )
    history = [request, response]

    result = handle_model_switch(mock_ctx, history)

    # ThinkingPart should be removed from response
    assert len(result[1].parts) == 1
    assert isinstance(result[1].parts[0], TextPart)
    assert not any(isinstance(part, ThinkingPart) for part in result[1].parts)

    # Request should remain unchanged
    assert len(result[0].parts) == 1


def test_handle_model_switch_multiple_responses() -> None:
    """Should use the last response's model_name for comparison."""
    mock_ctx = MagicMock()
    mock_ctx.model.model_name = "claude-3"

    request1 = ModelRequest(parts=[UserPromptPart(content="Hello")])
    response1 = ModelResponse(
        parts=[TextPart(content="Hi"), ThinkingPart(content="thinking 1")],
        model_name="gpt-4",
    )
    request2 = ModelRequest(parts=[UserPromptPart(content="Continue")])
    response2 = ModelResponse(
        parts=[TextPart(content="Sure"), ThinkingPart(content="thinking 2")],
        model_name="gpt-4",
    )
    history = [request1, response1, request2, response2]

    result = handle_model_switch(mock_ctx, history)

    # Both responses should have ThinkingParts removed
    for msg in result:
        if isinstance(msg, ModelResponse):
            assert not any(isinstance(part, ThinkingPart) for part in msg.parts)


def test_handle_model_switch_response_without_model_name() -> None:
    """Should skip responses without model_name when looking for last model."""
    mock_ctx = MagicMock()
    mock_ctx.model.model_name = "claude-3"

    request = ModelRequest(parts=[UserPromptPart(content="Hello")])
    response1 = ModelResponse(
        parts=[TextPart(content="Hi"), ThinkingPart(content="thinking 1")],
        model_name="gpt-4",
    )
    response2 = ModelResponse(
        parts=[TextPart(content="Sure")],
        model_name=None,  # No model_name
    )
    history = [request, response1, response2]

    result = handle_model_switch(mock_ctx, history)

    # Should use response1's model_name (gpt-4) and remove ThinkingParts
    assert not any(isinstance(part, ThinkingPart) for part in result[1].parts)
