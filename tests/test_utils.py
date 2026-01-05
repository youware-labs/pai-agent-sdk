"""Tests for pai_agent_sdk.utils module."""

import io
from collections.abc import AsyncIterator

import pytest
from PIL import Image
from pydantic_ai.messages import BinaryContent, ModelResponse, ToolCallPart

from pai_agent_sdk.utils import (
    _split_image_data_sync,
    get_available_port,
    get_tool_name_from_id,
    merge_async_iterators,
    run_in_threadpool,
    split_image_data,
)

# --- get_available_port tests ---


def test_get_available_port_returns_valid_port() -> None:
    """Should return a valid port number."""
    port = get_available_port()
    assert isinstance(port, int)
    assert 1 <= port <= 65535


def test_get_available_port_returns_different_ports() -> None:
    """Should return different ports on subsequent calls."""
    ports = {get_available_port() for _ in range(5)}
    assert len(ports) >= 3


# --- run_in_threadpool tests ---


@pytest.mark.asyncio
async def test_run_in_threadpool_runs_sync_function() -> None:
    """Should run synchronous function in threadpool."""

    def sync_add(a: int, b: int) -> int:
        return a + b

    result = await run_in_threadpool(sync_add, 1, 2)
    assert result == 3


@pytest.mark.asyncio
async def test_run_in_threadpool_passes_kwargs() -> None:
    """Should correctly pass keyword arguments."""

    def sync_func(a: int, *, multiplier: int = 1) -> int:
        return a * multiplier

    result = await run_in_threadpool(sync_func, 5, multiplier=3)
    assert result == 15


# --- merge_async_iterators tests ---


@pytest.mark.asyncio
async def test_merge_async_iterators_empty() -> None:
    """Should handle no iterators."""
    results = [item async for item in merge_async_iterators()]
    assert results == []


@pytest.mark.asyncio
async def test_merge_async_iterators_error_propagates() -> None:
    """Should propagate errors from iterators."""

    async def failing_gen() -> AsyncIterator[int]:
        yield 1
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        async for _ in merge_async_iterators(failing_gen()):
            pass


# --- get_tool_name_from_id tests ---


def test_get_tool_name_from_id_empty_history() -> None:
    """Should return None for empty history."""
    result = get_tool_name_from_id("test-id", [])
    assert result is None


def test_get_tool_name_from_id_no_match() -> None:
    """Should return None when tool ID not found."""
    tool_call = ToolCallPart(
        tool_name="test_tool",
        tool_call_id="other-id",
        args={"arg": "value"},
    )
    response = ModelResponse(parts=[tool_call])
    result = get_tool_name_from_id("test-id", [response])
    assert result is None


def test_get_tool_name_from_id_finds_match() -> None:
    """Should return tool name when ID matches."""
    tool_call = ToolCallPart(
        tool_name="my_tool",
        tool_call_id="target-id",
        args={"arg": "value"},
    )
    response = ModelResponse(parts=[tool_call])
    result = get_tool_name_from_id("target-id", [response])
    assert result == "my_tool"


def test_get_tool_name_from_id_multiple_messages() -> None:
    """Should search through multiple messages."""
    tool_call1 = ToolCallPart(tool_name="first_tool", tool_call_id="id-1", args={})
    tool_call2 = ToolCallPart(tool_name="second_tool", tool_call_id="id-2", args={})
    messages = [
        ModelResponse(parts=[tool_call1]),
        ModelResponse(parts=[tool_call2]),
    ]
    result = get_tool_name_from_id("id-2", messages)
    assert result == "second_tool"


# --- split_image_data tests ---


def _create_test_image(width: int, height: int, color: str = "red") -> bytes:
    """Create a test image and return its bytes."""
    img = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def test_split_image_data_small_image_no_split() -> None:
    """Should return single segment for small image."""
    image_bytes = _create_test_image(100, 100)
    result = _split_image_data_sync(image_bytes, max_height=200)
    assert len(result) == 1
    assert isinstance(result[0], BinaryContent)
    assert result[0].media_type == "image/png"


def test_split_image_data_exact_max_height() -> None:
    """Should return single segment when height equals max_height."""
    image_bytes = _create_test_image(100, 200)
    result = _split_image_data_sync(image_bytes, max_height=200)
    assert len(result) == 1


def test_split_image_data_large_image_split() -> None:
    """Should split large image into multiple segments."""
    image_bytes = _create_test_image(100, 500)
    result = _split_image_data_sync(image_bytes, max_height=200, overlap=50)
    assert len(result) > 1
    for segment in result:
        assert isinstance(segment, BinaryContent)


def test_split_image_data_different_media_types() -> None:
    """Should support different output formats."""
    image_bytes = _create_test_image(100, 100)

    for media_type in ["image/png", "image/jpeg", "image/webp"]:
        result = _split_image_data_sync(image_bytes, media_type=media_type)  # type: ignore[arg-type]
        assert result[0].media_type == media_type


@pytest.mark.asyncio
async def test_split_image_data_async() -> None:
    """Should work asynchronously."""
    image_bytes = _create_test_image(100, 100)
    result = await split_image_data(image_bytes, max_height=200)
    assert len(result) == 1
    assert isinstance(result[0], BinaryContent)
