from __future__ import annotations

import functools
import io
import socket
import typing
from typing import TYPE_CHECKING, Literal

import anyio.to_thread
from PIL import Image
from pydantic_ai import AbstractToolset, Agent, ModelMessage, ModelResponse, RequestUsage, RunContext, ToolCallPart
from pydantic_ai.messages import BinaryContent
from pydantic_ai.output import OutputDataT
from typing_extensions import TypeVar

if TYPE_CHECKING:
    from pai_agent_sdk.context import AgentContext

P = typing.ParamSpec("P")
T = typing.TypeVar("T")
AgentDepsT = TypeVar("AgentDepsT", bound="AgentContext")

ImageMediaType = Literal["image/png", "image/jpeg", "image/gif", "image/webp"]


def get_available_port() -> int:
    """Get an available port on localhost.

    Note: There is a small race condition window between getting the port
    and actually binding to it. For most use cases this is acceptable.

    Returns:
        int: Available port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def run_in_threadpool(func: typing.Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    # copied from fastapi.concurrency import run_in_threadpool
    func = functools.partial(func, *args, **kwargs)
    return await anyio.to_thread.run_sync(func)


def get_latest_request_usage(message_history: list[ModelMessage]) -> RequestUsage | None:
    """
    Retrieve the latest RequestUsage from the message history.

    Args:
        message_history: List of model messages from conversation

    Returns:
        The latest RequestUsage if available, otherwise None
    """
    for message in reversed(message_history):
        if isinstance(message, ModelResponse) and message.usage:
            return message.usage
    return None


def add_toolset_instructions(
    agent: Agent[AgentDepsT, OutputDataT], toolsets: list[AbstractToolset]
) -> Agent[AgentDepsT, OutputDataT]:
    """Add instructions from toolsets to the agent.

    Works with any toolset that implements InstructableToolset protocol
    (has get_instructions method), including Toolset and BrowserUseToolset.
    Supports both sync and async get_instructions methods.

    TODO: Skip subclasses of AbstractToolset when https://github.com/pydantic/pydantic-ai/pull/3780 merged
    """
    from pai_agent_sdk.toolsets.base import InstructableToolset, resolve_instructions

    @agent.instructions
    async def _(ctx: RunContext[AgentDepsT]) -> str | None:
        parts: list[str] = []
        for toolset in toolsets:
            if isinstance(toolset, InstructableToolset):
                instructions = await resolve_instructions(toolset.get_instructions(ctx))
                if instructions:
                    parts.append(instructions)
        if not parts:
            return None
        return "\n".join(parts)

    return agent


def get_tool_name_from_id(tool_id: str, message_history: list[ModelMessage]) -> str | None:
    """
    Retrieve the tool name corresponding to a given tool ID from message history.

    Args:
        tool_id: The tool call ID to look for
        message_history: List of model messages from conversation

    Returns:
        The tool name if found, otherwise None
    """
    if not message_history:
        return None
    for message in message_history:
        if isinstance(message, ModelResponse) and any(
            isinstance(p, ToolCallPart) and p.tool_call_id == tool_id for p in message.parts
        ):
            for p in message.parts:
                if isinstance(p, ToolCallPart) and p.tool_call_id == tool_id:
                    return p.tool_name
    return None


async def split_image_data(
    image_bytes: bytes,
    max_height: int = 4096,
    overlap: int = 50,
    media_type: ImageMediaType = "image/png",
) -> list[BinaryContent]:
    """Split a large image into smaller vertical segments.

    This function takes an image and splits it into multiple segments if the height
    exceeds max_height. Each segment overlaps with the next by the specified amount.

    Args:
        image_bytes: The raw image data as bytes.
        max_height: Maximum height for each segment. Defaults to 4096.
        overlap: Number of pixels to overlap between segments. Defaults to 50.
        media_type: The MIME type for output images. Defaults to "image/png".

    Returns:
        A list of BinaryContent objects, each containing a segment of the image.
    """
    return await run_in_threadpool(_split_image_data_sync, image_bytes, max_height, overlap, media_type)


def _split_image_data_sync(
    image_bytes: bytes,
    max_height: int = 4096,
    overlap: int = 50,
    media_type: ImageMediaType = "image/png",
) -> list[BinaryContent]:
    """Synchronous implementation of split_image_data."""
    image = Image.open(io.BytesIO(image_bytes))
    width, height = image.size

    if height <= max_height:
        return [BinaryContent(data=image_bytes, media_type=media_type)]

    segments: list[BinaryContent] = []
    y = 0

    format_map = {
        "image/png": "PNG",
        "image/jpeg": "JPEG",
        "image/gif": "GIF",
        "image/webp": "WEBP",
    }
    pil_format = format_map.get(media_type, "PNG")

    while y < height:
        segment_height = min(max_height, height - y)
        segment = image.crop((0, y, width, y + segment_height))

        buffer = io.BytesIO()
        segment.save(buffer, format=pil_format)
        segment_bytes = buffer.getvalue()

        segments.append(BinaryContent(data=segment_bytes, media_type=media_type))

        y += max_height - overlap
        if y + overlap >= height:
            break

    return segments
