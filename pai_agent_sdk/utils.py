from __future__ import annotations

import asyncio
import contextvars
import functools
import io
import socket
import typing
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

import anyio.to_thread
from pydantic_ai import Agent, ModelMessage, ModelResponse, RunContext, ToolCallPart
from pydantic_ai.messages import BinaryContent
from pydantic_ai.output import OutputDataT

from pai_agent_sdk.context import AgentContext

if TYPE_CHECKING:
    from pai_agent_sdk.toolsets.base import InstructableToolset

P = typing.ParamSpec("P")
T = typing.TypeVar("T")
U = typing.TypeVar("U")

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


async def _pump_iterator(
    idx: int,
    it: AsyncIterator[U],
    data_q: asyncio.Queue[tuple[int, U]],
    ctrl_q: asyncio.Queue[tuple[str, int, BaseException | None]],
) -> None:
    """Pump items from an async iterator into the data queue and signal completion."""
    try:
        async for item in it:
            await data_q.put((idx, item))
        await ctrl_q.put(("done", idx, None))
    except BaseException as e:
        await ctrl_q.put(("error", idx, e))


async def _cleanup_tasks(tasks: list[asyncio.Task], pending: set[asyncio.Task]) -> None:
    """Cancel and cleanup all tasks."""
    for t in pending:
        t.cancel()
    for t in tasks:
        if not t.done():
            t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def merge_async_iterators(*iterators: AsyncIterator[U]) -> AsyncIterator[U]:
    """Merge multiple async iterators into a single stream with context preservation.

    Each iterator runs in a separate task with preserved context variables.
    The operation exits immediately when any iterator completes or encounters an error.
    When both data and completion signals are ready simultaneously,
    completion signals take priority to maintain "exit-on-first-completion" semantics.

    Args:
        *iterators: The async iterators to merge

    Yields:
        Items from the iterators as they become available

    Raises:
        InterruptError: If any iterator raises InterruptError
        BaseException: If any iterator raises other exceptions
    """
    if not iterators:
        return

    data_q: asyncio.Queue[tuple[int, U]] = asyncio.Queue()
    ctrl_q: asyncio.Queue[tuple[str, int, BaseException | None]] = asyncio.Queue()

    tasks = [
        asyncio.create_task(_pump_iterator(i, it, data_q, ctrl_q), context=contextvars.copy_context())
        for i, it in enumerate(iterators)
    ]

    try:
        while True:
            ctrl_task = asyncio.create_task(ctrl_q.get())
            data_task = asyncio.create_task(data_q.get())
            done, pending = await asyncio.wait({ctrl_task, data_task}, return_when=asyncio.FIRST_COMPLETED)

            # Priority handling: completion/error signals take precedence over data
            if ctrl_task in done:
                _kind, _idx, exc = ctrl_task.result()
                await _cleanup_tasks(tasks, pending)
                if exc:
                    raise exc
                return

            # Only consume data when no completion/error signal received
            _, item = data_task.result()
            for t in pending:
                t.cancel()
            yield item
    finally:
        await _cleanup_tasks(tasks, set())


def add_toolset_instructions(
    agent: Agent[AgentContext, OutputDataT], toolsets: list[InstructableToolset]
) -> Agent[AgentContext, OutputDataT]:
    """Add instructions from toolsets to the agent.

    Works with any toolset that implements InstructableToolset protocol
    (has get_instructions method), including Toolset and BrowserUseToolset.

    TODO: Mark it deprecated and no-op when https://github.com/pydantic/pydantic-ai/pull/3780 merged
    """

    @agent.instructions
    def _(ctx: RunContext[AgentContext]) -> str | None:
        parts = [instructions for toolset in toolsets if (instructions := toolset.get_instructions(ctx))]
        if not parts:
            return None
        content = "\n".join(parts)
        return f"<toolset-instructions>\n{content}\n</toolset-instructions>"

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
    from PIL import Image

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
