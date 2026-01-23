"""Auto-load files history processor.

This module provides a history processor that automatically loads files
specified in AgentContext.auto_load_files and injects their content
into the message history.

Works with handoff tool and compact agent - both can set auto_load_files
to have files automatically loaded in the next context.

Example::

    from pydantic_ai import Agent

    from pai_agent_sdk.context import AgentContext
    from pai_agent_sdk.filters.auto_load_files import process_auto_load_files

    agent = Agent(
        'openai:gpt-4',
        deps_type=AgentContext,
        history_processors=[process_auto_load_files],
    )
"""

from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai.tools import RunContext

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.context import AgentContext

logger = get_logger(__name__)


async def process_auto_load_files(
    ctx: RunContext[AgentContext],
    message_history: list[ModelMessage],
) -> list[ModelMessage]:
    """Load files from auto_load_files and inject into message history.

    This processor reads files specified in ctx.deps.auto_load_files and
    appends their content as a UserPromptPart to the last ModelRequest.
    After loading, auto_load_files is cleared.

    Args:
        ctx: Runtime context with AgentContext.
        message_history: Current message history.

    Returns:
        Message history with auto-loaded file contents injected.
    """
    if not ctx.deps.auto_load_files:
        return message_history

    file_operator = ctx.deps.file_operator
    if not file_operator:
        logger.warning("auto_load_files specified but no file_operator available")
        return message_history

    # Find the last ModelRequest
    last_request: ModelRequest | None = None
    for msg in reversed(message_history):
        if isinstance(msg, ModelRequest):
            last_request = msg
            break

    if not last_request:
        return message_history

    # Skip if last request is tool return (only inject on user input)
    if any(isinstance(part, ToolReturnPart) for part in last_request.parts):
        return message_history

    # Load files
    file_contents: list[str] = []
    files_to_load = list(ctx.deps.auto_load_files)  # Copy before clearing

    for file_path in files_to_load:
        try:
            content = await file_operator.read_file(file_path)
            file_contents.append(f"### `{file_path}`\n\n```\n{content}\n```")
            logger.debug(f"Auto-loaded file: {file_path}")
        except Exception as e:
            file_contents.append(f"### `{file_path}`\n\n[Failed to load: {e}]")
            logger.warning(f"Failed to auto-load file {file_path}: {e}")

    # Clear after loading
    ctx.deps.auto_load_files = []

    if not file_contents:
        return message_history

    # Build content
    auto_load_content = "<auto-loaded-files>\n\n" + "\n\n".join(file_contents) + "\n\n</auto-loaded-files>"

    # Append to last request
    last_request.parts = [*last_request.parts, UserPromptPart(content=auto_load_content)]

    return message_history
