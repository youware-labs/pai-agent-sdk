"""Write tool for writing or appending file contents."""

from functools import cache
from pathlib import Path
from typing import Annotated, cast

from agent_environment import FileOperator
from pydantic import Field
from pydantic_ai import RunContext

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.base import BaseTool

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@cache
def _load_instruction() -> str:
    """Load write instruction from prompts/write.md."""
    prompt_file = _PROMPTS_DIR / "write.md"
    return prompt_file.read_text()


class WriteTool(BaseTool):
    """Tool for writing entire file contents."""

    name = "write"
    description = "Write or overwrite entire file content. For partial edits, use `edit` tool instead."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available (requires file_operator)."""
        if ctx.deps.file_operator is None:
            logger.debug("WriteTool unavailable: file_operator is not configured")
            return False
        return True

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Load instruction from prompts/write.md."""
        return _load_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        file_path: Annotated[str, Field(description="Relative path to the file to write")],
        content: Annotated[str, Field(description="Content to write to the file")],
        mode: Annotated[
            str,
            Field(
                description="'w' for write/overwrite (default), 'a' for append",
                default="w",
            ),
        ] = "w",
    ) -> str:
        """Write content to a file in the local filesystem."""
        file_operator = cast(FileOperator, ctx.deps.file_operator)

        if mode not in ("w", "a"):
            return f"Error: Invalid mode '{mode}'. Only 'w' and 'a' are supported."

        if mode == "w":
            await file_operator.write_file(file_path, content)
        else:
            await file_operator.append_file(file_path, content)

        return f"Successfully wrote to file: {file_path}"


__all__ = ["WriteTool"]
