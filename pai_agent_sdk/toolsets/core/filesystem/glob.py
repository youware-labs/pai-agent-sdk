"""Glob tool for file pattern matching."""

from functools import cache
from pathlib import Path
from typing import Annotated, cast

from agent_environment import FileOperator
from pydantic import Field
from pydantic_ai import RunContext

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.base import BaseTool
from pai_agent_sdk.toolsets.core.filesystem._gitignore import filter_gitignored

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@cache
def _load_instruction() -> str:
    """Load glob instruction from prompts/glob.md."""
    prompt_file = _PROMPTS_DIR / "glob.md"
    return prompt_file.read_text()


class GlobTool(BaseTool):
    """Tool for finding files matching glob patterns."""

    name = "glob"
    description = "Find files by glob pattern. Returns paths sorted by modification time (newest first)."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available (requires file_operator)."""
        if ctx.deps.file_operator is None:
            logger.debug("GlobTool unavailable: file_operator is not configured")
            return False
        return True

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Load instruction from prompts/glob.md."""
        return _load_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        pattern: Annotated[
            str,
            Field(description="Glob pattern to match files (e.g. '**/*.py')"),
        ],
        include_ignored: Annotated[
            bool,
            Field(description="Include files ignored by .gitignore (default: false)", default=False),
        ] = False,
    ) -> list[str]:
        """Find files matching the given glob pattern."""
        file_operator = cast(FileOperator, ctx.deps.file_operator)
        files = await file_operator.glob(pattern)

        # Filter out gitignored files by default
        if not include_ignored:
            files = await filter_gitignored(files, file_operator)

        return files


__all__ = ["GlobTool"]
