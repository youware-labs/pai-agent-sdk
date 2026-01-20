"""Shell command execution tool.

This module provides a tool for executing shell commands
using the shell provided by AgentContext.
"""

from functools import cache
from pathlib import Path
from typing import Annotated, cast

from agent_environment import Shell
from pydantic import Field
from pydantic_ai import RunContext
from typing_extensions import TypedDict

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.base import BaseTool

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

OUTPUT_TRUNCATE_LIMIT = 20000
DEFAULT_TIMEOUT_SECONDS = 180


@cache
def _load_instruction() -> str:
    """Load shell instruction from prompts/shell.md."""
    prompt_file = _PROMPTS_DIR / "shell.md"
    return prompt_file.read_text()


class ShellResult(TypedDict, total=False):
    """Result of shell command execution."""

    stdout: str
    stderr: str
    return_code: int
    stdout_file_path: str  # Present when stdout exceeds limit
    stderr_file_path: str  # Present when stderr exceeds limit
    error: str  # Present on execution error


class ShellTool(BaseTool):
    """Tool for executing shell commands."""

    name = "shell"
    description = "Execute a shell command."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available (requires shell)."""
        if ctx.deps.shell is None:
            logger.debug("ShellTool unavailable: shell is not configured")
            return False
        return True

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str:
        """Load instruction from prompts/shell.md."""
        return _load_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        command: Annotated[str, Field(description="The shell command to execute.")],
        timeout_seconds: Annotated[
            int,
            Field(
                default=DEFAULT_TIMEOUT_SECONDS,
                description="Maximum execution time in seconds.",
            ),
        ] = DEFAULT_TIMEOUT_SECONDS,
        environment: Annotated[
            dict[str, str] | None,
            Field(description="Environment variables to set for the command."),
        ] = None,
        cwd: Annotated[
            str | None,
            Field(description="Working directory (relative or absolute path)."),
        ] = None,
    ) -> ShellResult:
        if not command or not command.strip():
            return ShellResult(
                stdout="",
                stderr="",
                return_code=1,
                error="Command cannot be empty.",
            )

        shell = cast(Shell, ctx.deps.shell)
        file_op = ctx.deps.file_operator

        try:
            exit_code, stdout, stderr = await shell.execute(
                command,
                timeout=float(timeout_seconds),
                env=environment,
                cwd=cwd,
            )

            result = ShellResult(
                stdout=stdout,
                stderr=stderr,
                return_code=exit_code,
            )

            # Handle stdout truncation (only save to file if file_operator is available)
            if len(stdout) > OUTPUT_TRUNCATE_LIMIT:
                if file_op is not None:
                    stdout_file = f"stdout-{ctx.deps.run_id[:8]}.log"
                    stdout_path = await file_op.write_tmp_file(stdout_file, stdout)
                    result["stdout"] = (
                        stdout[:OUTPUT_TRUNCATE_LIMIT] + "\n...(truncated, full output at `stdout_file_path`)"
                    )
                    result["stdout_file_path"] = stdout_path
                else:
                    result["stdout"] = stdout[:OUTPUT_TRUNCATE_LIMIT] + "\n...(truncated)"

            # Handle stderr truncation (only save to file if file_operator is available)
            if len(stderr) > OUTPUT_TRUNCATE_LIMIT:
                if file_op is not None:
                    stderr_file = f"stderr-{ctx.deps.run_id[:8]}.log"
                    stderr_path = await file_op.write_tmp_file(stderr_file, stderr)
                    result["stderr"] = (
                        stderr[:OUTPUT_TRUNCATE_LIMIT] + "\n...(truncated, full output at `stderr_file_path`)"
                    )
                    result["stderr_file_path"] = stderr_path
                else:
                    result["stderr"] = stderr[:OUTPUT_TRUNCATE_LIMIT] + "\n...(truncated)"

            return result

        except Exception as e:
            return ShellResult(
                stdout="",
                stderr="",
                return_code=1,
                error=f"Failed to execute command: {e}",
            )
