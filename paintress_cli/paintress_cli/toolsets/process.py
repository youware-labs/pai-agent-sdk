"""Process management tools for TUI environment.

These tools allow the agent to spawn, manage, and interact with
background processes through the ProcessManager resource.

Example:
    from pai_agent_sdk.toolsets.core.base import Toolset
    from paintress_cli.toolsets.process import process_tools

    toolset = Toolset(tools=process_tools)
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field
from pydantic_ai import RunContext, UserError
from typing_extensions import TypedDict

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.base import BaseTool
from paintress_cli.processes import PROCESS_MANAGER_KEY, ProcessManager

# -----------------------------------------------------------------------------
# Result Types
# -----------------------------------------------------------------------------


class SpawnResult(TypedDict):
    """Result from spawning a process."""

    process_id: str
    command: str
    pid: int
    message: str


class KillResult(TypedDict):
    """Result from killing a process."""

    success: bool
    process_id: str
    message: str


class ProcessInfoDict(TypedDict, total=False):
    """Process info as dict."""

    process_id: str
    command: str
    args: list[str]
    pid: int
    is_running: bool
    exit_code: int | None
    elapsed_seconds: int


class ListResult(TypedDict):
    """Result from listing processes."""

    count: int
    processes: list[ProcessInfoDict]


class OutputResult(TypedDict):
    """Result from reading process output."""

    process_id: str
    lines_read: int
    has_more: bool
    stdout: list[str]
    stderr: list[str]


# -----------------------------------------------------------------------------
# Base Class
# -----------------------------------------------------------------------------


class ProcessToolBase(BaseTool):
    """Base class for process management tools.

    Provides common functionality for checking ProcessManager availability
    and accessing it from the resource registry.
    """

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if ProcessManager is available in resources."""
        return self._has_manager(ctx)

    def _has_manager(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if ProcessManager exists in resources."""
        if ctx.deps.resources is None:
            return False
        return ctx.deps.resources.get(PROCESS_MANAGER_KEY) is not None

    def _get_manager(self, ctx: RunContext[AgentContext]) -> ProcessManager:
        """Get ProcessManager from resources.

        Raises:
            UserError: If resources or ProcessManager not available.
        """
        if ctx.deps.resources is None:
            raise UserError("Resources not available in context")

        pm = ctx.deps.resources.get_typed(PROCESS_MANAGER_KEY, ProcessManager)
        if pm is None:
            raise UserError(
                f"ProcessManager not registered. "
                f"Register it with: env.resources.set('{PROCESS_MANAGER_KEY}', ProcessManager())"
            )
        return pm


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------


class SpawnProcessTool(ProcessToolBase):
    """Tool for spawning background processes."""

    name = "spawn_process"
    description = (
        "Spawn a background process. The process runs asynchronously and its output "
        "can be read later with read_process_output. Use for long-running tasks like "
        "dev servers (npm run dev) or build watchers."
    )

    async def call(
        self,
        ctx: RunContext[AgentContext],
        command: Annotated[str, Field(description="Command to execute (e.g., 'npm', 'python').")],
        args: Annotated[
            list[str] | None,
            Field(description="Command arguments (e.g., ['run', 'dev'])."),
        ] = None,
        cwd: Annotated[
            str | None,
            Field(description="Working directory for the process."),
        ] = None,
        process_id: Annotated[
            str | None,
            Field(description="Custom process ID. Auto-generated if not provided."),
        ] = None,
    ) -> SpawnResult:
        """Spawn a new background process."""
        pm = self._get_manager(ctx)

        managed = await pm.spawn(
            command=command,
            args=args,
            cwd=cwd,
            process_id=process_id,
        )

        return SpawnResult(
            process_id=managed.process_id,
            command=f"{command} {' '.join(args or [])}".strip(),
            pid=managed.pid,
            message=f"Process started with PID {managed.pid}",
        )


class KillProcessTool(ProcessToolBase):
    """Tool for killing a process."""

    name = "kill_process"
    description = "Kill a running background process by its process ID."

    async def call(
        self,
        ctx: RunContext[AgentContext],
        process_id: Annotated[str, Field(description="Process ID to kill.")],
        timeout: Annotated[
            float,
            Field(description="Seconds to wait for graceful termination before force kill."),
        ] = 5.0,
    ) -> KillResult:
        """Kill a process by ID."""
        pm = self._get_manager(ctx)

        success = await pm.kill(process_id, timeout=timeout)

        if success:
            return KillResult(
                success=True,
                process_id=process_id,
                message=f"Process {process_id} terminated",
            )
        else:
            return KillResult(
                success=False,
                process_id=process_id,
                message=f"Process {process_id} not found",
            )


class ListProcessesTool(ProcessToolBase):
    """Tool for listing all managed processes."""

    name = "list_processes"
    description = "List all managed background processes with their status."

    async def call(
        self,
        ctx: RunContext[AgentContext],
    ) -> ListResult:
        """List all processes."""
        pm = self._get_manager(ctx)

        from datetime import datetime

        processes: list[ProcessInfoDict] = []
        for info in pm.list_processes():
            elapsed = datetime.now() - info.started_at
            processes.append(
                ProcessInfoDict(
                    process_id=info.process_id,
                    command=info.command,
                    args=info.args,
                    pid=info.pid,
                    is_running=info.is_running,
                    exit_code=info.exit_code,
                    elapsed_seconds=int(elapsed.total_seconds()),
                )
            )

        return ListResult(
            count=len(processes),
            processes=processes,
        )


# Default max lines to read at once
DEFAULT_MAX_LINES = 100


class ReadProcessOutputTool(ProcessToolBase):
    """Tool for reading process output."""

    name = "read_process_output"
    description = (
        "Read buffered stdout/stderr output from a background process. "
        "Output is consumed (cleared from buffer) after reading. "
        "Use max_lines to limit output size."
    )

    async def call(
        self,
        ctx: RunContext[AgentContext],
        process_id: Annotated[str, Field(description="Process ID to read output from.")],
        max_lines: Annotated[
            int,
            Field(description="Maximum lines to read. Remaining lines stay in buffer."),
        ] = DEFAULT_MAX_LINES,
    ) -> OutputResult:
        """Read output from a process."""
        pm = self._get_manager(ctx)

        managed = pm.get_process(process_id)
        if managed is None:
            raise UserError(f"Process {process_id} not found")

        # Read output with limit - this clears read lines from buffer
        output = await managed.read_output(max_lines=max_lines)

        stdout_lines = [line for line, is_err in output if not is_err]
        stderr_lines = [line for line, is_err in output if is_err]

        # Check if there's more in buffer
        remaining = await managed.read_output(max_lines=1)
        has_more = len(remaining) > 0

        # Put back the peeked line if any
        if remaining:
            line, is_err = remaining[0]
            await managed._append_output(line, is_err)

        return OutputResult(
            process_id=process_id,
            lines_read=len(output),
            has_more=has_more,
            stdout=stdout_lines,
            stderr=stderr_lines,
        )


# -----------------------------------------------------------------------------
# Tools List
# -----------------------------------------------------------------------------

process_tools: list[type[BaseTool]] = [
    SpawnProcessTool,
    KillProcessTool,
    ListProcessesTool,
    ReadProcessOutputTool,
]
"""List of process management tools for use with Toolset."""
