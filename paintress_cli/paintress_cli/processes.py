"""Process management as an Environment Resource.

ProcessManager is a BaseResource that manages subprocess lifecycle.
It can be registered with Environment.resources and accessed by tools
or directly by TUI for visualization.

Example:
    # Register with environment
    async def create_process_manager(env: Environment) -> ProcessManager:
        return ProcessManager()

    env.resources.register_factory("process_manager", create_process_manager)

    # Access from toolset
    pm = ctx.deps.resources.get_typed("process_manager", ProcessManager)
    process = await pm.spawn("npm", ["run", "dev"])

    # TUI can directly access for visualization
    for info in pm.list_processes():
        print(f"{info.process_id}: {info.command} (pid={info.pid})")
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from xml.etree.ElementTree import Element, SubElement, indent, tostring

from agent_environment import BaseResource, Environment

if TYPE_CHECKING:
    pass


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class ProcessInfo:
    """Public information about a managed process."""

    process_id: str
    command: str
    args: list[str]
    pid: int
    started_at: datetime
    is_running: bool
    exit_code: int | None = None


@dataclass
class ManagedProcess:
    """Internal process wrapper with management capabilities."""

    process_id: str
    command: str
    args: list[str]
    process: asyncio.subprocess.Process
    started_at: datetime = field(default_factory=datetime.now)
    cwd: str | None = None
    exited_at: datetime | None = field(default=None)
    """Timestamp when process exited, for TTL-based cleanup."""
    _output_buffer: list[tuple[str, bool]] = field(default_factory=list)
    """Buffer of (line, is_stderr) tuples."""
    _output_tasks: list[asyncio.Task[None]] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def pid(self) -> int:
        """Get OS process ID."""
        return self.process.pid or 0

    @property
    def is_running(self) -> bool:
        """Check if process is still running."""
        was_running = self.process.returncode is None
        # Record exit time when we first detect the process has exited
        if not was_running and self.exited_at is None:
            # Use object.__setattr__ since dataclass might be frozen in some contexts
            object.__setattr__(self, "exited_at", datetime.now())
        return was_running

    @property
    def exit_code(self) -> int | None:
        """Get exit code, or None if still running."""
        return self.process.returncode

    def to_info(self) -> ProcessInfo:
        """Convert to public ProcessInfo."""
        return ProcessInfo(
            process_id=self.process_id,
            command=self.command,
            args=self.args,
            pid=self.pid,
            started_at=self.started_at,
            is_running=self.is_running,
            exit_code=self.exit_code,
        )

    async def read_output(self, max_lines: int | None = None) -> list[tuple[str, bool]]:
        """Read buffered output lines.

        Args:
            max_lines: Maximum lines to return. None for all.

        Returns:
            List of (line, is_stderr) tuples.
        """
        async with self._lock:
            if max_lines is None:
                result = self._output_buffer.copy()
                self._output_buffer.clear()
            else:
                result = self._output_buffer[:max_lines]
                self._output_buffer = self._output_buffer[max_lines:]
            return result

    async def kill(self, timeout: float = 5.0) -> int:
        """Kill the process and wait for cleanup.

        Args:
            timeout: Seconds to wait for graceful termination before force kill.

        Returns:
            Exit code.
        """
        if not self.is_running:
            return self.exit_code or 0

        # Try graceful termination first
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=timeout)
        except TimeoutError:
            # Force kill
            self.process.kill()
            await self.process.wait()

        # Cancel output tasks
        for task in self._output_tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        return self.exit_code if self.exit_code is not None else -1

    async def wait(self) -> int:
        """Wait for process to complete.

        Returns:
            Exit code.
        """
        await self.process.wait()
        return self.exit_code or 0

    async def _append_output(self, line: str, is_stderr: bool) -> None:
        """Append a line to output buffer."""
        async with self._lock:
            self._output_buffer.append((line, is_stderr))


# -----------------------------------------------------------------------------
# ProcessManager Resource
# -----------------------------------------------------------------------------


class ProcessManager(BaseResource):
    """Manages subprocess lifecycle as an Environment Resource.

    This resource tracks all spawned processes and provides:
    - Process spawning with output capture
    - Process lifecycle management (kill, wait)
    - Process listing for TUI visualization
    - Automatic cleanup on close()
    - TTL-based cleanup of exited processes

    The resource implements InstructableResource to report running
    processes to the agent context.
    """

    def __init__(self, exited_ttl: float = 300.0) -> None:
        """Initialize ProcessManager.

        Args:
            exited_ttl: Time-to-live in seconds for exited processes before auto-cleanup.
                       Default is 300 seconds (5 minutes). Set to 0 to disable auto-cleanup.
        """
        self._processes: dict[str, ManagedProcess] = {}
        self._lock = asyncio.Lock()
        self._exited_ttl = exited_ttl

    async def close(self) -> None:
        """Close the resource by killing all processes."""
        await self.kill_all()

    def get_toolsets(self) -> list[Any]:
        """Return process management toolset."""
        from pai_agent_sdk.toolsets.core.base import Toolset
        from paintress_cli.toolsets.process import process_tools

        return [Toolset(tools=process_tools, toolset_id="process")]

    async def _cleanup_expired(self) -> None:
        """Remove exited processes that have exceeded the TTL."""
        if self._exited_ttl <= 0:
            return

        now = datetime.now()
        expired_ids: list[str] = []

        async with self._lock:
            for proc_id, proc in self._processes.items():
                # Check is_running to trigger exited_at recording
                if not proc.is_running and proc.exited_at is not None:
                    elapsed = (now - proc.exited_at).total_seconds()
                    if elapsed > self._exited_ttl:
                        expired_ids.append(proc_id)

            for proc_id in expired_ids:
                del self._processes[proc_id]

    async def get_context_instructions(self) -> str | None:
        """Report all processes to agent in XML format.

        Returns context instructions listing all processes (running and exited),
        or None if no processes exist.
        """
        # Cleanup expired processes before reporting
        await self._cleanup_expired()

        if not self._processes:
            return None

        root = Element("processes")
        root.set("hint", "Use spawn_process, kill_process, list_processes, read_process_output tools to manage")

        for p in self._processes.values():
            status = "running" if p.is_running else "exited"
            proc_elem = SubElement(root, "process")
            proc_elem.set("id", p.process_id)
            proc_elem.set("status", status)

            cmd_elem = SubElement(proc_elem, "command")
            cmd_elem.text = f"{p.command} {' '.join(p.args)}".strip()

            pid_elem = SubElement(proc_elem, "pid")
            pid_elem.text = str(p.pid)

            if p.cwd:
                cwd_elem = SubElement(proc_elem, "cwd")
                cwd_elem.text = p.cwd

            started_elem = SubElement(proc_elem, "started_at")
            started_elem.text = p.started_at.isoformat()

            elapsed = datetime.now() - p.started_at
            elapsed_elem = SubElement(proc_elem, "elapsed_seconds")
            elapsed_elem.text = str(int(elapsed.total_seconds()))

            if not p.is_running and p.exit_code is not None:
                exit_elem = SubElement(proc_elem, "exit_code")
                exit_elem.text = str(p.exit_code)

        indent(root, space="  ")
        return tostring(root, encoding="unicode")

    async def spawn(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        process_id: str | None = None,
        capture_output: bool = True,
    ) -> ManagedProcess:
        """Spawn a new managed subprocess.

        Args:
            command: Command to execute.
            args: Command arguments.
            cwd: Working directory.
            env: Environment variables (merged with current).
            process_id: Custom process ID (auto-generated if None).
            capture_output: Whether to capture stdout/stderr to buffer.

        Returns:
            ManagedProcess instance.
        """
        # Cleanup expired processes before spawning new one
        await self._cleanup_expired()

        proc_id = process_id or f"proc-{uuid.uuid4().hex[:8]}"
        args = args or []

        # Prepare environment
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        # Spawn process
        process = await asyncio.create_subprocess_exec(
            command,
            *args,
            cwd=cwd,
            env=full_env,
            stdout=asyncio.subprocess.PIPE if capture_output else None,
            stderr=asyncio.subprocess.PIPE if capture_output else None,
        )

        managed = ManagedProcess(
            process_id=proc_id,
            command=command,
            args=args,
            process=process,
            cwd=cwd,
        )

        async with self._lock:
            self._processes[proc_id] = managed

        # Start output capture tasks
        if capture_output:
            if process.stdout:
                stdout_task = asyncio.create_task(self._capture_stream(managed, process.stdout, is_stderr=False))
                managed._output_tasks.append(stdout_task)
            if process.stderr:
                stderr_task = asyncio.create_task(self._capture_stream(managed, process.stderr, is_stderr=True))
                managed._output_tasks.append(stderr_task)

        return managed

    async def _capture_stream(
        self,
        managed: ManagedProcess,
        stream: asyncio.StreamReader,
        is_stderr: bool,
    ) -> None:
        """Capture process output to buffer."""
        try:
            async for line in stream:
                decoded = line.decode("utf-8", errors="replace").rstrip()
                if decoded:
                    await managed._append_output(decoded, is_stderr)
        except asyncio.CancelledError:
            pass

    async def kill(self, process_id: str, timeout: float = 5.0) -> bool:
        """Kill a specific process by ID.

        Args:
            process_id: Process identifier.
            timeout: Seconds to wait for graceful termination.

        Returns:
            True if process was found and killed, False if not found.
        """
        async with self._lock:
            managed = self._processes.get(process_id)
            if not managed:
                return False

        await managed.kill(timeout)
        return True

    async def kill_all(self, timeout: float = 10.0) -> None:
        """Kill all managed processes.

        Args:
            timeout: Seconds to wait for each process.
        """
        async with self._lock:
            processes = list(self._processes.values())

        # Kill all in parallel
        await asyncio.gather(
            *[p.kill(timeout) for p in processes],
            return_exceptions=True,
        )

    def list_processes(self) -> list[ProcessInfo]:
        """List all managed processes.

        Returns:
            List of ProcessInfo for all tracked processes.
        """
        return [p.to_info() for p in self._processes.values()]

    def get_process(self, process_id: str) -> ManagedProcess | None:
        """Get a specific process by ID.

        Args:
            process_id: Process identifier.

        Returns:
            ManagedProcess or None if not found.
        """
        return self._processes.get(process_id)

    def __len__(self) -> int:
        """Return number of managed processes."""
        return len(self._processes)

    def __contains__(self, process_id: str) -> bool:
        """Check if a process exists."""
        return process_id in self._processes


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------


async def create_process_manager(env: Environment) -> ProcessManager:
    """Factory function for creating ProcessManager.

    Register with environment:
        env.resources.register_factory("process_manager", create_process_manager)

    Args:
        env: The Environment instance (unused, but required by factory protocol).

    Returns:
        New ProcessManager instance.
    """
    _ = env  # Unused, but required by ResourceFactory protocol
    return ProcessManager()


# -----------------------------------------------------------------------------
# Resource Key Constant
# -----------------------------------------------------------------------------

PROCESS_MANAGER_KEY = "process_manager"
"""Standard key for registering ProcessManager in ResourceRegistry."""
