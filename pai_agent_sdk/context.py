import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

from pai_agent_sdk._config import AgentContextSettings

if TYPE_CHECKING:
    from typing import Self


def _generate_run_id() -> str:
    return uuid4().hex


def _get_default_working_dir() -> Path:
    settings = AgentContextSettings()
    return settings.working_dir if settings.working_dir else Path.cwd()


def _get_default_tmp_base_dir() -> Path | None:
    settings = AgentContextSettings()
    return settings.tmp_base_dir


class AgentContext(BaseModel):
    run_id: str = Field(default_factory=_generate_run_id)
    parent_run_id: str | None = None
    start_at: datetime | None = None
    end_at: datetime | None = None

    deferred_tool_metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Metadata for deferred tool calls, keyed by tool_call_id."""

    handoff_message: str | None = None
    """Rendered handoff message to be injected into new context after handoff."""

    working_dir: Path = Field(default_factory=_get_default_working_dir)
    """Working directory for tool path validation. Tools should not access paths outside this directory."""

    tmp_base_dir: Path | None = Field(default_factory=_get_default_tmp_base_dir)
    """Base directory for creating the session temporary directory. If None, uses system default."""

    _agent_name: str = "main"
    _tmp_dir: tempfile.TemporaryDirectory[str] | None = None

    @property
    def elapsed_time(self) -> timedelta | None:
        """Return elapsed time since start, or None if not started.

        If session has ended, returns the final duration.
        If session is running, returns the current elapsed time.
        """
        if self.start_at is None:
            return None
        end = self.end_at if self.end_at else datetime.now()
        return end - self.start_at

    @property
    def tmp_dir(self) -> Path:
        """Return the session-level temporary directory path.

        The temporary directory is created on first access (lazy initialization)
        and cleaned up when the context exits.

        Raises:
            RuntimeError: If accessed before context is entered (before __aenter__).
        """
        if self._tmp_dir is None:
            raise RuntimeError("tmp_dir is not available. Use 'async with context:' to enter the context first.")
        return Path(self._tmp_dir.name)

    def is_within_working_dir(self, path: str | Path) -> bool:
        """Check if a path is within the working directory.

        Args:
            path: The path to check (absolute or relative).

        Returns:
            True if the resolved path is within working_dir, False otherwise.
        """
        target = Path(path)
        if not target.is_absolute():
            target = self.working_dir / target
        try:
            target = target.resolve()
            working = self.working_dir.resolve()
            return target == working or working in target.parents
        except (OSError, ValueError):
            return False

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve a path relative to the working directory.

        Args:
            path: The path to resolve (absolute or relative).

        Returns:
            The resolved absolute path.

        Raises:
            ValueError: If the resolved path is outside the working directory.
        """
        target = Path(path)
        if not target.is_absolute():
            target = self.working_dir / target
        resolved = target.resolve()
        if not self.is_within_working_dir(resolved):
            raise ValueError(
                f"Path '{path}' resolves to '{resolved}' which is outside working directory '{self.working_dir}'"
            )
        return resolved

    def relative_path(self, path: str | Path) -> Path:
        """Get the path relative to the working directory.

        Args:
            path: The path to convert (absolute or relative).

        Returns:
            The path relative to working_dir.

        Raises:
            ValueError: If the path is outside the working directory.
        """
        resolved = self.resolve_path(path)
        return resolved.relative_to(self.working_dir.resolve())

    @asynccontextmanager
    async def enter_subagent(
        self,
        agent_name: str,
        **override: Any,
    ) -> AsyncGenerator["Self", None]:
        """Create a child context for subagent with independent timing.

        The subagent context inherits all fields but gets:
        - A new run_id
        - parent_run_id set to current run_id
        - Fresh start_at/end_at for independent timing
        - Shared working_dir and tmp_dir from parent

        Args:
            agent_name: Name of the subagent.
            **override: Additional fields to override in the subagent context.
                Subclasses can pass extra fields without overriding this method.
        """
        update: dict[str, Any] = {
            "run_id": _generate_run_id(),
            "parent_run_id": self.run_id,
            "start_at": datetime.now(),
            "end_at": None,
            "handoff_message": None,  # Subagents don't inherit handoff state
            **override,
        }
        new_ctx = self.model_copy(update=update)
        new_ctx._agent_name = agent_name
        new_ctx._tmp_dir = self._tmp_dir  # Share tmp_dir with subagent
        try:
            yield new_ctx
        finally:
            new_ctx.end_at = datetime.now()

    async def __aenter__(self):
        self.start_at = datetime.now()
        self._tmp_dir = tempfile.TemporaryDirectory(
            prefix="pai_agent_",
            dir=str(self.tmp_base_dir) if self.tmp_base_dir else None,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.end_at = datetime.now()
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None
