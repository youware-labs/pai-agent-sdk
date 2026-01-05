"""Tests for pai_agent_sdk.context module."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from pai_agent_sdk.context import AgentContext


def test_agent_context_default_run_id() -> None:
    """Should generate a unique run_id by default."""
    ctx1 = AgentContext()
    ctx2 = AgentContext()
    assert ctx1.run_id != ctx2.run_id
    assert len(ctx1.run_id) == 32  # uuid4().hex length


def test_agent_context_no_parent_by_default() -> None:
    """Should have no parent by default."""
    ctx = AgentContext()
    assert ctx.parent_run_id is None


def test_agent_context_elapsed_time_before_start() -> None:
    """Should return None before context is started."""
    ctx = AgentContext()
    assert ctx.elapsed_time is None


def test_agent_context_elapsed_time_after_start() -> None:
    """Should return elapsed time after start."""
    ctx = AgentContext()
    ctx.start_at = datetime.now()
    elapsed = ctx.elapsed_time
    assert elapsed is not None
    assert isinstance(elapsed, timedelta)
    assert elapsed.total_seconds() >= 0


def test_agent_context_elapsed_time_after_end() -> None:
    """Should return final duration after end."""
    ctx = AgentContext()
    start = datetime.now()
    ctx.start_at = start
    ctx.end_at = start + timedelta(seconds=5)
    elapsed = ctx.elapsed_time
    assert elapsed is not None
    assert elapsed.total_seconds() == 5


async def test_agent_context_enter_subagent() -> None:
    """Should create child context with proper inheritance."""
    parent = AgentContext()
    parent.start_at = datetime.now()

    async with parent.enter_subagent("search") as child:
        assert child.parent_run_id == parent.run_id
        assert child.run_id != parent.run_id
        assert child._agent_name == "search"
        assert child.start_at is not None
        assert child.end_at is None

    # After exiting, end_at should be set
    assert child.end_at is not None


async def test_agent_context_enter_subagent_with_override() -> None:
    """Should allow field overrides in subagent context."""
    parent = AgentContext()

    async with parent.enter_subagent("reasoning", deferred_tool_metadata={"key": {}}) as child:
        assert child.deferred_tool_metadata == {"key": {}}


@pytest.mark.asyncio
async def test_agent_context_async_context_manager() -> None:
    """Should set start/end times in async context."""
    ctx = AgentContext()
    assert ctx.start_at is None
    assert ctx.end_at is None

    async with ctx:
        assert ctx.start_at is not None
        assert ctx.end_at is None

    assert ctx.end_at is not None
    assert ctx.end_at >= ctx.start_at


def test_agent_context_deferred_tool_metadata_default() -> None:
    """Should have empty metadata by default."""
    ctx = AgentContext()
    assert ctx.deferred_tool_metadata == {}


def test_agent_context_deferred_tool_metadata_storage() -> None:
    """Should store metadata by tool_call_id."""
    ctx = AgentContext()
    ctx.deferred_tool_metadata["call-1"] = {"user_choice": "option_a"}
    assert ctx.deferred_tool_metadata["call-1"]["user_choice"] == "option_a"


def test_agent_context_working_dir_default() -> None:
    """Should default to current working directory."""
    ctx = AgentContext()
    assert ctx.working_dir == Path.cwd()


def test_agent_context_working_dir_custom(tmp_path: Path) -> None:
    """Should accept custom working directory."""
    custom_path = tmp_path / "custom"
    custom_path.mkdir()
    ctx = AgentContext(working_dir=custom_path)
    assert ctx.working_dir == custom_path


def test_agent_context_is_within_working_dir(tmp_path: Path) -> None:
    """Should correctly validate paths within working directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = AgentContext(working_dir=workspace)

    # Paths within working dir
    assert ctx.is_within_working_dir(str(workspace)) is True
    assert ctx.is_within_working_dir(str(workspace / "subdir")) is True
    assert ctx.is_within_working_dir(str(workspace / "subdir" / "file.txt")) is True

    # Relative paths (resolved against working dir)
    assert ctx.is_within_working_dir("subdir") is True
    assert ctx.is_within_working_dir("subdir/file.txt") is True

    # Paths outside working dir
    assert ctx.is_within_working_dir(str(tmp_path)) is False
    assert ctx.is_within_working_dir(str(tmp_path / "other")) is False
    assert ctx.is_within_working_dir("/etc/passwd") is False


def test_agent_context_is_within_working_dir_path_traversal(tmp_path: Path) -> None:
    """Should reject path traversal attempts."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = AgentContext(working_dir=workspace)

    # Path traversal attempts
    assert ctx.is_within_working_dir(str(workspace / ".." / "other")) is False
    assert ctx.is_within_working_dir("../other") is False
    assert ctx.is_within_working_dir("subdir/../../other") is False


def test_agent_context_resolve_path(tmp_path: Path) -> None:
    """Should resolve paths relative to working directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = AgentContext(working_dir=workspace)

    # Relative path
    resolved = ctx.resolve_path("subdir/file.txt")
    assert resolved == (workspace / "subdir" / "file.txt").resolve()

    # Absolute path within working dir
    subdir = workspace / "subdir"
    resolved = ctx.resolve_path(str(subdir))
    assert resolved == subdir.resolve()


def test_agent_context_resolve_path_raises_on_escape(tmp_path: Path) -> None:
    """Should raise ValueError when path escapes working directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = AgentContext(working_dir=workspace)

    with pytest.raises(ValueError, match="outside working directory"):
        ctx.resolve_path("/etc/passwd")

    with pytest.raises(ValueError, match="outside working directory"):
        ctx.resolve_path("../other")


def test_agent_context_relative_path(tmp_path: Path) -> None:
    """Should return path relative to working directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = AgentContext(working_dir=workspace)

    # Relative path stays relative
    assert ctx.relative_path("subdir/file.txt") == Path("subdir/file.txt")

    # Absolute path becomes relative
    abs_path = workspace / "subdir" / "file.txt"
    assert ctx.relative_path(str(abs_path)) == Path("subdir/file.txt")

    # Working dir itself returns empty path
    assert ctx.relative_path(str(workspace)) == Path(".")


def test_agent_context_relative_path_raises_on_escape(tmp_path: Path) -> None:
    """Should raise ValueError when path escapes working directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = AgentContext(working_dir=workspace)

    with pytest.raises(ValueError, match="outside working directory"):
        ctx.relative_path("/etc/passwd")


@pytest.mark.asyncio
async def test_agent_context_tmp_dir() -> None:
    """Should create and cleanup temporary directory."""
    ctx = AgentContext()

    # tmp_dir should not be available before entering context
    with pytest.raises(RuntimeError, match="tmp_dir is not available"):
        _ = ctx.tmp_dir

    async with ctx:
        tmp_path = ctx.tmp_dir
        assert tmp_path.exists()
        assert tmp_path.is_dir()
        assert "pai_agent_" in tmp_path.name

        # Create a file to verify cleanup later
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        assert test_file.exists()

    # After exit, tmp_dir should be cleaned up
    assert not tmp_path.exists()


@pytest.mark.asyncio
async def test_agent_context_tmp_dir_custom_base(tmp_path: Path) -> None:
    """Should create temporary directory in custom base directory."""
    custom_base = tmp_path / "custom_tmp"
    custom_base.mkdir()
    ctx = AgentContext(tmp_base_dir=custom_base)

    async with ctx:
        session_tmp = ctx.tmp_dir
        assert session_tmp.exists()
        assert session_tmp.parent == custom_base
        assert "pai_agent_" in session_tmp.name

    assert not session_tmp.exists()


@pytest.mark.asyncio
async def test_agent_context_subagent_shares_dirs(tmp_path: Path) -> None:
    """Subagent should share working_dir and tmp_dir with parent."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = AgentContext(working_dir=workspace)

    async with ctx:
        parent_tmp = ctx.tmp_dir

        async with ctx.enter_subagent("search") as child:
            # Should share working_dir
            assert child.working_dir == ctx.working_dir

            # Should share tmp_dir
            assert child.tmp_dir == parent_tmp
