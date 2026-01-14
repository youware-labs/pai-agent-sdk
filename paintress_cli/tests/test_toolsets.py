"""Tests for process toolset."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from paintress_cli.processes import ProcessManager
from paintress_cli.toolsets.process import (
    KillProcessTool,
    ListProcessesTool,
    ProcessToolBase,
    ReadProcessOutputTool,
    SpawnProcessTool,
    process_tools,
)
from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def process_manager() -> ProcessManager:
    """Create a ProcessManager instance."""
    return ProcessManager()


@pytest.fixture
def ctx_with_pm(process_manager: ProcessManager) -> MagicMock:
    """Create mock AgentContext with ProcessManager in resources."""
    # Create mock resources
    mock_resources = MagicMock()
    mock_resources.get.return_value = process_manager
    mock_resources.get_typed.return_value = process_manager

    # Create mock ctx with resources property
    mock_ctx = MagicMock()
    mock_ctx.resources = mock_resources

    return mock_ctx


@pytest.fixture
def run_ctx_with_pm(ctx_with_pm: MagicMock) -> RunContext[AgentContext]:
    """Create RunContext with ProcessManager."""
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = ctx_with_pm
    return run_ctx


@pytest.fixture
def ctx_no_pm() -> MagicMock:
    """Create mock AgentContext without ProcessManager."""
    mock_ctx = MagicMock()
    mock_ctx.resources = None
    return mock_ctx


@pytest.fixture
def run_ctx_no_pm(ctx_no_pm: MagicMock) -> RunContext[AgentContext]:
    """Create RunContext without ProcessManager."""
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = ctx_no_pm
    return run_ctx


# -----------------------------------------------------------------------------
# ProcessToolBase Tests
# -----------------------------------------------------------------------------


class TestProcessToolBase:
    """Tests for ProcessToolBase."""

    def test_is_available_with_pm(self, run_ctx_with_pm: RunContext[AgentContext]) -> None:
        """is_available should return True when ProcessManager exists."""

        class TestTool(ProcessToolBase):
            name = "test"
            description = "test"

            async def call(self, ctx: RunContext[AgentContext]) -> str:
                return "ok"

        tool = TestTool(run_ctx_with_pm.deps)
        assert tool.is_available(run_ctx_with_pm) is True

    def test_is_available_without_pm(self, run_ctx_no_pm: RunContext[AgentContext]) -> None:
        """is_available should return False when no ProcessManager."""

        class TestTool(ProcessToolBase):
            name = "test"
            description = "test"

            async def call(self, ctx: RunContext[AgentContext]) -> str:
                return "ok"

        tool = TestTool(run_ctx_no_pm.deps)
        assert tool.is_available(run_ctx_no_pm) is False


# -----------------------------------------------------------------------------
# SpawnProcessTool Tests
# -----------------------------------------------------------------------------


class TestSpawnProcessTool:
    """Tests for SpawnProcessTool."""

    def test_tool_metadata(self) -> None:
        """Tool should have correct name and description."""
        assert SpawnProcessTool.name == "spawn_process"
        assert "background process" in SpawnProcessTool.description.lower()

    @pytest.mark.asyncio
    async def test_spawn_process(
        self,
        run_ctx_with_pm: RunContext[AgentContext],
        process_manager: ProcessManager,
    ) -> None:
        """spawn_process should start a process and return info."""
        tool = SpawnProcessTool(run_ctx_with_pm.deps)

        try:
            result = await tool.call(
                run_ctx_with_pm,
                command=sys.executable,
                args=["-c", "import time; time.sleep(0.5)"],
                process_id="test-spawn",
            )

            assert result["process_id"] == "test-spawn"
            assert result["pid"] > 0
            assert "started" in result["message"].lower()
            assert len(process_manager) == 1

        finally:
            await process_manager.close()


# -----------------------------------------------------------------------------
# KillProcessTool Tests
# -----------------------------------------------------------------------------


class TestKillProcessTool:
    """Tests for KillProcessTool."""

    def test_tool_metadata(self) -> None:
        """Tool should have correct name and description."""
        assert KillProcessTool.name == "kill_process"
        assert "kill" in KillProcessTool.description.lower()

    @pytest.mark.asyncio
    async def test_kill_existing_process(
        self,
        run_ctx_with_pm: RunContext[AgentContext],
        process_manager: ProcessManager,
    ) -> None:
        """kill_process should terminate existing process."""
        # Spawn a process first
        await process_manager.spawn(
            sys.executable,
            ["-c", "import time; time.sleep(10)"],
            process_id="to-kill",
        )

        tool = KillProcessTool(run_ctx_with_pm.deps)

        try:
            result = await tool.call(run_ctx_with_pm, process_id="to-kill")

            assert result["success"] is True
            assert result["process_id"] == "to-kill"
            assert "terminated" in result["message"].lower()

        finally:
            await process_manager.close()

    @pytest.mark.asyncio
    async def test_kill_nonexistent_process(
        self,
        run_ctx_with_pm: RunContext[AgentContext],
    ) -> None:
        """kill_process should return failure for nonexistent process."""
        tool = KillProcessTool(run_ctx_with_pm.deps)

        result = await tool.call(run_ctx_with_pm, process_id="nonexistent")

        assert result["success"] is False
        assert "not found" in result["message"].lower()


# -----------------------------------------------------------------------------
# ListProcessesTool Tests
# -----------------------------------------------------------------------------


class TestListProcessesTool:
    """Tests for ListProcessesTool."""

    def test_tool_metadata(self) -> None:
        """Tool should have correct name and description."""
        assert ListProcessesTool.name == "list_processes"
        assert "list" in ListProcessesTool.description.lower()

    @pytest.mark.asyncio
    async def test_list_empty(
        self,
        run_ctx_with_pm: RunContext[AgentContext],
    ) -> None:
        """list_processes should return empty list when no processes."""
        tool = ListProcessesTool(run_ctx_with_pm.deps)

        result = await tool.call(run_ctx_with_pm)

        assert result["count"] == 0
        assert result["processes"] == []

    @pytest.mark.asyncio
    async def test_list_with_processes(
        self,
        run_ctx_with_pm: RunContext[AgentContext],
        process_manager: ProcessManager,
    ) -> None:
        """list_processes should return all processes with info."""
        await process_manager.spawn(
            sys.executable,
            ["-c", "import time; time.sleep(0.5)"],
            process_id="proc-1",
        )
        await process_manager.spawn(
            sys.executable,
            ["-c", "import time; time.sleep(0.5)"],
            process_id="proc-2",
        )

        tool = ListProcessesTool(run_ctx_with_pm.deps)

        try:
            result = await tool.call(run_ctx_with_pm)

            assert result["count"] == 2
            assert len(result["processes"]) == 2

            proc_ids = [p["process_id"] for p in result["processes"]]
            assert "proc-1" in proc_ids
            assert "proc-2" in proc_ids

            # Check fields
            proc = result["processes"][0]
            assert "pid" in proc
            assert "is_running" in proc
            assert "elapsed_seconds" in proc

        finally:
            await process_manager.close()


# -----------------------------------------------------------------------------
# ReadProcessOutputTool Tests
# -----------------------------------------------------------------------------


class TestReadProcessOutputTool:
    """Tests for ReadProcessOutputTool."""

    def test_tool_metadata(self) -> None:
        """Tool should have correct name and description."""
        assert ReadProcessOutputTool.name == "read_process_output"
        assert "output" in ReadProcessOutputTool.description.lower()

    @pytest.mark.asyncio
    async def test_read_output(
        self,
        run_ctx_with_pm: RunContext[AgentContext],
        process_manager: ProcessManager,
    ) -> None:
        """read_process_output should return stdout and stderr."""
        import asyncio

        managed = await process_manager.spawn(
            sys.executable,
            ["-c", "print('hello'); import sys; print('error', file=sys.stderr)"],
            process_id="output-test",
        )

        # Wait for process to complete and output to be captured
        await managed.wait()
        await asyncio.sleep(0.1)

        tool = ReadProcessOutputTool(run_ctx_with_pm.deps)

        try:
            result = await tool.call(run_ctx_with_pm, process_id="output-test")

            assert result["process_id"] == "output-test"
            assert result["lines_read"] >= 2
            assert "hello" in result["stdout"]
            assert "error" in result["stderr"]

        finally:
            await process_manager.close()

    @pytest.mark.asyncio
    async def test_read_output_with_limit(
        self,
        run_ctx_with_pm: RunContext[AgentContext],
        process_manager: ProcessManager,
    ) -> None:
        """read_process_output should respect max_lines limit."""
        import asyncio

        # Spawn process that outputs multiple lines
        managed = await process_manager.spawn(
            sys.executable,
            ["-c", "for i in range(10): print(f'line{i}')"],
            process_id="multi-line",
        )

        await managed.wait()
        await asyncio.sleep(0.1)

        tool = ReadProcessOutputTool(run_ctx_with_pm.deps)

        try:
            # Read only 3 lines
            result = await tool.call(
                run_ctx_with_pm,
                process_id="multi-line",
                max_lines=3,
            )

            assert result["lines_read"] == 3
            assert result["has_more"] is True
            assert len(result["stdout"]) == 3

            # Read remaining
            result2 = await tool.call(
                run_ctx_with_pm,
                process_id="multi-line",
                max_lines=100,
            )

            assert result2["lines_read"] == 7  # Remaining lines
            assert result2["has_more"] is False

        finally:
            await process_manager.close()

    @pytest.mark.asyncio
    async def test_read_nonexistent_process(
        self,
        run_ctx_with_pm: RunContext[AgentContext],
    ) -> None:
        """read_process_output should raise error for nonexistent process."""
        from pydantic_ai import UserError

        tool = ReadProcessOutputTool(run_ctx_with_pm.deps)

        with pytest.raises(UserError, match="not found"):
            await tool.call(run_ctx_with_pm, process_id="nonexistent")


# -----------------------------------------------------------------------------
# Module Level Tests
# -----------------------------------------------------------------------------


class TestProcessTools:
    """Tests for process_tools list."""

    def test_all_tools_included(self) -> None:
        """process_tools should include all tool classes."""
        assert SpawnProcessTool in process_tools
        assert KillProcessTool in process_tools
        assert ListProcessesTool in process_tools
        assert ReadProcessOutputTool in process_tools

    def test_tools_are_base_tool_subclasses(self) -> None:
        """All tools should inherit from BaseTool."""
        from pai_agent_sdk.toolsets.core.base import BaseTool

        for tool_cls in process_tools:
            assert issubclass(tool_cls, BaseTool)
