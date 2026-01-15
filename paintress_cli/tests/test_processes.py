"""Tests for ProcessManager resource."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from paintress_cli.processes import (
    PROCESS_MANAGER_KEY,
    ManagedProcess,
    ProcessInfo,
    ProcessManager,
    create_process_manager,
)

# -----------------------------------------------------------------------------
# ProcessInfo Tests
# -----------------------------------------------------------------------------


class TestProcessInfo:
    """Tests for ProcessInfo dataclass."""

    def test_process_info_creation(self) -> None:
        """ProcessInfo should hold process metadata."""
        from datetime import datetime

        info = ProcessInfo(
            process_id="proc-123",
            command="python",
            args=["-c", "print('hello')"],
            pid=12345,
            started_at=datetime.now(),
            is_running=True,
            exit_code=None,
        )

        assert info.process_id == "proc-123"
        assert info.command == "python"
        assert info.args == ["-c", "print('hello')"]
        assert info.pid == 12345
        assert info.is_running is True
        assert info.exit_code is None


# -----------------------------------------------------------------------------
# ManagedProcess Tests
# -----------------------------------------------------------------------------


class TestManagedProcess:
    """Tests for ManagedProcess."""

    @pytest.fixture
    def mock_process(self) -> MagicMock:
        """Create a mock asyncio.subprocess.Process."""
        proc = MagicMock()
        proc.pid = 12345
        proc.returncode = None
        proc.terminate = MagicMock()
        proc.kill = MagicMock()
        proc.wait = AsyncMock(return_value=0)
        return proc

    def test_managed_process_properties(self, mock_process: MagicMock) -> None:
        """ManagedProcess should expose process properties."""
        managed = ManagedProcess(
            process_id="test-1",
            command="echo",
            args=["hello"],
            process=mock_process,
        )

        assert managed.pid == 12345
        assert managed.is_running is True
        assert managed.exit_code is None

    def test_managed_process_exit_code(self, mock_process: MagicMock) -> None:
        """ManagedProcess should report exit code when process exits."""
        mock_process.returncode = 0
        managed = ManagedProcess(
            process_id="test-1",
            command="echo",
            args=["hello"],
            process=mock_process,
        )

        assert managed.is_running is False
        assert managed.exit_code == 0

    def test_to_info(self, mock_process: MagicMock) -> None:
        """to_info should convert to ProcessInfo."""
        managed = ManagedProcess(
            process_id="test-1",
            command="echo",
            args=["hello"],
            process=mock_process,
        )

        info = managed.to_info()
        assert isinstance(info, ProcessInfo)
        assert info.process_id == "test-1"
        assert info.command == "echo"
        assert info.args == ["hello"]
        assert info.pid == 12345

    @pytest.mark.asyncio
    async def test_read_output_empty(self, mock_process: MagicMock) -> None:
        """read_output should return empty list when no output."""
        managed = ManagedProcess(
            process_id="test-1",
            command="echo",
            args=[],
            process=mock_process,
        )

        output = await managed.read_output()
        assert output == []

    @pytest.mark.asyncio
    async def test_read_output_with_data(self, mock_process: MagicMock) -> None:
        """read_output should return and clear buffered output."""
        managed = ManagedProcess(
            process_id="test-1",
            command="echo",
            args=[],
            process=mock_process,
        )

        # Simulate buffered output
        await managed._append_output("line1", False)
        await managed._append_output("error1", True)

        output = await managed.read_output()
        assert output == [("line1", False), ("error1", True)]

        # Buffer should be cleared
        output2 = await managed.read_output()
        assert output2 == []

    @pytest.mark.asyncio
    async def test_read_output_with_limit(self, mock_process: MagicMock) -> None:
        """read_output should respect max_lines limit."""
        managed = ManagedProcess(
            process_id="test-1",
            command="echo",
            args=[],
            process=mock_process,
        )

        await managed._append_output("line1", False)
        await managed._append_output("line2", False)
        await managed._append_output("line3", False)

        output = await managed.read_output(max_lines=2)
        assert len(output) == 2
        assert output[0] == ("line1", False)

        # Remaining should still be buffered
        remaining = await managed.read_output()
        assert remaining == [("line3", False)]

    @pytest.mark.asyncio
    async def test_kill_already_exited(self, mock_process: MagicMock) -> None:
        """kill should return immediately if process already exited."""
        mock_process.returncode = 0
        managed = ManagedProcess(
            process_id="test-1",
            command="echo",
            args=[],
            process=mock_process,
        )

        exit_code = await managed.kill()
        assert exit_code == 0
        mock_process.terminate.assert_not_called()

    @pytest.mark.asyncio
    async def test_kill_graceful(self, mock_process: MagicMock) -> None:
        """kill should try graceful termination first."""
        # Mock wait to set returncode and return
        wait_call_count = 0

        async def wait_side_effect() -> int:
            nonlocal wait_call_count
            wait_call_count += 1
            # First wait call (from wait_for) sets returncode
            mock_process.returncode = 0
            return 0

        mock_process.wait = AsyncMock(side_effect=wait_side_effect)

        managed = ManagedProcess(
            process_id="test-1",
            command="echo",
            args=[],
            process=mock_process,
        )

        exit_code = await managed.kill()
        # After kill, returncode should be set
        assert mock_process.returncode == 0
        assert exit_code == 0
        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_not_called()


# -----------------------------------------------------------------------------
# ProcessManager Tests
# -----------------------------------------------------------------------------


class TestProcessManager:
    """Tests for ProcessManager resource."""

    @pytest.mark.asyncio
    async def test_initial_state(self) -> None:
        """ProcessManager should start with no processes."""
        pm = ProcessManager()
        assert len(pm) == 0
        assert pm.list_processes() == []

    @pytest.mark.asyncio
    async def test_spawn_and_list(self) -> None:
        """spawn should create and track a process."""
        pm = ProcessManager()

        try:
            # Spawn a simple command
            managed = await pm.spawn(
                sys.executable,
                ["-c", "import time; time.sleep(0.1)"],
            )

            assert managed.process_id in pm
            assert len(pm) == 1

            processes = pm.list_processes()
            assert len(processes) == 1
            assert processes[0].command == sys.executable
            assert processes[0].is_running is True

        finally:
            await pm.close()

    @pytest.mark.asyncio
    async def test_spawn_with_custom_id(self) -> None:
        """spawn should use custom process_id if provided."""
        pm = ProcessManager()

        try:
            managed = await pm.spawn(
                sys.executable,
                ["-c", "pass"],
                process_id="my-custom-id",
            )

            assert managed.process_id == "my-custom-id"
            assert "my-custom-id" in pm
        finally:
            await pm.close()

    @pytest.mark.asyncio
    async def test_get_process(self) -> None:
        """get_process should return process by ID."""
        pm = ProcessManager()

        try:
            managed = await pm.spawn(
                sys.executable,
                ["-c", "pass"],
                process_id="test-proc",
            )

            retrieved = pm.get_process("test-proc")
            assert retrieved is managed

            assert pm.get_process("nonexistent") is None
        finally:
            await pm.close()

    @pytest.mark.asyncio
    async def test_kill_process(self) -> None:
        """kill should terminate a specific process."""
        pm = ProcessManager()

        try:
            await pm.spawn(
                sys.executable,
                ["-c", "import time; time.sleep(10)"],
                process_id="to-kill",
            )

            proc = pm.get_process("to-kill")
            assert proc is not None
            assert proc.is_running is True

            result = await pm.kill("to-kill")
            assert result is True

            # Give process time to terminate
            await asyncio.sleep(0.1)
            proc = pm.get_process("to-kill")
            assert proc is not None
            assert proc.is_running is False

        finally:
            await pm.close()

    @pytest.mark.asyncio
    async def test_kill_nonexistent(self) -> None:
        """kill should return False for nonexistent process."""
        pm = ProcessManager()

        result = await pm.kill("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_kill_all(self) -> None:
        """kill_all should terminate all processes."""
        pm = ProcessManager()

        try:
            await pm.spawn(
                sys.executable,
                ["-c", "import time; time.sleep(10)"],
                process_id="proc-1",
            )
            await pm.spawn(
                sys.executable,
                ["-c", "import time; time.sleep(10)"],
                process_id="proc-2",
            )

            assert len(pm) == 2

            await pm.kill_all()

            # Give processes time to terminate
            await asyncio.sleep(0.1)

            for info in pm.list_processes():
                assert info.is_running is False

        finally:
            await pm.close()

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """close should kill all processes."""
        pm = ProcessManager()

        await pm.spawn(
            sys.executable,
            ["-c", "import time; time.sleep(10)"],
        )

        await pm.close()

        # After close, processes should be terminated
        await asyncio.sleep(0.1)
        for info in pm.list_processes():
            assert info.is_running is False

    @pytest.mark.asyncio
    async def test_output_capture(self) -> None:
        """spawn should capture stdout/stderr."""
        pm = ProcessManager()

        try:
            managed = await pm.spawn(
                sys.executable,
                ["-c", "print('hello'); import sys; print('error', file=sys.stderr)"],
            )

            # Wait for process to complete
            await managed.wait()

            # Give time for output to be captured
            await asyncio.sleep(0.1)

            output = await managed.read_output()
            # Should have captured both stdout and stderr
            stdout_lines = [line for line, is_err in output if not is_err]
            stderr_lines = [line for line, is_err in output if is_err]

            assert "hello" in stdout_lines
            assert "error" in stderr_lines

        finally:
            await pm.close()

    @pytest.mark.asyncio
    async def test_get_context_instructions_no_processes(self) -> None:
        """get_context_instructions should return None when no processes."""
        pm = ProcessManager()

        instructions = await pm.get_context_instructions()
        assert instructions is None

    @pytest.mark.asyncio
    async def test_get_context_instructions_with_processes(self) -> None:
        """get_context_instructions should list processes in XML format."""
        pm = ProcessManager()

        try:
            await pm.spawn(
                sys.executable,
                ["-c", "import time; time.sleep(10)"],
                process_id="dev-server",
            )

            instructions = await pm.get_context_instructions()
            assert instructions is not None
            # Should be XML format with hint attribute
            assert "<processes hint=" in instructions
            assert '<process id="dev-server" status="running">' in instructions
            assert "<command>" in instructions
            assert "<pid>" in instructions
            assert "<elapsed_seconds>" in instructions
            # Hint should mention available tools
            assert "spawn_process" in instructions
            assert "kill_process" in instructions

        finally:
            await pm.close()

    @pytest.mark.asyncio
    async def test_get_context_instructions_with_exited_process(self) -> None:
        """get_context_instructions should include exit code for exited processes."""
        pm = ProcessManager()

        try:
            managed = await pm.spawn(
                sys.executable,
                ["-c", "pass"],  # Exits immediately with code 0
                process_id="quick-task",
            )

            # Wait for process to complete
            await managed.wait()
            await asyncio.sleep(0.1)

            instructions = await pm.get_context_instructions()
            assert instructions is not None
            assert '<process id="quick-task" status="exited">' in instructions
            assert "<exit_code>0</exit_code>" in instructions

        finally:
            await pm.close()


# -----------------------------------------------------------------------------
# Factory Tests
# -----------------------------------------------------------------------------


class TestFactory:
    """Tests for factory function."""

    @pytest.mark.asyncio
    async def test_create_process_manager(self) -> None:
        """create_process_manager should create a new ProcessManager."""
        mock_env = MagicMock()

        pm = await create_process_manager(mock_env)

        assert isinstance(pm, ProcessManager)
        assert len(pm) == 0

    def test_process_manager_key(self) -> None:
        """PROCESS_MANAGER_KEY should be the standard key."""
        assert PROCESS_MANAGER_KEY == "process_manager"
