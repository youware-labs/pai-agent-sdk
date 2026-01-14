"""Tests for TUIEnvironment."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from paintress_cli.environment import TUIEnvironment
from paintress_cli.processes import PROCESS_MANAGER_KEY, ProcessManager


class TestTUIEnvironment:
    """Tests for TUIEnvironment."""

    @pytest.mark.asyncio
    async def test_enter_exit(self, tmp_path: Path) -> None:
        """TUIEnvironment should enter and exit cleanly."""
        async with TUIEnvironment(default_path=tmp_path) as env:
            assert env.file_operator is not None
            assert env.shell is not None
            assert env.resources is not None

    @pytest.mark.asyncio
    async def test_process_manager_available(self, tmp_path: Path) -> None:
        """ProcessManager should be available after entering."""
        async with TUIEnvironment(default_path=tmp_path) as env:
            # Via property
            pm = env.process_manager
            assert isinstance(pm, ProcessManager)

            # Via resources
            pm2 = env.resources.get_typed(PROCESS_MANAGER_KEY, ProcessManager)
            assert pm2 is pm

    @pytest.mark.asyncio
    async def test_process_manager_not_available_before_enter(self, tmp_path: Path) -> None:
        """process_manager should raise before entering."""
        env = TUIEnvironment(default_path=tmp_path)

        with pytest.raises(RuntimeError, match="not entered"):
            _ = env.process_manager

    @pytest.mark.asyncio
    async def test_spawn_process(self, tmp_path: Path) -> None:
        """Should be able to spawn processes."""
        async with TUIEnvironment(default_path=tmp_path) as env:
            process = await env.process_manager.spawn(
                sys.executable,
                ["-c", "import time; time.sleep(0.1)"],
                process_id="test-spawn",
            )

            assert process.process_id == "test-spawn"
            assert len(env.process_manager) == 1

    @pytest.mark.asyncio
    async def test_processes_killed_on_exit(self, tmp_path: Path) -> None:
        """All processes should be killed when exiting context."""
        pm_ref: ProcessManager | None = None

        async with TUIEnvironment(default_path=tmp_path) as env:
            await env.process_manager.spawn(
                sys.executable,
                ["-c", "import time; time.sleep(10)"],
                process_id="long-running",
            )
            pm_ref = env.process_manager

            # Process should be running
            proc = pm_ref.get_process("long-running")
            assert proc is not None
            assert proc.is_running is True

        # After exit, process should be killed
        # Note: pm_ref still exists but process should be terminated
        import asyncio

        await asyncio.sleep(0.1)
        proc = pm_ref.get_process("long-running")
        assert proc is not None
        assert proc.is_running is False

    @pytest.mark.asyncio
    async def test_inherits_local_environment_features(self, tmp_path: Path) -> None:
        """Should inherit file_operator and shell from LocalEnvironment."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        async with TUIEnvironment(default_path=tmp_path) as env:
            # File operator should work
            content = await env.file_operator.read_file("test.txt")
            assert content == "hello"

            # Shell should work
            exit_code, stdout, _ = await env.shell.execute("echo test")
            assert exit_code == 0
            assert "test" in stdout

    @pytest.mark.asyncio
    async def test_tmp_dir_created(self, tmp_path: Path) -> None:
        """Session tmp_dir should be created."""
        async with TUIEnvironment(default_path=tmp_path, enable_tmp_dir=True) as env:
            assert env.tmp_dir is not None
            assert env.tmp_dir.exists()

    @pytest.mark.asyncio
    async def test_tmp_dir_disabled(self, tmp_path: Path) -> None:
        """tmp_dir should be None when disabled."""
        async with TUIEnvironment(default_path=tmp_path, enable_tmp_dir=False) as env:
            assert env.tmp_dir is None
