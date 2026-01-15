"""TUI Environment for paintress-cli.

TUIEnvironment extends LocalEnvironment with capabilities specifically
designed for TUI operation:

1. ProcessManager - Track and control background processes
2. Process Tools - Tool classes for process management via env.toolsets
3. Browser Integration - Optional headless browser sandbox (future)

Example:
    async with TUIEnvironment(default_path=Path.cwd()) as env:
        # Spawn a background process
        process = await env.process_manager.spawn("npm", ["run", "dev"])

        # List running processes
        for info in env.process_manager.list_processes():
            print(f"{info.process_id}: {info.command}")

        # Use env.toolsets with Agent
        from pai_agent_sdk.toolsets.core.base import Toolset
        agent = Agent(..., toolsets=[*core_toolsets, *env.toolsets])

    # All processes automatically killed on exit
"""

from __future__ import annotations

from pathlib import Path

from agent_environment import ResourceFactory, ResourceRegistryState

from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.toolsets.core.base import Toolset
from paintress_cli.processes import PROCESS_MANAGER_KEY, ProcessManager
from paintress_cli.toolsets.process import process_tools


class TUIEnvironment(LocalEnvironment):
    """Extended environment for TUI with process management.

    Inherits from LocalEnvironment and adds:
    - ProcessManager resource for background process management
    - Process tool classes available via env.toolsets
    - Automatic cleanup of all processes on exit

    The ProcessManager is registered as a resource and can be accessed via:
    - env.process_manager (convenience property)
    - env.resources.get_typed(PROCESS_MANAGER_KEY, ProcessManager)
    - ctx.deps.resources.get_typed(...) in tools
    """

    def __init__(
        self,
        allowed_paths: list[Path] | None = None,
        default_path: Path | None = None,
        shell_timeout: float = 30.0,
        tmp_base_dir: Path | None = None,
        enable_tmp_dir: bool = True,
        resource_state: ResourceRegistryState | None = None,
        resource_factories: dict[str, ResourceFactory] | None = None,
    ) -> None:
        """Initialize TUIEnvironment.

        Args:
            allowed_paths: Directories accessible by file and shell operations.
            default_path: Default working directory for operations.
            shell_timeout: Default shell command timeout in seconds.
            tmp_base_dir: Base directory for session temporary directory.
            enable_tmp_dir: Whether to create a session temporary directory.
            resource_state: Optional state to restore resources from.
            resource_factories: Optional dictionary of resource factories.
        """
        super().__init__(
            allowed_paths=allowed_paths,
            default_path=default_path,
            shell_timeout=shell_timeout,
            tmp_base_dir=tmp_base_dir,
            enable_tmp_dir=enable_tmp_dir,
            resource_state=resource_state,
            resource_factories=resource_factories,
        )
        self._process_manager: ProcessManager | None = None

    async def _setup(self) -> None:
        """Initialize file operator, shell, register ProcessManager, and setup toolsets."""
        await super()._setup()

        # Create and register ProcessManager as a resource
        self._process_manager = ProcessManager()
        self.resources.set(PROCESS_MANAGER_KEY, self._process_manager)

        # Register tool classes for TUI environment
        self._toolsets.append(
            Toolset(
                tools=process_tools,
                toolset_id="process",
            )
        )

    async def _teardown(self) -> None:
        """Clean up resources.

        Note: ProcessManager.close() is called automatically by
        resources.close_all() in the parent class __aexit__.
        """
        self._process_manager = None
        self._toolsets = []
        await super()._teardown()

    @property
    def process_manager(self) -> ProcessManager:
        """Get the ProcessManager resource.

        Returns:
            ProcessManager instance.

        Raises:
            RuntimeError: If environment not entered (use async with).
        """
        if self._process_manager is None:
            raise RuntimeError("TUIEnvironment not entered. Use 'async with TUIEnvironment() as env:'")
        return self._process_manager
