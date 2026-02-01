"""TUI Environment for paintress-cli.

TUIEnvironment extends LocalEnvironment with ProcessManager for
background process management. Process tools are provided via
ProcessManager.get_toolsets() and collected automatically.

Example:
    async with TUIEnvironment(default_path=Path.cwd()) as env:
        process = await env.process_manager.spawn("npm", ["run", "dev"])
        for info in env.process_manager.list_processes():
            print(f"{info.process_id}: {info.command}")
"""

from __future__ import annotations

from pathlib import Path

from agent_environment import ResourceFactory, ResourceRegistryState

from pai_agent_sdk.environment.local import LocalEnvironment
from paintress_cli.processes import PROCESS_MANAGER_KEY, ProcessManager


class TUIEnvironment(LocalEnvironment):
    """Extended environment for TUI with process management.

    ProcessManager is registered as a resource and accessible via:
    - env.process_manager (convenience property)
    - env.resources.get_typed(PROCESS_MANAGER_KEY, ProcessManager)
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
        await super()._setup()
        self._process_manager = ProcessManager()
        self.resources.set(PROCESS_MANAGER_KEY, self._process_manager)

    async def _teardown(self) -> None:
        self._process_manager = None
        await super()._teardown()

    @property
    def process_manager(self) -> ProcessManager:
        """Get the ProcessManager resource."""
        if self._process_manager is None:
            raise RuntimeError("TUIEnvironment not entered. Use 'async with TUIEnvironment() as env:'")
        return self._process_manager
