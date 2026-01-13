"""Docker environment implementation.

This module provides a Docker-based environment that:
- Uses docker exec for shell commands (runs inside container)
- Uses local filesystem for file operations (via mount directory)
- Supports existing containers or creating new ones
- Supports optional container cleanup on exit for cross-session sharing

Architecture:
    - File operations: Local filesystem at mount_dir (host side)
    - Shell execution: Docker exec inside container at container_workdir
    - Mount relationship: host mount_dir <-> container container_workdir
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from pai_agent_sdk.environment.base import Environment, ResourceFactory, ResourceRegistryState, Shell
from pai_agent_sdk.environment.exceptions import (
    EnvironmentNotEnteredError,
    ShellExecutionError,
    ShellTimeoutError,
)
from pai_agent_sdk.environment.local import LocalFileOperator

if TYPE_CHECKING:
    pass

try:
    import docker
    import docker.errors
except ImportError as e:
    raise ImportError(
        "The 'docker' package is required for DockerEnvironment. Install it with: pip install pai-agent-sdk[docker]"
    ) from e


class DockerShell(Shell):
    """Shell implementation that executes commands inside a Docker container.

    Uses docker exec to run commands in the specified container.
    The working directory inside the container is specified by container_workdir.
    """

    def __init__(
        self,
        container_id: str,
        container_workdir: str = "/workspace",
        default_timeout: float = 30.0,
    ):
        """Initialize DockerShell.

        Args:
            container_id: Docker container ID to execute commands in.
            container_workdir: Working directory inside the container.
            default_timeout: Default timeout in seconds.
        """
        # DockerShell doesn't use allowed_paths or default_cwd from base Shell
        # since path validation happens inside the container
        super().__init__(
            default_cwd=Path(container_workdir),
            allowed_paths=None,
            default_timeout=default_timeout,
        )
        self._container_id = container_id
        self._container_workdir = container_workdir
        self._client: docker.DockerClient | None = None

    @property
    def client(self) -> docker.DockerClient:
        """Get Docker client with lazy initialization."""
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> tuple[int, str, str]:
        """Execute a command inside the Docker container.

        Args:
            command: Command string to execute via shell.
            timeout: Execution timeout in seconds.
            env: Environment variables for the command.
            cwd: Working directory (relative to container_workdir, or absolute).

        Returns:
            Tuple of (exit_code, stdout, stderr).
        """
        if not command:
            raise ShellExecutionError(command, stderr="Empty command")

        effective_timeout = timeout if timeout is not None else self._default_timeout

        # Determine working directory inside container
        if cwd is not None:
            workdir = cwd if cwd.startswith("/") else f"{self._container_workdir}/{cwd}"
        else:
            workdir = self._container_workdir

        def _exec_command() -> tuple[int, str, str]:
            try:
                container = self.client.containers.get(self._container_id)
                result = container.exec_run(
                    cmd=["/bin/sh", "-c", command],
                    stdout=True,
                    stderr=True,
                    demux=True,
                    workdir=workdir,
                    environment=env,
                )

                exit_code: int = result.exit_code
                stdout_stderr = result.output

                if isinstance(stdout_stderr, tuple) and len(stdout_stderr) == 2:
                    out, err = stdout_stderr
                    stdout_bytes = out if out is not None else b""
                    stderr_bytes = err if err is not None else b""
                else:
                    stdout_bytes = bytes(stdout_stderr) if stdout_stderr is not None else b""
                    stderr_bytes = b""

                stdout = stdout_bytes.decode("utf-8", errors="replace")
                stderr = stderr_bytes.decode("utf-8", errors="replace")
                return (exit_code, stdout, stderr)

            except docker.errors.NotFound as e:
                raise ShellExecutionError(
                    command,
                    stderr=f"Container not found: {self._container_id}",
                ) from e
            except docker.errors.APIError as e:
                raise ShellExecutionError(command, stderr=str(e)) from e

        try:
            loop = asyncio.get_running_loop()
            if effective_timeout > 0:
                return await asyncio.wait_for(
                    loop.run_in_executor(None, _exec_command),
                    timeout=effective_timeout,
                )
            else:
                return await loop.run_in_executor(None, _exec_command)
        except TimeoutError as e:
            raise ShellTimeoutError(command, effective_timeout) from e

    async def get_context_instructions(self) -> str | None:
        """Return instructions for the agent about shell capabilities."""
        return f"""<shell-execution>
  <type>docker-exec</type>
  <container-id>{self._container_id}</container-id>
  <container-workdir>{self._container_workdir}</container-workdir>
  <default-timeout>{self._default_timeout}s</default-timeout>
  <note>Commands are executed inside the Docker container via docker exec.</note>
  <note>File edits are performed on the host filesystem, which is mounted into the container.</note>
</shell-execution>"""


class DockerEnvironment(Environment):
    """Docker-based environment with local file operations and containerized shell.

    This environment provides:
    - File operations via local filesystem (at mount_dir on host)
    - Shell execution via docker exec (inside container at container_workdir)
    - Support for existing containers (pass container_id)
    - Support for creating new containers (pass image)
    - Optional container cleanup on exit (cleanup_on_exit=False for cross-session sharing)

    The mount relationship:
    - Host: mount_dir (e.g., /home/user/project)
    - Container: container_workdir (e.g., /workspace)
    - Files edited at mount_dir appear at container_workdir inside the container

    Example:
        Using existing container (cross-session sharing):

        ```python
        async with DockerEnvironment(
            container_id="abc123",
            mount_dir=Path("/home/user/project"),
            container_workdir="/workspace",
            cleanup_on_exit=False,  # Keep container for next session
        ) as env:
            # File operations work on /home/user/project
            await env.file_operator.write_file("test.py", "print('hello')")
            # Shell executes in container at /workspace
            code, stdout, stderr = await env.shell.execute(["python", "test.py"])
        ```

        Creating new container:

        ```python
        async with DockerEnvironment(
            image="python:3.11",
            mount_dir=Path("/home/user/project"),
            container_workdir="/workspace",
            cleanup_on_exit=True,  # Remove container when done
        ) as env:
            ...
        ```
    """

    def __init__(
        self,
        mount_dir: Path,
        container_workdir: str = "/workspace",
        container_id: str | None = None,
        image: str | None = None,
        cleanup_on_exit: bool = True,
        shell_timeout: float = 30.0,
        enable_tmp_dir: bool = True,
        tmp_base_dir: Path | None = None,
        resource_state: ResourceRegistryState | None = None,
        resource_factories: dict[str, ResourceFactory] | None = None,
    ):
        """Initialize DockerEnvironment.

        Args:
            mount_dir: Host directory to mount into container.
                This is where file operations are performed.
            container_workdir: Path inside container where mount_dir is mounted.
                This is where shell commands are executed.
            container_id: Existing container ID to use.
                If provided, the container must already be running.
            image: Docker image to create new container from.
                Required if container_id is not provided.
            cleanup_on_exit: Whether to stop/remove container on exit.
                Set to False for cross-session container sharing.
            shell_timeout: Default timeout for shell commands.
            enable_tmp_dir: Whether to create a session temporary directory.
            tmp_base_dir: Base directory for creating session temporary directory.
            resource_state: Optional state to restore resources from.
                Resources will be restored when entering the context.
            resource_factories: Optional dictionary of resource factories.
                Required for any resources in resource_state.

        Raises:
            ValueError: If neither container_id nor image is provided.
        """
        if container_id is None and image is None:
            raise ValueError("Either container_id or image must be provided")

        super().__init__(
            resource_state=resource_state,
            resource_factories=resource_factories,
        )
        self._mount_dir = mount_dir.resolve()
        self._container_workdir = container_workdir
        self._container_id = container_id
        self._image = image
        self._cleanup_on_exit = cleanup_on_exit
        self._shell_timeout = shell_timeout
        self._enable_tmp_dir = enable_tmp_dir
        self._tmp_base_dir = tmp_base_dir

        # Runtime state
        self._created_container: bool = False
        self._client: docker.DockerClient | None = None
        self._tmp_dir_obj: tempfile.TemporaryDirectory[str] | None = None

    @property
    def client(self) -> docker.DockerClient:
        """Get Docker client with lazy initialization."""
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    @property
    def container_id(self) -> str | None:
        """Return the container ID (available after entering context)."""
        return self._container_id

    @property
    def tmp_dir(self) -> Path | None:
        """Return the session temporary directory path, or None if not enabled."""
        if self._tmp_dir_obj is None:
            return None
        return Path(self._tmp_dir_obj.name)

    async def _setup(self) -> None:
        """Initialize file operator, shell, and container."""
        # Create tmp directory if enabled
        tmp_dir_path: Path | None = None
        if self._enable_tmp_dir:
            self._tmp_dir_obj = tempfile.TemporaryDirectory(
                prefix="pai_agent_docker_",
                dir=str(self._tmp_base_dir) if self._tmp_base_dir else None,
            )
            tmp_dir_path = Path(self._tmp_dir_obj.name)

        # Ensure mount_dir exists
        self._mount_dir.mkdir(parents=True, exist_ok=True)

        # Build allowed paths for file operator
        allowed_paths = [self._mount_dir]
        if tmp_dir_path:
            allowed_paths.append(tmp_dir_path)

        # Create or verify container
        if self._container_id is None:
            # Create new container
            self._container_id = await self._create_container()
            self._created_container = True
        else:
            # Verify existing container is running
            await self._verify_container()

        # Create file operator (local filesystem at mount_dir)
        self._file_operator = LocalFileOperator(
            default_path=self._mount_dir,
            allowed_paths=allowed_paths,
            tmp_dir=tmp_dir_path,
        )

        # Create shell (docker exec)
        self._shell = DockerShell(
            container_id=self._container_id,
            container_workdir=self._container_workdir,
            default_timeout=self._shell_timeout,
        )

    async def _teardown(self) -> None:
        """Clean up container and tmp directory."""
        # Cleanup container if we created it and cleanup_on_exit is True
        if self._cleanup_on_exit and self._container_id is not None:
            await self._stop_container()

        # Cleanup tmp directory
        if self._tmp_dir_obj is not None:
            self._tmp_dir_obj.cleanup()
            self._tmp_dir_obj = None

        self._file_operator = None
        self._shell = None

    async def _create_container(self) -> str:
        """Create and start a new container."""
        if self._image is None:
            raise ValueError("Image must be provided to create a new container")

        image = self._image  # Capture for closure

        def _run_container() -> str:
            try:
                container = self.client.containers.run(
                    image=image,
                    volumes={str(self._mount_dir): {"bind": self._container_workdir, "mode": "rw"}},
                    working_dir=self._container_workdir,
                    detach=True,
                    stdin_open=True,
                    tty=True,
                )
                container_id = container.id
                if container_id is None:
                    raise RuntimeError("Container was created but has no ID")
                return container_id
            except docker.errors.ImageNotFound as e:
                raise RuntimeError(f"Docker image not found: {image}") from e
            except docker.errors.APIError as e:
                raise RuntimeError(f"Failed to start container: {e}") from e

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run_container)

    async def _verify_container(self) -> None:
        """Verify that the existing container is running."""
        container_id = self._container_id
        if container_id is None:
            raise RuntimeError("Container ID is not set")

        def _check_container() -> None:
            try:
                container = self.client.containers.get(container_id)
                container.reload()
                if container.status != "running":
                    raise RuntimeError(f"Container {container_id} is not running (status: {container.status})")
            except docker.errors.NotFound as e:
                raise RuntimeError(f"Container not found: {container_id}") from e
            except docker.errors.APIError as e:
                raise RuntimeError(f"Failed to verify container: {e}") from e

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _check_container)

    async def _stop_container(self) -> None:
        """Stop and remove the container."""
        container_id = self._container_id
        if container_id is None:
            return

        def _stop() -> None:
            try:
                container = self.client.containers.get(container_id)
                container.stop(timeout=10)
                container.remove(force=True)
            except docker.errors.NotFound:
                pass  # Container already gone
            except docker.errors.APIError:
                pass  # Best effort cleanup

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _stop)

    async def get_context_instructions(self) -> str:
        """Return combined context instructions for file operations and shell.

        Raises:
            EnvironmentNotEnteredError: If environment has not been entered yet.
        """
        if not self._file_operator or not self._shell:
            raise EnvironmentNotEnteredError("get_context_instructions")

        file_instructions = await self.file_operator.get_context_instructions()
        shell_instructions = await self.shell.get_context_instructions()

        mount_info = f"""<docker-environment>
  <mount-mapping>
    <host-path>{self._mount_dir}</host-path>
    <container-path>{self._container_workdir}</container-path>
  </mount-mapping>
  <note>File edits modify files at host path, which are visible at container path inside the container.</note>
  <note>Shell commands run inside the container at container path.</note>
</docker-environment>"""

        parts = [mount_info]
        if file_instructions:
            parts.append(file_instructions)
        if shell_instructions:
            parts.append(shell_instructions)

        return "\n\n".join(parts)
