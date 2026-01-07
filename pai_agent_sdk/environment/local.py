"""Local environment implementations.

This module provides local file system and shell implementations
using standard library functions.
"""

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import anyio

from pai_agent_sdk.environment.base import (
    Environment,
    FileOperator,
    Shell,
    TmpFileOperator,
)
from pai_agent_sdk.environment.exceptions import (
    FileOperationError,
    PathNotAllowedError,
    ShellExecutionError,
    ShellTimeoutError,
)

if TYPE_CHECKING:
    pass


class LocalFileOperator(FileOperator):
    """Local file system operator with path validation.

    Implements the FileOperator ABC for local file system access.
    Validates all paths against a list of allowed directories.

    This class is unaware of tmp_dir handling - it simply implements
    the _xxx_impl methods. The base class FileOperator handles tmp
    routing transparently.
    """

    def __init__(
        self,
        allowed_paths: list[Path] | None = None,
        default_path: Path | None = None,
        instructions_skip_dirs: frozenset[str] | None = None,
        instructions_max_depth: int = 3,
        tmp_dir: Path | None = None,
        tmp_file_operator: TmpFileOperator | None = None,
    ):
        super().__init__(
            allowed_paths=allowed_paths,
            default_path=default_path,
            instructions_skip_dirs=instructions_skip_dirs,
            instructions_max_depth=instructions_max_depth,
            tmp_dir=tmp_dir,
            tmp_file_operator=tmp_file_operator,
        )

    def _resolve_path(self, path: str) -> Path:
        """Resolve path and validate against allowed directories."""
        target = Path(path)
        if not target.is_absolute():
            target = self._default_path / target
        resolved = target.resolve()
        if not self._is_path_allowed(resolved):
            raise PathNotAllowedError(
                path,
                [str(p) for p in self._allowed_paths],
            )
        return resolved

    def _is_path_allowed(self, resolved: Path) -> bool:
        """Check if resolved path is within allowed directories."""
        for allowed in self._allowed_paths:
            try:
                resolved.relative_to(allowed)
                return True
            except ValueError:
                continue
        return False

    async def _read_file_impl(
        self,
        path: str,
        *,
        encoding: str = "utf-8",
        offset: int = 0,
        length: int | None = None,
    ) -> str:
        """Read file content as string.

        Args:
            path: File path.
            encoding: Text encoding (default: utf-8).
            offset: Character offset to start reading from (default: 0).
            length: Maximum number of characters to read (default: None = read all).

        Returns:
            File content as string (or substring if offset/length specified).
        """
        resolved = self._resolve_path(path)
        try:
            content = await anyio.Path(resolved).read_text(encoding=encoding)
            if offset > 0 or length is not None:
                end = None if length is None else offset + length
                content = content[offset:end]
            return content
        except FileNotFoundError as e:
            raise FileOperationError("read", path, "file not found") from e
        except PermissionError as e:
            raise FileOperationError("read", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("read", path, str(e)) from e

    async def _read_bytes_impl(
        self,
        path: str,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes:
        """Read file content as bytes.

        Args:
            path: File path.
            offset: Byte offset to start reading from (default: 0).
            length: Maximum number of bytes to read (default: None = read all).

        Returns:
            File content as bytes (or slice if offset/length specified).
        """
        resolved = self._resolve_path(path)
        try:
            content = await anyio.Path(resolved).read_bytes()
            if offset > 0 or length is not None:
                end = None if length is None else offset + length
                content = content[offset:end]
            return content
        except FileNotFoundError as e:
            raise FileOperationError("read", path, "file not found") from e
        except PermissionError as e:
            raise FileOperationError("read", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("read", path, str(e)) from e

    async def _write_file_impl(
        self,
        path: str,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> None:
        """Write content to file."""
        resolved = self._resolve_path(path)
        try:
            apath = anyio.Path(resolved)
            if isinstance(content, bytes):
                await apath.write_bytes(content)
            else:
                await apath.write_text(content, encoding=encoding)
        except PermissionError as e:
            raise FileOperationError("write", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("write", path, str(e)) from e

    async def _append_file_impl(
        self,
        path: str,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> None:
        """Append content to file."""
        resolved = self._resolve_path(path)
        try:
            # anyio.Path doesn't support append mode, use sync in thread
            def _append():
                mode = "ab" if isinstance(content, bytes) else "a"
                with open(resolved, mode, encoding=None if isinstance(content, bytes) else encoding) as f:
                    f.write(content)

            await anyio.to_thread.run_sync(_append)  # type: ignore[reportAttributeAccessIssue]
        except PermissionError as e:
            raise FileOperationError("append", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("append", path, str(e)) from e

    async def _delete_impl(self, path: str) -> None:
        """Delete file or empty directory."""
        resolved = self._resolve_path(path)
        try:
            apath = anyio.Path(resolved)
            if await apath.is_dir():
                await apath.rmdir()
            else:
                await apath.unlink()
        except FileNotFoundError as e:
            raise FileOperationError("delete", path, "file not found") from e
        except PermissionError as e:
            raise FileOperationError("delete", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("delete", path, str(e)) from e

    async def _list_dir_impl(self, path: str) -> list[str]:
        """List directory contents."""
        resolved = self._resolve_path(path)
        try:
            apath = anyio.Path(resolved)
            entries = []
            async for entry in apath.iterdir():
                entries.append(entry.name)
            return sorted(entries)
        except FileNotFoundError as e:
            raise FileOperationError("list", path, "directory not found") from e
        except NotADirectoryError as e:
            raise FileOperationError("list", path, "not a directory") from e
        except PermissionError as e:
            raise FileOperationError("list", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("list", path, str(e)) from e

    async def _exists_impl(self, path: str) -> bool:
        """Check if path exists."""
        resolved = self._resolve_path(path)
        return await anyio.Path(resolved).exists()

    async def _is_file_impl(self, path: str) -> bool:
        """Check if path is a file."""
        resolved = self._resolve_path(path)
        return await anyio.Path(resolved).is_file()

    async def _is_dir_impl(self, path: str) -> bool:
        """Check if path is a directory."""
        resolved = self._resolve_path(path)
        return await anyio.Path(resolved).is_dir()

    async def _mkdir_impl(self, path: str, *, parents: bool = False) -> None:
        """Create directory."""
        resolved = self._resolve_path(path)
        try:
            await anyio.Path(resolved).mkdir(parents=parents, exist_ok=True)
        except PermissionError as e:
            raise FileOperationError("mkdir", path, "permission denied") from e
        except OSError as e:
            raise FileOperationError("mkdir", path, str(e)) from e

    async def _move_impl(self, src: str, dst: str) -> None:
        """Move file or directory."""
        src_resolved = self._resolve_path(src)
        dst_resolved = self._resolve_path(dst)
        try:
            await anyio.to_thread.run_sync(lambda: shutil.move(src_resolved, dst_resolved))  # type: ignore[reportAttributeAccessIssue]
        except FileNotFoundError as e:
            raise FileOperationError("move", src, "source not found") from e
        except PermissionError as e:
            raise FileOperationError("move", src, "permission denied") from e
        except OSError as e:
            raise FileOperationError("move", src, str(e)) from e

    async def _copy_impl(self, src: str, dst: str) -> None:
        """Copy file or directory."""
        src_resolved = self._resolve_path(src)
        dst_resolved = self._resolve_path(dst)
        try:
            if src_resolved.is_dir():
                await anyio.to_thread.run_sync(lambda: shutil.copytree(src_resolved, dst_resolved))  # type: ignore[reportAttributeAccessIssue]
            else:
                await anyio.to_thread.run_sync(lambda: shutil.copy2(src_resolved, dst_resolved))  # type: ignore[reportAttributeAccessIssue]
        except FileNotFoundError as e:
            raise FileOperationError("copy", src, "source not found") from e
        except PermissionError as e:
            raise FileOperationError("copy", src, "permission denied") from e
        except OSError as e:
            raise FileOperationError("copy", src, str(e)) from e


class LocalShell(Shell):
    """Local shell command executor with path validation.

    Implements the Shell ABC for local command execution.
    Validates working directory against allowed paths.
    """

    def __init__(
        self,
        allowed_paths: list[Path] | None = None,
        default_cwd: Path | None = None,
        default_timeout: float = 30.0,
    ):
        """Initialize LocalShell."""
        super().__init__(
            allowed_paths=allowed_paths,
            default_cwd=default_cwd,
            default_timeout=default_timeout,
        )

    def _resolve_cwd(self, cwd: str | None) -> Path:
        """Resolve and validate working directory."""
        if cwd is None:
            return self._default_cwd

        target = Path(cwd)
        if not target.is_absolute():
            target = self._default_cwd / target
        resolved = target.resolve()

        if not self._is_path_allowed(resolved):
            raise PathNotAllowedError(
                cwd,
                [str(p) for p in self._allowed_paths],
            )
        return resolved

    def _is_path_allowed(self, resolved: Path) -> bool:
        """Check if resolved path is within allowed directories."""
        for allowed in self._allowed_paths:
            try:
                resolved.relative_to(allowed)
                return True
            except ValueError:
                continue
        return False

    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> tuple[int, str, str]:
        """Execute a command and return results.

        Args:
            command: Command string to execute via shell.
            timeout: Timeout in seconds (uses default if None).
            env: Environment variables.
            cwd: Working directory (relative or absolute path).

        Returns:
            Tuple of (exit_code, stdout, stderr).
        """
        if not command:
            raise ShellExecutionError("", stderr="Empty command")

        resolved_cwd = self._resolve_cwd(cwd)
        effective_timeout = timeout if timeout is not None else self._default_timeout

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=resolved_cwd,
                env=env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout,
                )
            except TimeoutError as e:
                # Try graceful termination first
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except TimeoutError:
                    # Force kill if graceful termination fails
                    process.kill()
                    await process.wait()
                raise ShellTimeoutError(command, effective_timeout) from e

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            return (process.returncode or 0, stdout, stderr)

        except FileNotFoundError as e:
            raise ShellExecutionError(
                command,
                stderr="Command not found",
            ) from e
        except PermissionError as e:
            raise ShellExecutionError(
                command,
                stderr="Permission denied",
            ) from e
        except OSError as e:
            raise ShellExecutionError(command, stderr=str(e)) from e


class LocalEnvironment(Environment):
    """Local environment with filesystem and shell access.

    Creates LocalFileOperator and LocalShell with shared configuration,
    and manages temporary directory lifecycle.

    Example:
        Using AsyncExitStack (recommended for dependent contexts):

        ```python
        from contextlib import AsyncExitStack

        async with AsyncExitStack() as stack:
            env = await stack.enter_async_context(
                LocalEnvironment(allowed_paths=[Path("/workspace")])
            )
            ctx = await stack.enter_async_context(
                AgentContext(
                    file_operator=env.file_operator,
                    shell=env.shell,
                    resources=env.resources,
                )
            )
            await ctx.file_operator.read_file("test.txt")
        # Resources cleaned up when stack exits
        ```
    """

    def __init__(
        self,
        allowed_paths: list[Path] | None = None,
        default_path: Path | None = None,
        shell_timeout: float = 30.0,
        tmp_base_dir: Path | None = None,
        enable_tmp_dir: bool = True,
    ):
        """Initialize LocalEnvironment.

        Args:
            allowed_paths: Directories accessible by both file and shell operations.
            default_path: Default working directory for operations.
            shell_timeout: Default shell command timeout.
            tmp_base_dir: Base directory for creating session temporary directory.
                If None, uses system default temp directory.
            enable_tmp_dir: Whether to create a session temporary directory.
                Defaults to True.
        """
        super().__init__()  # Initialize ResourceRegistry
        self._allowed_paths = allowed_paths
        self._default_path = default_path
        self._shell_timeout = shell_timeout
        self._tmp_base_dir = tmp_base_dir
        self._enable_tmp_dir = enable_tmp_dir
        self._tmp_dir_obj: tempfile.TemporaryDirectory[str] | None = None

    @property
    def tmp_dir(self) -> Path | None:
        """Return the session temporary directory path, or None if not enabled."""
        if self._tmp_dir_obj is None:
            return None
        return Path(self._tmp_dir_obj.name)

    async def _setup(self) -> None:
        """Initialize file operator, shell, and tmp directory."""
        tmp_dir_path: Path | None = None
        if self._enable_tmp_dir:
            self._tmp_dir_obj = tempfile.TemporaryDirectory(
                prefix="pai_agent_",
                dir=str(self._tmp_base_dir) if self._tmp_base_dir else None,
            )
            tmp_dir_path = Path(self._tmp_dir_obj.name)

        allowed = list(self._allowed_paths) if self._allowed_paths else []
        if tmp_dir_path:
            allowed.append(tmp_dir_path)

        self._file_operator = LocalFileOperator(
            allowed_paths=allowed or None,
            default_path=self._default_path,
            tmp_dir=tmp_dir_path,
        )

        self._shell = LocalShell(
            allowed_paths=allowed or None,
            default_cwd=self._default_path,
            default_timeout=self._shell_timeout,
        )

    async def _teardown(self) -> None:
        """Clean up tmp directory and reset operators."""
        if self._tmp_dir_obj is not None:
            self._tmp_dir_obj.cleanup()
            self._tmp_dir_obj = None

        self._file_operator = None
        self._shell = None
