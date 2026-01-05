"""Docker-based headless Chrome browser sandbox."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import httpx

try:
    import docker
    import docker.errors
except ImportError as e:
    raise ImportError(
        "docker package is required for DockerBrowserSandbox. Install it with: pip install pai-agent-sdk[docker]"
    ) from e

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.sandbox.browser.base import BrowserSandbox
from pai_agent_sdk.utils import get_available_port, run_in_threadpool

logger = get_logger(__name__)


# Alias for backward compatibility
get_port = get_available_port


class DockerBrowserSandbox(BrowserSandbox):
    """Docker-based headless Chrome browser sandbox.

    Runs headless Chrome in an isolated Docker container with remote debugging enabled.

    Requires the docker extra: pip install pai-agent-sdk[docker]

    Example:
        async with DockerBrowserSandbox() as cdp_url:
            # Use cdp_url to connect to Chrome DevTools Protocol
            print(f"Chrome running at: {cdp_url}")

    Attributes:
        IMAGE: Docker image for headless Chrome.
        CONTAINER_PREFIX: Prefix for container names.
    """

    IMAGE = "zenika/alpine-chrome:latest"
    CONTAINER_PREFIX = "headless-chrome"

    def __init__(
        self,
        port: int | None = None,
        container_name: str | None = None,
        auto_remove: bool = True,
    ) -> None:
        """Initialize Docker browser sandbox.

        Args:
            port: Host port to bind Chrome remote debugging (auto-assigned if None).
            container_name: Custom container name (auto-generated if None).
            auto_remove: Automatically remove container when stopped.
        """
        self._client: docker.DockerClient | None = None
        self._container: Any = None
        self._port = port
        self._container_name = container_name or f"{self.CONTAINER_PREFIX}-{uuid4().hex[:8]}"
        self._auto_remove = auto_remove

    @property
    def client(self) -> docker.DockerClient:
        """Get Docker client with lazy initialization."""
        if self._client is None:
            self._client = docker.from_env()
        return self._client  # type: ignore[return-value]

    async def start_browser(self) -> str:
        """Start the headless Chrome browser in Docker container.

        Returns:
            str: Chrome DevTools Protocol endpoint URL (e.g., http://localhost:9222/json/version).

        Raises:
            RuntimeError: If container fails to start or Chrome fails to initialize.
        """
        # Get available port if not specified
        if self._port is None:
            self._port = get_port()

        logger.info(f"Starting headless Chrome on port {self._port}...")

        # Remove existing container with same name
        await self._remove_existing_container()

        # Start Chrome container
        await self._start_container()

        # Wait for Chrome to be ready
        cdp_url = await self._wait_for_ready()

        logger.info(f"Headless Chrome is running at: {cdp_url}")
        return cdp_url

    async def stop_browser(self) -> None:
        """Stop the headless Chrome browser and remove container."""
        if self._container is None:
            logger.warning("No container to stop")
            return

        container = self._container

        def _stop() -> None:
            try:
                container.stop(timeout=10)
                logger.info(f"Container {self._container_name} stopped")
            except docker.errors.NotFound:
                logger.warning(f"Container {self._container_name} not found")
            except docker.errors.APIError:
                logger.exception("Failed to stop container")

        await run_in_threadpool(_stop)
        self._container = None

    async def _remove_existing_container(self) -> None:
        """Remove existing container with same name if it exists."""

        def _remove() -> None:
            try:
                existing = self.client.containers.get(self._container_name)
                existing.remove(force=True)
                logger.debug(f"Removed existing container: {self._container_name}")
            except docker.errors.NotFound:
                pass
            except docker.errors.APIError as e:
                logger.warning(f"Failed to remove existing container: {e}")

        await run_in_threadpool(_remove)

    async def _start_container(self) -> None:
        """Start Chrome container with remote debugging enabled."""

        def _run() -> Any:
            return self.client.containers.run(
                self.IMAGE,
                command=[
                    "chromium-browser",
                    "--headless",
                    "--remote-debugging-port=9222",
                    "--remote-debugging-address=0.0.0.0",
                    "--no-sandbox",
                ],
                name=self._container_name,
                ports={"9222": self._port},
                detach=True,
                remove=self._auto_remove,
            )

        try:
            self._container = await run_in_threadpool(_run)
            logger.debug(f"Container {self._container_name} started")
        except docker.errors.ImageNotFound as e:
            logger.exception("Docker image not found: %s", self.IMAGE)
            raise RuntimeError(f"Docker image not found: {self.IMAGE}") from e
        except docker.errors.APIError as e:
            logger.exception("Failed to start container")
            raise RuntimeError(f"Failed to start container: {e}") from e

    async def _wait_for_ready(self, timeout: int = 30) -> str:
        """Wait for Chrome to be ready and accepting connections.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            str: Chrome DevTools Protocol endpoint URL.

        Raises:
            RuntimeError: If Chrome fails to start within timeout.
        """
        cdp_url = f"http://localhost:{self._port}/json/version"

        async def _check_ready() -> bool:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(cdp_url, timeout=5)
                    return response.status_code == 200
            except Exception:
                return False

        # Retry until Chrome is ready
        for _ in range(timeout):
            if await _check_ready():
                return cdp_url
            await asyncio.sleep(1)

        # Cleanup on failure
        await self.stop_browser()
        raise RuntimeError(f"Chrome failed to start within {timeout}s")
