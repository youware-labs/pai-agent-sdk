"""Optional browser sandbox integration for paintress-cli.

This module provides browser sandbox management with graceful degradation.
If browser startup fails, the application continues without browser capabilities.

Example:
    from paintress_cli.browser import BrowserManager
    from paintress_cli.config import BrowserConfig

    config = BrowserConfig(cdp_url="auto")

    async with BrowserManager(config) as manager:
        if manager.cdp_url:
            # Browser available
            toolset = manager.get_browser_toolset()
        else:
            # Browser not available, continue without it
            pass
"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING

from paintress_cli.logging import get_logger

if TYPE_CHECKING:
    from pai_agent_sdk.toolsets.browser_use import BrowserUseToolset
    from paintress_cli.config import BrowserConfig

logger = get_logger(__name__)


class BrowserManager:
    """Manages optional browser sandbox lifecycle.

    Handles three modes based on BrowserConfig.cdp_url:
    - None: Browser disabled, no action
    - "auto": Start DockerBrowserSandbox, fallback to disabled on failure
    - URL string: Use existing browser at specified CDP URL

    All failures are non-fatal - the manager continues with cdp_url=None.

    Attributes:
        cdp_url: Active CDP URL, or None if browser unavailable.
    """

    def __init__(self, config: BrowserConfig) -> None:
        """Initialize browser manager.

        Args:
            config: Browser configuration.
        """
        self._config = config
        self._sandbox: object | None = None  # DockerBrowserSandbox if started
        self._cdp_url: str | None = None
        self._toolset: BrowserUseToolset | None = None

    @property
    def cdp_url(self) -> str | None:
        """Get active CDP URL, or None if browser unavailable."""
        return self._cdp_url

    @property
    def is_available(self) -> bool:
        """Check if browser is available."""
        return self._cdp_url is not None

    async def start(self) -> str | None:
        """Start browser if configured.

        Returns:
            CDP URL if browser started successfully, None otherwise.
        """
        if not self._config.cdp_url:
            logger.debug("Browser disabled (cdp_url not configured)")
            return None

        if self._config.cdp_url == "auto":
            return await self._start_docker_browser()
        else:
            # Use provided CDP URL directly
            self._cdp_url = self._config.cdp_url
            logger.info("Using external browser at: %s", self._cdp_url)
            return self._cdp_url

    async def stop(self) -> None:
        """Stop browser sandbox if we started it."""
        if self._sandbox is not None:
            try:
                await self._sandbox.stop_browser()  # type: ignore[union-attr]
                logger.info("Browser sandbox stopped")
            except Exception:
                logger.exception("Failed to stop browser sandbox")
            finally:
                self._sandbox = None
                self._cdp_url = None
                self._toolset = None

    async def _start_docker_browser(self) -> str | None:
        """Start Docker browser sandbox with graceful fallback.

        Returns:
            CDP URL if successful, None on failure.
        """
        try:
            from pai_agent_sdk.sandbox.browser.docker_ import DockerBrowserSandbox
        except ImportError:
            logger.warning("Docker browser sandbox not available. Install with: pip install pai-agent-sdk[docker]")
            return None

        try:
            logger.info("Starting Docker browser sandbox...")
            sandbox = DockerBrowserSandbox(
                image=self._config.browser_image,
            )
            cdp_url = await sandbox.start_browser()
            self._sandbox = sandbox
            self._cdp_url = cdp_url
            logger.info("Browser sandbox started at: %s", cdp_url)
            return cdp_url

        except Exception:
            logger.warning(
                "Failed to start Docker browser sandbox. Continuing without browser capabilities.",
                exc_info=True,
            )
            return None

    def get_browser_toolset(self) -> BrowserUseToolset | None:
        """Get BrowserUseToolset if browser is available.

        Returns:
            BrowserUseToolset instance, or None if browser unavailable.
        """
        if not self._cdp_url:
            return None

        if self._toolset is None:
            try:
                from pai_agent_sdk.toolsets.browser_use import BrowserUseToolset

                self._toolset = BrowserUseToolset(cdp_url=self._cdp_url)
                logger.debug("Created BrowserUseToolset")
            except ImportError:
                logger.warning("BrowserUseToolset not available")
                return None

        return self._toolset

    async def __aenter__(self) -> BrowserManager:
        """Enter async context manager."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context manager."""
        await self.stop()
