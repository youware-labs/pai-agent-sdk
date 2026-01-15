"""Tests for paintress_cli.browser module."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from paintress_cli.browser import BrowserManager
from paintress_cli.config import BrowserConfig

# =============================================================================
# BrowserManager Initialization Tests
# =============================================================================


def test_browser_manager_init() -> None:
    """Test BrowserManager initialization."""
    config = BrowserConfig(cdp_url=None)
    manager = BrowserManager(config)

    assert manager.cdp_url is None
    assert manager.is_available is False


def test_browser_manager_init_with_url() -> None:
    """Test BrowserManager with explicit CDP URL."""
    config = BrowserConfig(cdp_url="ws://localhost:9222")
    manager = BrowserManager(config)

    # URL not set until start() is called
    assert manager.cdp_url is None


# =============================================================================
# BrowserManager.start() Tests
# =============================================================================


@pytest.mark.asyncio
async def test_browser_manager_start_disabled() -> None:
    """Test start() when browser is disabled."""
    config = BrowserConfig(cdp_url=None)
    manager = BrowserManager(config)

    result = await manager.start()

    assert result is None
    assert manager.cdp_url is None
    assert manager.is_available is False


@pytest.mark.asyncio
async def test_browser_manager_start_external_url() -> None:
    """Test start() with external CDP URL."""
    config = BrowserConfig(cdp_url="ws://localhost:9222")
    manager = BrowserManager(config)

    result = await manager.start()

    assert result == "ws://localhost:9222"
    assert manager.cdp_url == "ws://localhost:9222"
    assert manager.is_available is True


@pytest.mark.asyncio
async def test_browser_manager_start_auto_no_docker() -> None:
    """Test start() with auto mode when docker not available."""
    config = BrowserConfig(cdp_url="auto")
    manager = BrowserManager(config)

    # Mock ImportError for docker module
    with patch.dict("sys.modules", {"pai_agent_sdk.sandbox.browser.docker_": None}):
        with patch(
            "paintress_cli.browser.BrowserManager._start_docker_browser",
            return_value=None,
        ):
            result = await manager.start()

    assert result is None
    assert manager.is_available is False


# =============================================================================
# BrowserManager.stop() Tests
# =============================================================================


@pytest.mark.asyncio
async def test_browser_manager_stop_no_sandbox() -> None:
    """Test stop() when no sandbox was started."""
    config = BrowserConfig(cdp_url=None)
    manager = BrowserManager(config)

    # Should not raise
    await manager.stop()


# =============================================================================
# Context Manager Tests
# =============================================================================


@pytest.mark.asyncio
async def test_browser_manager_context_manager_disabled() -> None:
    """Test BrowserManager as context manager when disabled."""
    config = BrowserConfig(cdp_url=None)

    async with BrowserManager(config) as manager:
        assert manager.cdp_url is None
        assert manager.is_available is False


@pytest.mark.asyncio
async def test_browser_manager_context_manager_external() -> None:
    """Test BrowserManager as context manager with external URL."""
    config = BrowserConfig(cdp_url="ws://localhost:9222")

    async with BrowserManager(config) as manager:
        assert manager.cdp_url == "ws://localhost:9222"
        assert manager.is_available is True

    # After exit, cdp_url should still be set (we didn't start sandbox)
    assert manager.cdp_url == "ws://localhost:9222"


# =============================================================================
# get_browser_toolset Tests
# =============================================================================


def test_get_browser_toolset_not_available() -> None:
    """Test get_browser_toolset when browser not available."""
    config = BrowserConfig(cdp_url=None)
    manager = BrowserManager(config)

    toolset = manager.get_browser_toolset()

    assert toolset is None
