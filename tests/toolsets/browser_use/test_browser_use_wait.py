"""Test wait tools."""

from __future__ import annotations

from pai_agent_sdk.toolsets.browser_use._tools import build_tool
from pai_agent_sdk.toolsets.browser_use.tools import execute_javascript, navigate_to_url
from pai_agent_sdk.toolsets.browser_use.tools.wait import (
    wait_for_load_state,
    wait_for_navigation,
    wait_for_selector,
)
from pai_agent_sdk.toolsets.browser_use.toolset import BrowserUseToolset


async def test_wait_for_selector_visible(cdp_url, test_server):
    """Test waiting for a visible element."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Wait for an element that exists
        wait_tool = build_tool(session, wait_for_selector)
        result = await wait_tool.function_schema.call({"selector": "h1", "timeout": 5000, "state": "visible"}, None)

        assert result["status"] == "success"
        assert result["wait_type"] == "selector"
        assert result["selector"] == "h1"
        assert "elapsed_time" in result


async def test_wait_for_selector_timeout(cdp_url, test_server):
    """Test waiting for a non-existent element (timeout)."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Wait for an element that doesn't exist
        wait_tool = build_tool(session, wait_for_selector)
        result = await wait_tool.function_schema.call(
            {"selector": "#non-existent-element", "timeout": 1000, "state": "visible"}, None
        )

        assert result["status"] == "timeout"
        assert result["wait_type"] == "selector"
        assert "error_message" in result


async def test_wait_for_selector_attached(cdp_url, test_server):
    """Test waiting for element in attached state."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Wait for an element in attached state
        wait_tool = build_tool(session, wait_for_selector)
        result = await wait_tool.function_schema.call({"selector": "body", "timeout": 5000, "state": "attached"}, None)

        assert result["status"] == "success"
        assert result["wait_type"] == "selector"


async def test_wait_for_load_state_load(cdp_url, test_server):
    """Test waiting for page load state."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Wait for load state
        wait_tool = build_tool(session, wait_for_load_state)
        result = await wait_tool.function_schema.call({"state": "load", "timeout": 10000}, None)

        assert result["status"] == "success"
        assert result["wait_type"] == "load_state"
        assert "elapsed_time" in result


async def test_wait_for_load_state_domcontentloaded(cdp_url, test_server):
    """Test waiting for DOMContentLoaded state."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Wait for DOMContentLoaded
        wait_tool = build_tool(session, wait_for_load_state)
        result = await wait_tool.function_schema.call({"state": "domcontentloaded", "timeout": 10000}, None)

        assert result["status"] == "success"
        assert result["wait_type"] == "load_state"


async def test_wait_for_navigation_with_link_click(cdp_url, test_server):
    """Test waiting for navigation after triggering navigation."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate to example.com
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Create a link that navigates and click it
        js_tool = build_tool(session, execute_javascript)
        nav_url = f"{test_server}/test_fixtures/navigation/page1.html"
        await js_tool.function_schema.call(
            {
                "script": f"""
            const link = document.createElement('a');
            link.href = '{nav_url}';
            link.id = 'test-link';
            link.textContent = 'Test Link';
            document.body.appendChild(link);
        """
            },
            None,
        )

        # Click the link to trigger navigation (in background)
        await js_tool.function_schema.call({"script": "document.getElementById('test-link').click();"}, None)

        # Wait for navigation
        wait_tool = build_tool(session, wait_for_navigation)
        result = await wait_tool.function_schema.call({"timeout": 10000}, None)

        assert result["status"] in ["success", "timeout"]
        assert result["wait_type"] == "navigation"


async def test_wait_for_load_state_networkidle(cdp_url, test_server):
    """Test waiting for network idle state."""
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate first
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/basic.html"}, None)

        # Wait for network idle - should succeed after 500ms of idle
        wait_tool = build_tool(session, wait_for_load_state)
        result = await wait_tool.function_schema.call({"state": "networkidle", "timeout": 5000}, None)

        # With the bug fix, networkidle should now succeed
        assert result["status"] == "success", (
            f"Expected success but got {result['status']}: {result.get('error_message', '')}"
        )
        assert result["wait_type"] == "load_state"
        assert result["elapsed_time"] >= 0.5, "Should wait at least 500ms for idle detection"


async def test_wait_for_load_state_networkidle_with_delayed_requests(cdp_url, test_server):
    """
    Test network idle with delayed requests after page load.

    This test verifies that the event-driven implementation correctly:
    1. Detects network requests that start AFTER initial page load
    2. Waits for all delayed requests to complete
    3. Only returns after 500ms of true network inactivity

    The test page (delayed_requests.html) schedules fetch requests at:
    - 100ms after load
    - 300ms after load
    - 500ms after load

    Network should be idle only after the last request completes + 500ms idle time.
    Expected minimum wait time: ~1000ms (500ms last request + 500ms idle)
    """
    async with BrowserUseToolset(cdp_url) as toolset:
        session = toolset._browser_session

        # Navigate to page with delayed requests
        nav_tool = build_tool(session, navigate_to_url)
        await nav_tool.function_schema.call({"url": f"{test_server}/test_fixtures/delayed_requests.html"}, None)

        # Wait for network idle - must wait for ALL delayed requests to complete
        wait_tool = build_tool(session, wait_for_load_state)
        result = await wait_tool.function_schema.call({"state": "networkidle", "timeout": 10000}, None)

        # Should succeed after all requests complete + 500ms idle
        assert result["status"] == "success", (
            f"Expected success but got {result['status']}: {result.get('error_message', '')}"
        )
        assert result["wait_type"] == "load_state"

        # The page schedules requests at 100ms, 300ms, and 500ms
        # Network idle requires 500ms of no activity after the last request
        # So minimum time should be around 1000ms (500ms last request + 500ms idle)
        # Use 0.9s threshold to account for timing variance in test environments
        assert result["elapsed_time"] >= 0.9, (
            f"Should wait at least ~900ms for delayed requests + idle detection, "
            f"but only waited {result['elapsed_time']:.3f}s"
        )

        # Verify the page content shows all requests completed
        js_tool = build_tool(session, execute_javascript)
        status_result = await js_tool.function_schema.call(
            {"script": "document.getElementById('status').textContent"}, None
        )

        # The final status should indicate all requests are complete
        status_text = status_result.get("result", "")
        assert status_text and "complete" in status_text.lower(), (
            f"Expected status to show completion, got: {status_text}"
        )
