"""Wait and synchronization tools for browser control."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.toolsets.browser_use._session import BrowserSession
from pai_agent_sdk.toolsets.browser_use._tools import get_browser_session
from pai_agent_sdk.toolsets.browser_use.tools._types import WaitResult

logger = get_logger(__name__)


async def wait_for_selector(
    selector: str,
    timeout: int = 30000,
    state: Literal["attached", "visible"] = "visible",
) -> dict[str, Any]:
    """Wait for element matching selector to appear.

    Args:
        selector: CSS selector to wait for
        timeout: Maximum wait time in milliseconds (default: 30000)
        state: Element state to wait for - "attached" (in DOM) or "visible" (default: "visible")

    Returns:
        WaitResult dictionary
    """
    logger.info(f"Waiting for selector: {selector} (timeout: {timeout}ms, state: {state})")
    session = get_browser_session()
    start_time = time.time()

    try:
        await session.cdp_client.send.DOM.enable(session_id=session.page)

        timeout_seconds = timeout / 1000
        poll_interval = 0.1  # 100ms polling

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.warning(f"Timeout waiting for selector: {selector}")
                return WaitResult(
                    status="timeout",
                    wait_type="selector",
                    selector=selector,
                    elapsed_time=elapsed,
                    error_message=f"Timeout after {timeout}ms waiting for selector: {selector}",
                ).model_dump()

            try:
                # Get document
                doc = await session.cdp_client.send.DOM.getDocument(session_id=session.page)
                root_node_id = doc["root"]["nodeId"]

                # Query selector
                result = await session.cdp_client.send.DOM.querySelector(
                    params={
                        "nodeId": root_node_id,
                        "selector": selector,
                    },
                    session_id=session.page,
                )

                node_id = result.get("nodeId")
                if node_id and node_id != 0:
                    # Element is attached to DOM
                    if state == "attached":
                        elapsed = time.time() - start_time
                        logger.info(f"Element found (attached): {selector} after {elapsed:.2f}s")
                        return WaitResult(
                            status="success",
                            wait_type="selector",
                            selector=selector,
                            elapsed_time=elapsed,
                        ).model_dump()

                    # Check if visible
                    if state == "visible":
                        try:
                            # Check if element has box model (is visible)
                            await session.cdp_client.send.DOM.getBoxModel(
                                params={"nodeId": node_id}, session_id=session.page
                            )
                            elapsed = time.time() - start_time
                            logger.info(f"Element found (visible): {selector} after {elapsed:.2f}s")
                            return WaitResult(
                                status="success",
                                wait_type="selector",
                                selector=selector,
                                elapsed_time=elapsed,
                            ).model_dump()
                        except Exception as box_error:
                            # Element exists but not visible yet
                            logger.debug(f"Element not yet visible: {box_error}")

            except Exception:
                logger.debug("Polling error (will retry)")

            # Wait before next poll
            await asyncio.sleep(poll_interval)

    except Exception as e:  # pragma: no cover
        elapsed = time.time() - start_time
        logger.exception(f"Error waiting for selector {selector}")
        return WaitResult(
            status="error",
            wait_type="selector",
            selector=selector,
            elapsed_time=elapsed,
            error_message=str(e),
        ).model_dump()


async def wait_for_navigation(timeout: int = 30000) -> dict[str, Any]:
    """Wait for navigation to complete using CDP events.

    Args:
        timeout: Maximum wait time in milliseconds (default: 30000)

    Returns:
        WaitResult dictionary
    """
    logger.info(f"Waiting for navigation (timeout: {timeout}ms)")
    session = get_browser_session()
    start_time = time.time()

    try:
        await session.cdp_client.send.Page.enable(session_id=session.page)

        timeout_seconds = timeout / 1000
        navigation_occurred = False

        # Event handler for frame navigation
        def on_frame_navigated(event: Any, session_id: str | None) -> None:
            nonlocal navigation_occurred
            frame = event.get("frame", {})
            logger.debug(f"Frame navigated: {frame.get('url')}")
            navigation_occurred = True

        # Register event handler
        session.cdp_client.register.Page.frameNavigated(on_frame_navigated)

        try:
            # Wait for navigation event or timeout
            while True:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    if not navigation_occurred:
                        logger.warning("Timeout waiting for navigation")
                        return WaitResult(
                            status="timeout",
                            wait_type="navigation",
                            elapsed_time=elapsed,
                            error_message=f"Timeout after {timeout}ms waiting for navigation",
                        ).model_dump()
                    break

                if navigation_occurred:
                    logger.debug("Navigation detected via event")
                    break

                # Brief sleep to avoid busy waiting
                await asyncio.sleep(0.05)

            # Wait briefly for page to stabilize after navigation
            await asyncio.sleep(0.2)

            elapsed = time.time() - start_time
            logger.info(f"Navigation completed after {elapsed:.2f}s")
            return WaitResult(
                status="success",
                wait_type="navigation",
                elapsed_time=elapsed,
            ).model_dump()

        finally:
            # Event handlers are cleaned up when session/client is closed
            pass

    except Exception as e:  # pragma: no cover
        elapsed = time.time() - start_time
        logger.exception("Error waiting for navigation")
        return WaitResult(
            status="error",
            wait_type="navigation",
            elapsed_time=elapsed,
            error_message=str(e),
        ).model_dump()


async def _wait_for_document_ready(
    session: BrowserSession,
    target_state: str,
    timeout_seconds: float,
    start_time: float,
    timeout: int,
    state: str,
) -> dict[str, Any]:
    """Helper function to wait for document ready state using CDP events + polling fallback."""
    load_event_fired = False
    dom_content_loaded = False

    # Event handlers
    def on_load_event(event: Any, session_id: str | None) -> None:
        nonlocal load_event_fired
        load_event_fired = True
        logger.debug("Page.loadEventFired event received")

    def on_dom_content_loaded(event: Any, session_id: str | None) -> None:
        nonlocal dom_content_loaded
        dom_content_loaded = True
        logger.debug("Page.domContentEventFired event received")

    # Register appropriate event handlers based on target state
    if target_state == "complete":
        session.cdp_client.register.Page.loadEventFired(on_load_event)
    else:  # interactive
        session.cdp_client.register.Page.domContentEventFired(on_dom_content_loaded)

    try:
        # Check current state first (event may have already fired)
        try:
            result = await session.cdp_client.send.Runtime.evaluate(
                params={
                    "expression": "document.readyState",
                    "returnByValue": True,
                },
                session_id=session.page,
            )
            ready_state = result["result"]["value"]
            if ready_state == target_state or ready_state == "complete":
                elapsed = time.time() - start_time
                logger.info(f"Load state already reached: {state} (readyState: {ready_state})")
                return WaitResult(
                    status="success",
                    wait_type="load_state",
                    elapsed_time=elapsed,
                ).model_dump()
        except Exception:
            logger.debug("Error checking initial ready state")

        # Wait for event or timeout
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.warning(f"Timeout waiting for load state: {state}")
                return WaitResult(
                    status="timeout",
                    wait_type="load_state",
                    elapsed_time=elapsed,
                    error_message=f"Timeout after {timeout}ms waiting for load state: {state}",
                ).model_dump()

            # Check if the desired event has fired
            if (target_state == "complete" and load_event_fired) or (
                target_state == "interactive" and dom_content_loaded
            ):
                elapsed = time.time() - start_time
                logger.info(f"Load state reached via event: {state} after {elapsed:.2f}s")
                return WaitResult(
                    status="success",
                    wait_type="load_state",
                    elapsed_time=elapsed,
                ).model_dump()

            # Brief sleep to avoid busy waiting
            await asyncio.sleep(0.05)

    finally:
        # Event handlers are cleaned up when session/client is closed
        pass


async def _wait_for_network_idle(
    session: BrowserSession,
    timeout_seconds: float,
    start_time: float,
    timeout: int,
) -> dict[str, Any]:
    """Helper function to wait for network idle state.

    Monitors actual Network.* CDP events to detect network activity.
    Considers network idle when 500ms pass with no network requests/responses.
    """
    await session.cdp_client.send.Network.enable(session_id=session.page)

    last_activity_time = time.time()
    idle_timeout = 0.5  # 500ms of no activity

    # Track network activity via CDP events
    def on_network_activity(event: Any, session_id: str | None) -> None:
        """Update last activity time when any network event occurs."""
        nonlocal last_activity_time
        last_activity_time = time.time()
        logger.debug(f"Network activity detected at {last_activity_time}")

    # Register event handlers for network activity
    # These will fire whenever there's network activity
    session.cdp_client.register.Network.requestWillBeSent(on_network_activity)
    session.cdp_client.register.Network.responseReceived(on_network_activity)
    session.cdp_client.register.Network.loadingFinished(on_network_activity)
    session.cdp_client.register.Network.loadingFailed(on_network_activity)

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.warning("Timeout waiting for network idle")
                return WaitResult(
                    status="timeout",
                    wait_type="load_state",
                    elapsed_time=elapsed,
                    error_message=f"Timeout after {timeout}ms waiting for network idle",
                ).model_dump()

            # Check if we've been idle long enough
            idle_duration = time.time() - last_activity_time
            if idle_duration >= idle_timeout:
                elapsed = time.time() - start_time
                logger.info(f"Network idle detected after {elapsed:.2f}s (idle for {idle_duration:.2f}s)")
                return WaitResult(
                    status="success",
                    wait_type="load_state",
                    elapsed_time=elapsed,
                ).model_dump()

            # Sleep briefly before next check
            # Network events will update last_activity_time via registered handlers
            await asyncio.sleep(0.1)
    finally:
        # Note: cdp-use may not provide explicit unregister mechanism
        # Event handlers will be cleaned up when the session/client is closed
        pass


async def wait_for_load_state(
    state: Literal["load", "domcontentloaded", "networkidle"] = "load",
    timeout: int = 30000,
) -> dict[str, Any]:
    """Wait for specific page load state.

    Args:
        state: Load state to wait for:
            - "load": Wait for load event (document fully loaded)
            - "domcontentloaded": Wait for DOMContentLoaded event
            - "networkidle": Wait for network to be idle (no requests for 500ms)
        timeout: Maximum wait time in milliseconds (default: 30000)

    Returns:
        WaitResult dictionary
    """
    logger.info(f"Waiting for load state: {state} (timeout: {timeout}ms)")
    session = get_browser_session()
    start_time = time.time()

    try:
        await session.cdp_client.send.Page.enable(session_id=session.page)
        timeout_seconds = timeout / 1000

        if state in ["load", "domcontentloaded"]:
            target_state = "complete" if state == "load" else "interactive"
            return await _wait_for_document_ready(session, target_state, timeout_seconds, start_time, timeout, state)
        else:  # networkidle
            return await _wait_for_network_idle(session, timeout_seconds, start_time, timeout)

    except Exception as e:  # pragma: no cover
        elapsed = time.time() - start_time
        logger.exception(f"Error waiting for load state {state}")
        return WaitResult(
            status="error",
            wait_type="load_state",
            elapsed_time=elapsed,
            error_message=str(e),
        ).model_dump()
