"""Navigation tools for browser control."""

from __future__ import annotations

import json
from typing import Any, Literal

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.toolsets.browser_use._tools import get_browser_session
from pai_agent_sdk.toolsets.browser_use.tools._types import NavigationResult
from pai_agent_sdk.toolsets.browser_use.tools.wait import wait_for_load_state

logger = get_logger(__name__)


async def _wait_for_page_ready(
    state: Literal["load", "domcontentloaded"] = "load",
    timeout_ms: int = 30000,
) -> None:
    """Wait for page to reach ready state with timeout control.

    Args:
        state: Load state to wait for ("load" or "domcontentloaded")
        timeout_ms: Timeout in milliseconds

    Raises:
        TimeoutError: If page does not reach ready state within timeout
        RuntimeError: If page load fails
    """
    wait_result = await wait_for_load_state(state, timeout=timeout_ms)

    if wait_result["status"] == "error":  # pragma: no cover
        error_msg = wait_result.get("error_message", "Unknown error")
        logger.error(f"Page load state check failed: {error_msg}")
        raise RuntimeError(f"Page load failed: {error_msg}")

    if wait_result["status"] == "timeout":
        elapsed = wait_result.get("elapsed_time", 0)
        logger.warning(f"Page load state timeout after {elapsed:.2f}s")
        raise TimeoutError(f"Page did not reach '{state}' state within {timeout_ms}ms")

    # status == "success"
    elapsed = wait_result.get("elapsed_time", 0)
    logger.info(f"Page ready (state: {state}) after {elapsed:.2f}s")


async def navigate_to_url(url: str, timeout: int = 30000) -> dict[str, Any]:
    """Navigate to a URL.

    Args:
        url: Target URL to navigate to
        timeout: Navigation timeout in milliseconds (default: 30000)

    Returns:
        NavigationResult dictionary with status, url, and title
    """
    logger.info(f"Starting navigation to URL: {url} (timeout: {timeout}ms)")
    session = get_browser_session()

    try:
        # Enable Page domain
        logger.info("Enabling Page domain...")
        await session.cdp_client.send.Page.enable(session_id=session.page)

        # Navigate via CDP
        logger.info(f"Sending CDP Page.navigate command for: {url}")
        await session.cdp_client.send.Page.navigate(params={"url": url}, session_id=session.page)

        # Wait for page to load with timeout control
        try:
            await _wait_for_page_ready("load", timeout_ms=timeout)
        except TimeoutError as e:
            # Timeout but continue to try getting page info
            logger.warning(f"Navigation timeout: {e}, attempting to get current page info")
        except Exception:  # pragma: no cover
            # Other errors, log but continue
            logger.exception("Error during page load wait")

        # Get page info after navigation
        logger.info("Fetching page information after navigation...")
        result = await session.cdp_client.send.Runtime.evaluate(
            params={
                "expression": """
                    JSON.stringify({
                        url: window.location.href,
                        title: document.title
                    })
                """,
                "returnByValue": True,
            },
            session_id=session.page,
        )

        info = json.loads(result["result"]["value"])
        logger.info(f"Navigation info - URL: {info['url']}, Title: {info['title']}")

        # Update session state
        session.current_url = info["url"]
        session.current_title = info["title"]
        session.navigation_history.append(info["url"])

        return NavigationResult(
            status="success",
            url=info["url"],
            title=info["title"],
        ).model_dump()

    except TimeoutError:  # pragma: no cover
        logger.warning(f"Navigation timeout after {timeout}ms for URL: {url}")
        return NavigationResult(
            status="timeout",
            url=url,
            error_message=f"Navigation timeout after {timeout}ms",
        ).model_dump()

    except Exception as e:  # pragma: no cover
        logger.exception(f"Navigation failed for URL {url}")
        return NavigationResult(
            status="error",
            url=url,
            error_message=str(e),
        ).model_dump()


async def go_back() -> dict[str, Any]:
    """Navigate back in browser history.

    Returns:
        NavigationResult dictionary
    """
    logger.info("Attempting to navigate back in history")
    session = get_browser_session()

    try:
        # Get navigation history
        logger.info("Fetching navigation history...")
        history = await session.cdp_client.send.Page.getNavigationHistory(session_id=session.page)

        current_index = history["currentIndex"]
        logger.info(f"Current history index: {current_index}, total entries: {len(history['entries'])}")

        if current_index > 0:
            # Navigate to previous entry
            entry_id = history["entries"][current_index - 1]["id"]
            logger.info(f"Navigating back to history entry: {entry_id}")
            await session.cdp_client.send.Page.navigateToHistoryEntry(
                params={"entryId": entry_id}, session_id=session.page
            )

            # Wait for page to load (history navigation usually faster)
            try:
                await _wait_for_page_ready("domcontentloaded", timeout_ms=5000)
            except TimeoutError:
                logger.warning("History navigation timeout, but continuing")
            except Exception:  # pragma: no cover
                logger.exception("Error during history navigation wait")

            # Get updated info
            result = await session.cdp_client.send.Runtime.evaluate(
                params={
                    "expression": """
                        JSON.stringify({
                            url: window.location.href,
                            title: document.title
                        })
                    """,
                    "returnByValue": True,
                },
                session_id=session.page,
            )

            info = json.loads(result["result"]["value"])

            session.current_url = info["url"]
            session.current_title = info["title"]
            logger.info(f"Successfully navigated back to: {info['url']}")

            return NavigationResult(
                status="success",
                url=info["url"],
                title=info["title"],
            ).model_dump()
        else:  # pragma: no cover
            logger.warning("Cannot go back - already at the first page in history")
            return NavigationResult(
                status="error",
                url=session.current_url,
                error_message="No previous page in history",
            ).model_dump()

    except Exception as e:  # pragma: no cover
        logger.exception("Failed to navigate back")
        return NavigationResult(
            status="error",
            url=session.current_url,
            error_message=str(e),
        ).model_dump()


async def go_forward() -> dict[str, Any]:
    """Navigate forward in browser history.

    Returns:
        NavigationResult dictionary
    """
    logger.info("Attempting to navigate forward in history")
    session = get_browser_session()

    try:
        # Get navigation history
        logger.info("Fetching navigation history...")
        history = await session.cdp_client.send.Page.getNavigationHistory(session_id=session.page)

        current_index = history["currentIndex"]
        logger.info(f"Current history index: {current_index}, total entries: {len(history['entries'])}")

        if current_index < len(history["entries"]) - 1:
            # Navigate to next entry
            entry_id = history["entries"][current_index + 1]["id"]
            logger.info(f"Navigating forward to history entry: {entry_id}")
            await session.cdp_client.send.Page.navigateToHistoryEntry(
                params={"entryId": entry_id}, session_id=session.page
            )

            # Wait for page to load (history navigation usually faster)
            try:
                await _wait_for_page_ready("domcontentloaded", timeout_ms=5000)
            except TimeoutError:
                logger.warning("History navigation timeout, but continuing")
            except Exception:  # pragma: no cover
                logger.exception("Error during history navigation wait")

            # Get updated info
            result = await session.cdp_client.send.Runtime.evaluate(
                params={
                    "expression": """
                        JSON.stringify({
                            url: window.location.href,
                            title: document.title
                        })
                    """,
                    "returnByValue": True,
                },
                session_id=session.page,
            )

            info = json.loads(result["result"]["value"])

            session.current_url = info["url"]
            session.current_title = info["title"]
            logger.info(f"Successfully navigated forward to: {info['url']}")

            return NavigationResult(
                status="success",
                url=info["url"],
                title=info["title"],
            ).model_dump()
        else:
            logger.warning("Cannot go forward - already at the last page in history")
            return NavigationResult(
                status="error",
                url=session.current_url,
                error_message="No next page in history",
            ).model_dump()

    except Exception as e:  # pragma: no cover
        logger.exception("Failed to navigate forward")
        return NavigationResult(
            status="error",
            url=session.current_url,
            error_message=str(e),
        ).model_dump()


async def reload_page(ignore_cache: bool = False) -> dict[str, Any]:
    """Reload the current page.

    Args:
        ignore_cache: If True, reload ignoring cache

    Returns:
        NavigationResult dictionary
    """
    logger.info(f"Reloading page (ignore_cache: {ignore_cache})")
    session = get_browser_session()

    try:
        # Reload using CDP
        logger.info("Sending CDP Page.reload command...")
        await session.cdp_client.send.Page.reload(params={"ignoreCache": ignore_cache}, session_id=session.page)

        # Wait for reload to complete (longer timeout if ignoring cache)
        timeout_ms = 45000 if ignore_cache else 30000
        try:
            await _wait_for_page_ready("load", timeout_ms=timeout_ms)
        except TimeoutError:
            logger.warning(f"Page reload timeout after {timeout_ms}ms")
        except Exception:  # pragma: no cover
            logger.exception("Error during reload wait")

        # Get updated page info
        result = await session.cdp_client.send.Runtime.evaluate(
            params={
                "expression": """
                    JSON.stringify({
                        url: window.location.href,
                        title: document.title
                    })
                """,
                "returnByValue": True,
            },
            session_id=session.page,
        )

        info = json.loads(result["result"]["value"])

        session.current_url = info["url"]
        session.current_title = info["title"]
        logger.info(f"Page reloaded successfully: {info['url']}")

        return NavigationResult(
            status="success",
            url=info["url"],
            title=info["title"],
        ).model_dump()

    except Exception as e:  # pragma: no cover
        logger.exception("Failed to reload page")
        return NavigationResult(
            status="error",
            url=session.current_url,
            error_message=str(e),
        ).model_dump()
