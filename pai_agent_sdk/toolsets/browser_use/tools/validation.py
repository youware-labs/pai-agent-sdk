"""Element validation and state checking tools."""

from __future__ import annotations

from typing import Any

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.toolsets.browser_use._tools import get_browser_session
from pai_agent_sdk.toolsets.browser_use.tools._types import ValidationResult

logger = get_logger(__name__)


async def is_visible(selector: str) -> dict[str, Any]:
    """Check if an element is visible on the page.

    Args:
        selector: CSS selector for the element

    Returns:
        ValidationResult dictionary with result=True if visible
    """
    logger.info(f"Checking visibility of element: {selector}")
    session = get_browser_session()

    try:
        # Enable DOM
        await session.cdp_client.send.DOM.enable(session_id=session.page)

        # Find element
        doc = await session.cdp_client.send.DOM.getDocument(session_id=session.page)
        root_node_id = doc["root"]["nodeId"]

        result = await session.cdp_client.send.DOM.querySelector(
            params={
                "nodeId": root_node_id,
                "selector": selector,
            },
            session_id=session.page,
        )

        node_id = result.get("nodeId")
        if not node_id or node_id == 0:
            logger.warning(f"Element not found: {selector}")
            return ValidationResult(
                status="not_found",
                selector=selector,
                result=False,
                error_message=f"Element not found: {selector}",
            ).model_dump()

        # Check visibility using multiple methods
        script = f"""
            (() => {{
                const element = document.querySelector({selector!r});
                if (!element) return false;

                // Check if element is hidden
                const style = window.getComputedStyle(element);
                if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {{
                    return false;
                }}

                // Check if element has dimensions
                const rect = element.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) {{
                    return false;
                }}

                // Element is visible
                return true;
            }})()
        """

        eval_result = await session.cdp_client.send.Runtime.evaluate(
            params={
                "expression": script,
                "returnByValue": True,
            },
            session_id=session.page,
        )

        is_visible_result = eval_result["result"].get("value", False)
        logger.info(f"Element {selector} visibility: {is_visible_result}")

        return ValidationResult(
            status="success",
            selector=selector,
            result=is_visible_result,
        ).model_dump()

    except Exception as e:  # pragma: no cover
        logger.exception(f"Failed to check visibility of {selector}")
        return ValidationResult(
            status="error",
            selector=selector,
            result=False,
            error_message=str(e),
        ).model_dump()


async def is_enabled(selector: str) -> dict[str, Any]:
    """Check if an element is enabled (not disabled).

    Args:
        selector: CSS selector for the element

    Returns:
        ValidationResult dictionary with result=True if enabled
    """
    logger.info(f"Checking if element is enabled: {selector}")
    session = get_browser_session()

    try:
        # Enable DOM
        await session.cdp_client.send.DOM.enable(session_id=session.page)

        # Find element
        doc = await session.cdp_client.send.DOM.getDocument(session_id=session.page)
        root_node_id = doc["root"]["nodeId"]

        result = await session.cdp_client.send.DOM.querySelector(
            params={
                "nodeId": root_node_id,
                "selector": selector,
            },
            session_id=session.page,
        )

        node_id = result.get("nodeId")
        if not node_id or node_id == 0:
            logger.warning(f"Element not found: {selector}")
            return ValidationResult(
                status="not_found",
                selector=selector,
                result=False,
                error_message=f"Element not found: {selector}",
            ).model_dump()

        # Check if element is enabled
        script = f"""
            (() => {{
                const element = document.querySelector({selector!r});
                if (!element) return false;

                // Check disabled attribute
                if (element.disabled) return false;

                // Check readonly attribute (for inputs)
                if (element.readOnly) return false;

                // Check if any parent has disabled attribute
                let parent = element.parentElement;
                while (parent) {{
                    if (parent.disabled) return false;
                    parent = parent.parentElement;
                }}

                return true;
            }})()
        """

        eval_result = await session.cdp_client.send.Runtime.evaluate(
            params={
                "expression": script,
                "returnByValue": True,
            },
            session_id=session.page,
        )

        is_enabled_result = eval_result["result"].get("value", False)
        logger.info(f"Element {selector} enabled: {is_enabled_result}")

        return ValidationResult(
            status="success",
            selector=selector,
            result=is_enabled_result,
        ).model_dump()

    except Exception as e:  # pragma: no cover
        logger.exception(f"Failed to check if element is enabled {selector}")
        return ValidationResult(
            status="error",
            selector=selector,
            result=False,
            error_message=str(e),
        ).model_dump()


async def is_checked(selector: str) -> dict[str, Any]:
    """Check if a checkbox or radio button is checked.

    Args:
        selector: CSS selector for the checkbox/radio element

    Returns:
        ValidationResult dictionary with result=True if checked
    """
    logger.info(f"Checking if element is checked: {selector}")
    session = get_browser_session()

    try:
        # Enable DOM
        await session.cdp_client.send.DOM.enable(session_id=session.page)

        # Find element
        doc = await session.cdp_client.send.DOM.getDocument(session_id=session.page)
        root_node_id = doc["root"]["nodeId"]

        result = await session.cdp_client.send.DOM.querySelector(
            params={
                "nodeId": root_node_id,
                "selector": selector,
            },
            session_id=session.page,
        )

        node_id = result.get("nodeId")
        if not node_id or node_id == 0:
            logger.warning(f"Element not found: {selector}")
            return ValidationResult(
                status="not_found",
                selector=selector,
                result=False,
                error_message=f"Element not found: {selector}",
            ).model_dump()

        # Check if element is checked
        script = f"""
            (() => {{
                const element = document.querySelector({selector!r});
                if (!element) return {{ error: 'Element not found' }};
                if (element.type !== 'checkbox' && element.type !== 'radio') {{
                    return {{ error: 'Element is not a checkbox or radio button' }};
                }}
                return {{ checked: element.checked }};
            }})()
        """

        eval_result = await session.cdp_client.send.Runtime.evaluate(
            params={
                "expression": script,
                "returnByValue": True,
            },
            session_id=session.page,
        )

        script_result = eval_result["result"].get("value", {})

        if "error" in script_result:
            logger.warning(f"Element check failed: {script_result['error']}")
            return ValidationResult(
                status="error",
                selector=selector,
                result=False,
                error_message=script_result["error"],
            ).model_dump()

        is_checked_result = script_result.get("checked", False)
        logger.info(f"Element {selector} checked: {is_checked_result}")

        return ValidationResult(
            status="success",
            selector=selector,
            result=is_checked_result,
        ).model_dump()

    except Exception as e:  # pragma: no cover
        logger.exception(f"Failed to check if element is checked {selector}")
        return ValidationResult(
            status="error",
            selector=selector,
            result=False,
            error_message=str(e),
        ).model_dump()
