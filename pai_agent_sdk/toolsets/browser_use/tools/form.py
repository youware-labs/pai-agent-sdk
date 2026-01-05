"""Form operation tools for browser control."""

from __future__ import annotations

from typing import Any

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.toolsets.browser_use._tools import get_browser_session
from pai_agent_sdk.toolsets.browser_use.tools._types import CheckboxResult, FileUploadResult, SelectOptionResult

logger = get_logger(__name__)


async def select_option(
    selector: str,
    value: str | None = None,
    label: str | None = None,
    index: int | None = None,
) -> dict[str, Any]:
    """Select an option from a dropdown/select element.

    Args:
        selector: CSS selector for the select element
        value: Option value to select (mutually exclusive with label and index)
        label: Option label text to select (mutually exclusive with value and index)
        index: Option index to select (mutually exclusive with value and label)

    Returns:
        SelectOptionResult dictionary
    """
    logger.info(f"Selecting option in {selector} (value={value}, label={label}, index={index})")
    session = get_browser_session()

    try:
        # Validate parameters
        provided_params = sum([value is not None, label is not None, index is not None])
        if provided_params == 0:
            return SelectOptionResult(
                status="error",
                selector=selector,
                error_message="Must provide one of: value, label, or index",
            ).model_dump()
        if provided_params > 1:
            return SelectOptionResult(
                status="error",
                selector=selector,
                error_message="Must provide only one of: value, label, or index",
            ).model_dump()

        # Enable DOM
        await session.cdp_client.send.DOM.enable(session_id=session.page)

        # Find select element
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
            logger.warning(f"Select element not found: {selector}")
            return SelectOptionResult(
                status="not_found",
                selector=selector,
                error_message=f"Select element not found: {selector}",
            ).model_dump()

        # Build selection script
        if value is not None:
            script = f"""
                (() => {{
                    const select = document.querySelector({selector!r});
                    if (!select) return {{ success: false, error: 'Element not found' }};
                    select.value = {value!r};
                    select.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    return {{ success: true, value: select.value }};
                }})()
            """
        elif label is not None:
            script = f"""
                (() => {{
                    const select = document.querySelector({selector!r});
                    if (!select) return {{ success: false, error: 'Element not found' }};
                    const option = Array.from(select.options).find(opt => opt.text === {label!r});
                    if (!option) return {{ success: false, error: 'Option not found' }};
                    select.value = option.value;
                    select.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    return {{ success: true, value: select.value, label: option.text }};
                }})()
            """
        else:  # index is not None
            script = f"""
                (() => {{
                    const select = document.querySelector({selector!r});
                    if (!select) return {{ success: false, error: 'Element not found' }};
                    if ({index} < 0 || {index} >= select.options.length) {{
                        return {{ success: false, error: 'Index out of range' }};
                    }}
                    select.selectedIndex = {index};
                    select.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    return {{ success: true, value: select.value, index: {index} }};
                }})()
            """

        # Execute script
        eval_result = await session.cdp_client.send.Runtime.evaluate(
            params={
                "expression": script,
                "returnByValue": True,
            },
            session_id=session.page,
        )

        script_result = eval_result["result"].get("value", {})

        if not script_result.get("success"):
            error_msg = script_result.get("error", "Unknown error")
            logger.warning(f"Failed to select option: {error_msg}")
            return SelectOptionResult(
                status="error",
                selector=selector,
                value=value,
                label=label,
                index=index,
                error_message=error_msg,
            ).model_dump()

        logger.info(f"Successfully selected option in {selector}")
        return SelectOptionResult(
            status="success",
            selector=selector,
            value=script_result.get("value"),
            label=script_result.get("label"),
            index=script_result.get("index"),
        ).model_dump()

    except Exception as e:  # pragma: no cover
        logger.exception(f"Failed to select option in {selector}")
        return SelectOptionResult(
            status="error",
            selector=selector,
            value=value,
            label=label,
            index=index,
            error_message=str(e),
        ).model_dump()


async def check(selector: str) -> dict[str, Any]:
    """Check a checkbox or radio button.

    Args:
        selector: CSS selector for the checkbox/radio element

    Returns:
        CheckboxResult dictionary
    """
    logger.info(f"Checking element: {selector}")
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
            logger.warning(f"Checkbox element not found: {selector}")
            return CheckboxResult(
                status="not_found",
                selector=selector,
                checked=False,
                error_message=f"Element not found: {selector}",
            ).model_dump()

        # Check the element using JavaScript
        script = f"""
            (() => {{
                const element = document.querySelector({selector!r});
                if (!element) return {{ success: false, error: 'Element not found' }};
                if (element.type !== 'checkbox' && element.type !== 'radio') {{
                    return {{ success: false, error: 'Element is not a checkbox or radio button' }};
                }}
                if (!element.checked) {{
                    element.checked = true;
                    element.dispatchEvent(new Event('change', {{ bubbles: true }}));
                }}
                return {{ success: true, checked: element.checked }};
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

        if not script_result.get("success"):
            error_msg = script_result.get("error", "Unknown error")
            logger.warning(f"Failed to check element: {error_msg}")
            return CheckboxResult(
                status="error",
                selector=selector,
                checked=False,
                error_message=error_msg,
            ).model_dump()

        logger.info(f"Successfully checked element: {selector}")
        return CheckboxResult(
            status="success",
            selector=selector,
            checked=script_result.get("checked", True),
        ).model_dump()

    except Exception as e:  # pragma: no cover
        logger.exception(f"Failed to check element {selector}")
        return CheckboxResult(
            status="error",
            selector=selector,
            checked=False,
            error_message=str(e),
        ).model_dump()


async def uncheck(selector: str) -> dict[str, Any]:
    """Uncheck a checkbox.

    Args:
        selector: CSS selector for the checkbox element

    Returns:
        CheckboxResult dictionary
    """
    logger.info(f"Unchecking element: {selector}")
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
            logger.warning(f"Checkbox element not found: {selector}")
            return CheckboxResult(
                status="not_found",
                selector=selector,
                checked=True,
                error_message=f"Element not found: {selector}",
            ).model_dump()

        # Uncheck the element using JavaScript
        script = f"""
            (() => {{
                const element = document.querySelector({selector!r});
                if (!element) return {{ success: false, error: 'Element not found' }};
                if (element.type !== 'checkbox') {{
                    return {{ success: false, error: 'Element is not a checkbox' }};
                }}
                if (element.checked) {{
                    element.checked = false;
                    element.dispatchEvent(new Event('change', {{ bubbles: true }}));
                }}
                return {{ success: true, checked: element.checked }};
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

        if not script_result.get("success"):
            error_msg = script_result.get("error", "Unknown error")
            logger.warning(f"Failed to uncheck element: {error_msg}")
            return CheckboxResult(
                status="error",
                selector=selector,
                checked=True,
                error_message=error_msg,
            ).model_dump()

        logger.info(f"Successfully unchecked element: {selector}")
        return CheckboxResult(
            status="success",
            selector=selector,
            checked=script_result.get("checked", False),
        ).model_dump()

    except Exception as e:  # pragma: no cover
        logger.exception(f"Failed to uncheck element {selector}")
        return CheckboxResult(
            status="error",
            selector=selector,
            checked=True,
            error_message=str(e),
        ).model_dump()


async def upload_file(selector: str, file_paths: list[str]) -> dict[str, Any]:
    """Upload file(s) to a file input element.

    Args:
        selector: CSS selector for the file input element
        file_paths: List of absolute file paths to upload

    Returns:
        FileUploadResult dictionary
    """
    logger.info(f"Uploading {len(file_paths)} file(s) to {selector}")
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
            logger.warning(f"File input element not found: {selector}")
            return FileUploadResult(
                status="not_found",
                selector=selector,
                files=file_paths,
                error_message=f"Element not found: {selector}",
            ).model_dump()

        # Set files using CDP DOM.setFileInputFiles
        await session.cdp_client.send.DOM.setFileInputFiles(
            params={
                "files": file_paths,
                "nodeId": node_id,
            },
            session_id=session.page,
        )

        logger.info(f"Successfully uploaded {len(file_paths)} file(s) to {selector}")
        return FileUploadResult(
            status="success",
            selector=selector,
            files=file_paths,
        ).model_dump()

    except Exception as e:  # pragma: no cover
        logger.exception(f"Failed to upload files to {selector}")
        return FileUploadResult(
            status="error",
            selector=selector,
            files=file_paths,
            error_message=str(e),
        ).model_dump()
