"""Element query tools for finding and inspecting elements."""

from __future__ import annotations

import json
from typing import Any

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.toolsets.browser_use._tools import get_browser_session
from pai_agent_sdk.toolsets.browser_use.tools._types import ElementInfo, FindElementsResult

logger = get_logger(__name__)


def _parse_node_attributes(node: dict[str, Any]) -> dict[str, str]:
    """Parse node attributes from CDP response."""
    attrs: dict[str, str] = {}
    if "attributes" in node:
        # Attributes come as [name1, value1, name2, value2, ...]
        attr_list = node["attributes"]
        for i in range(0, len(attr_list), 2):
            attrs[attr_list[i]] = attr_list[i + 1]
    return attrs


async def _get_element_bounding_box(session: Any, node_id: int) -> dict[str, float] | None:
    """Get bounding box for an element node."""
    try:
        box_result = await session.cdp_client.send.DOM.getBoxModel(params={"nodeId": node_id}, session_id=session.page)
        border = box_result["model"]["border"]
        x = min(border[0], border[2], border[4], border[6])
        y = min(border[1], border[3], border[5], border[7])
        width = max(border[0], border[2], border[4], border[6]) - x
        height = max(border[1], border[3], border[5], border[7]) - y
        return {"x": x, "y": y, "width": width, "height": height}
    except Exception:  # pragma: no cover
        logger.debug(f"Failed to get bounding box for node {node_id}")
        return None


async def _extract_element_info(session: Any, node_id: int, selector: str) -> ElementInfo | None:
    """Extract information from a single element node."""
    try:
        # Get node attributes and info
        node_result = await session.cdp_client.send.DOM.describeNode(
            params={"nodeId": node_id}, session_id=session.page
        )
        node = node_result["node"]

        tag_name = node.get("localName", "")
        attrs = _parse_node_attributes(node)

        # Get text content using JavaScript
        text_result = await session.cdp_client.send.Runtime.callFunctionOn(
            params={
                "functionDeclaration": "function() { return this.textContent.trim(); }",
                "objectId": await _get_object_id(session, node_id),
                "returnByValue": True,
            },
            session_id=session.page,
        )
        text = text_result["result"].get("value", "")[:200]

        bounding_box = await _get_element_bounding_box(session, node_id)

        return ElementInfo(
            selector=selector,
            tag_name=tag_name,
            text=text,
            attributes=attrs,
            bounding_box=bounding_box,
        )
    except Exception:  # pragma: no cover
        logger.debug(f"Failed to process element node {node_id}")
        return None


async def find_elements(selector: str, limit: int = 10) -> dict[str, Any]:
    """Find elements matching a CSS selector.

    Args:
        selector: CSS selector to search for
        limit: Maximum number of elements to return (default: 10)

    Returns:
        FindElementsResult dictionary with list of matching elements
    """
    logger.info(f"Finding elements with selector: {selector} (limit: {limit})")
    session = get_browser_session()

    try:
        # Enable DOM
        logger.info("Enabling DOM domain...")
        await session.cdp_client.send.DOM.enable(session_id=session.page)

        # Get document
        logger.info("Getting document root...")
        doc = await session.cdp_client.send.DOM.getDocument(session_id=session.page)
        root_node_id = doc["root"]["nodeId"]

        # Query all matching elements
        logger.info(f"Querying all elements matching: {selector}")
        result = await session.cdp_client.send.DOM.querySelectorAll(
            params={
                "nodeId": root_node_id,
                "selector": selector,
            },
            session_id=session.page,
        )

        node_ids = result.get("nodeIds", [])[:limit]
        logger.info(f"Found {len(result.get('nodeIds', []))} elements, processing first {len(node_ids)}")

        # Extract info from each element
        element_infos: list[ElementInfo] = []
        logger.info("Extracting element information...")
        for node_id in node_ids:
            elem_info = await _extract_element_info(session, node_id, selector)
            if elem_info:
                element_infos.append(elem_info)

        logger.info(f"Successfully extracted information for {len(element_infos)} elements")

        # Debug: Show detailed element information
        if element_infos:
            logger.debug(f"Element details for selector '{selector}':")
            for idx, elem in enumerate(element_infos[:3]):  # Show first 3 elements
                logger.debug(f"  [{idx}] Tag: {elem.tag_name}, Text: {elem.text[:100] if elem.text else 'N/A'}")
                logger.debug(f"      Attributes: {elem.attributes}")
                if elem.bounding_box:
                    logger.debug(
                        f"      Position: x={elem.bounding_box['x']}, y={elem.bounding_box['y']}, w={elem.bounding_box['width']}, h={elem.bounding_box['height']}"
                    )
            if len(element_infos) > 3:
                logger.debug(f"  ... and {len(element_infos) - 3} more elements")

        return FindElementsResult(
            status="success",
            selector=selector,
            count=len(element_infos),
            elements=element_infos,
        ).model_dump()

    except Exception as e:  # pragma: no cover
        logger.exception(f"Failed to find elements for selector {selector}")
        return FindElementsResult(
            status="error",
            selector=selector,
            count=0,
            error_message=str(e),
        ).model_dump()


async def get_element_text(selector: str) -> str:
    """Get text content of an element.

    Args:
        selector: CSS selector for the element

    Returns:
        Text content or empty string if not found
    """
    logger.info(f"Getting text content for element: {selector}")
    session = get_browser_session()

    try:
        result = await session.cdp_client.send.Runtime.evaluate(
            params={
                "expression": f"document.querySelector({json.dumps(selector)})?.textContent?.trim() || ''",
                "returnByValue": True,
            },
            session_id=session.page,
        )

        text = result["result"].get("value", "")
        logger.info(f"Retrieved text content ({len(text)} characters) for {selector}")
        logger.debug(f"Element text for '{selector}': {text[:200]}{'...' if len(text) > 200 else ''}")
        return text

    except Exception:  # pragma: no cover
        logger.exception(f"Failed to get element text for {selector}")
        return ""


async def get_element_attributes(selector: str, attributes: list[str] | None = None) -> dict[str, str]:
    """Get attributes of an element.

    Args:
        selector: CSS selector for the element
        attributes: List of attribute names to retrieve (None = all attributes)

    Returns:
        Dictionary of attribute name-value pairs
    """
    logger.info(f"Getting attributes for element: {selector}")
    session = get_browser_session()

    try:
        if attributes:
            # Get specific attributes
            script = f"""
                (() => {{
                    const el = document.querySelector({json.dumps(selector)});
                    if (!el) return {{}};
                    const result = {{}};
                    {json.dumps(attributes)}.forEach(attr => {{
                        const val = el.getAttribute(attr);
                        if (val !== null) result[attr] = val;
                    }});
                    return result;
                }})()
            """
        else:
            # Get all attributes
            script = f"""
                (() => {{
                    const el = document.querySelector({json.dumps(selector)});
                    if (!el) return {{}};
                    const result = {{}};
                    for (const attr of el.attributes) {{
                        result[attr.name] = attr.value;
                    }}
                    return result;
                }})()
            """

        result = await session.cdp_client.send.Runtime.evaluate(
            params={
                "expression": script,
                "returnByValue": True,
            },
            session_id=session.page,
        )

        attrs = result["result"].get("value", {})
        logger.info(f"Retrieved {len(attrs)} attributes for {selector}")
        logger.debug(f"Element attributes for '{selector}': {attrs}")
        return attrs

    except Exception:  # pragma: no cover
        logger.exception(f"Failed to get element attributes for {selector}")
        return {}


async def _get_object_id(session: Any, node_id: int) -> str:
    """Get JavaScript object ID for a DOM node."""
    result = await session.cdp_client.send.DOM.resolveNode(params={"nodeId": node_id}, session_id=session.page)
    return result["object"]["objectId"]
