"""Tool building infrastructure with context-based session injection."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from typing import Any, TypeAlias

from pydantic_ai import Tool
from typing_extensions import ParamSpec

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.toolsets.browser_use._session import BrowserSession

logger = get_logger(__name__)

ToolParams = ParamSpec("ToolParams", default=...)
# Tool functions don't need browser_session parameter
CleanToolFunc: TypeAlias = Callable[ToolParams, Any]

# Use ContextVar to store current browser_session
_browser_session_context: ContextVar[BrowserSession | None] = ContextVar("browser_session", default=None)


def get_browser_session() -> BrowserSession:
    """Get the current browser session from context.

    This function can be called within tool functions to access the browser session.

    Returns:
        Current BrowserSession instance

    Raises:
        RuntimeError: If no browser session is available in current context
    """
    session = _browser_session_context.get()
    if session is None:  # pragma: no cover
        logger.error("Attempted to get browser session, but no session is available in context")
        raise RuntimeError("No browser session available in current context")
    logger.debug(f"Retrieved browser session from context (page: {session.page}, url: {session.current_url})")
    return session


def build_tool(
    browser_session: BrowserSession,
    func: CleanToolFunc,
    max_retries: int = 3,
    prefix: str | None = None,
) -> Tool:
    """Build a tool by injecting browser_session through context variables.

    The original function doesn't need browser_session parameter.
    Tool functions can access it via get_browser_session().

    Args:
        browser_session: BrowserSession instance to inject
        func: Tool function to wrap
        max_retries: Maximum number of retries for this tool (default: 3)
        prefix: Optional prefix to add to the tool name

    Returns:
        Configured Tool instance
    """
    tool_name = func.__name__
    logger.info(f"Building tool: {tool_name} (max_retries: {max_retries}, prefix: {prefix})")

    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger.info(f"Executing tool: {tool_name}")
        logger.debug(f"Tool {tool_name} called with args: {args}, kwargs: {kwargs}")
        # Set current context's browser_session
        token = _browser_session_context.set(browser_session)
        try:
            result = await func(*args, **kwargs)
            logger.info(f"Tool {tool_name} completed successfully")
            logger.debug(f"Tool {tool_name} result type: {type(result).__name__}")
            return result
        except Exception:  # pragma: no cover
            logger.exception(f"Tool {tool_name} execution failed")
            raise
        finally:
            # Restore context
            _browser_session_context.reset(token)

    # Preserve original function's signature
    wrapper.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]

    # Apply prefix to tool name if provided
    tool_name_with_prefix = f"{prefix}_{tool_name}" if prefix else None
    if tool_name_with_prefix:
        logger.debug(f"Applied prefix to tool: {tool_name} -> {tool_name_with_prefix}")

    return Tool(
        function=wrapper,
        max_retries=max_retries,
        name=tool_name_with_prefix,
    )
