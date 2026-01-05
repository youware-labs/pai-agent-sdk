from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, Self

import httpx
from cdp_use.client import CDPClient
from pydantic_ai import RunContext, Tool
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool
from typing_extensions import TypeVar

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.toolsets.browser_use._config import BrowserUseSettings
from pai_agent_sdk.toolsets.browser_use._session import BrowserSession
from pai_agent_sdk.toolsets.browser_use._tools import build_tool
from pai_agent_sdk.toolsets.browser_use.tools import ALL_TOOLS

logger = get_logger(__name__)

AgentDepsT = TypeVar("AgentDepsT", default=None, contravariant=True)
"""Keep this for custom context types in the future."""


def get_cdp_websocket_url(cdp_url: str) -> str:
    """Resolve CDP WebSocket URL from HTTP endpoint or direct ws:// URL."""
    logger.info(f"Resolving CDP WebSocket URL from: {cdp_url}")

    # If the URL already starts with ws:// or wss://, treat it as a WebSocket URL
    if cdp_url.startswith(("ws://", "wss://")):
        logger.info(f"Using direct WebSocket URL: {cdp_url}")
        return cdp_url

    # Otherwise, treat it as an HTTP endpoint and fetch the WebSocket URL
    logger.info(f"Fetching WebSocket URL from HTTP endpoint: {cdp_url}")
    response = httpx.get(cdp_url)
    response.raise_for_status()
    try:
        data = response.json()
    except ValueError as e:  # pragma: no cover
        logger.exception(f"Failed to parse CDP response as JSON: {response.text}")
        raise ValueError(f"Invalid CDP response. {response.text}") from e
    if "webSocketDebuggerUrl" not in data:  # pragma: no cover
        logger.error(f"CDP response missing webSocketDebuggerUrl field: {data}")
        raise ValueError(f"Invalid CDP response. {data=}")

    websocket_url = data["webSocketDebuggerUrl"]
    logger.info(f"Resolved WebSocket URL: {websocket_url}")
    return websocket_url


@dataclass(kw_only=True)
class BrowserUseTool(ToolsetTool[AgentDepsT]):
    """Tool definition that wraps a browser automation function."""

    call_func: Callable[[dict[str, Any], RunContext[AgentDepsT]], Awaitable[Any]]


class BrowserUseToolset(AbstractToolset, Generic[AgentDepsT]):
    """Pydantic AI toolset for browser automation via Chrome DevTools Protocol."""

    def __init__(
        self,
        cdp_url: str,
        max_retries: int | None = None,
        prefix: str | None = None,
        always_use_new_page: bool | None = None,
        auto_cleanup_page: bool | None = None,
    ) -> None:
        """Initialize the browser toolset.

        Args:
            cdp_url: CDP endpoint URL (HTTP or WebSocket).
            max_retries: Max retry attempts for tool calls. If None, loads from environment or defaults to 3.
            prefix: Tool name prefix. If None, loads from environment or defaults to toolset ID.
            always_use_new_page: Force create new page instead of reusing existing. If None, loads from environment or defaults to False.
            auto_cleanup_page: Automatically close created page targets on context exit. If None, loads from environment or defaults to False.
                Can be combined with always_use_new_page=True to create new pages and automatically clean them up.
        """
        # Load settings from environment variables
        settings = BrowserUseSettings()

        self.cdp_url = cdp_url
        # Use parameter value if provided, otherwise fall back to settings
        self.max_retries = max_retries if max_retries is not None else settings.max_retries
        self.prefix = prefix if prefix is not None else settings.prefix or self.id
        self.always_use_new_page = (
            always_use_new_page if always_use_new_page is not None else settings.always_use_new_page
        )
        self.auto_cleanup_page = auto_cleanup_page if auto_cleanup_page is not None else settings.auto_cleanup_page

        # Internal state initialized during context entry
        self._cdp_client: CDPClient | None = None
        self._browser_session: BrowserSession | None = None
        self._tools: list[Tool[AgentDepsT]] | None = None
        self._created_target_id: str | None = None  # Track created page target for cleanup

    @property
    def id(self) -> str | None:
        """Unique identifier for this toolset instance."""
        return "browser_use"

    async def __aenter__(self) -> Self:
        """Setup CDP connection and attach to browser page.

        Creates new page or reuses existing based on configuration.
        Initializes all browser tools with active session.
        """
        logger.info("Initializing BrowserUseToolset context")

        # Resolve WebSocket URL and establish CDP connection
        websocket_url = get_cdp_websocket_url(self.cdp_url)
        logger.info("Connecting to CDP WebSocket...")
        self._cdp_client = await CDPClient(websocket_url).__aenter__()
        logger.info("CDP client connected successfully")

        # Page targeting strategy: create new or reuse existing
        if self.always_use_new_page:
            # Force create new page target
            logger.info("always_use_new_page is True, creating new page...")
            create_response = await self._cdp_client.send.Target.createTarget(params={"url": "about:blank"})
            target_id = create_response["targetId"]
            self._created_target_id = target_id  # Track for cleanup on exit
            logger.info(f"Created new page target: {target_id}")
        else:
            # Attempt to reuse existing page target
            logger.info("Fetching existing browser targets...")
            targets_response = await self._cdp_client.send.Target.getTargets()
            target_infos = targets_response.get("targetInfos", [])
            logger.info(f"Found {len(target_infos)} existing targets")
            logger.debug(
                f"Target list: {[{'targetId': t.get('targetId'), 'type': t.get('type'), 'url': t.get('url', 'N/A')[:50]} for t in target_infos]}"
            )

            # Search for existing page type target
            page_target = None
            for target_info in target_infos:
                if target_info.get("type") == "page":
                    page_target = target_info
                    break

            if page_target:
                # Reuse found page target
                target_id = page_target["targetId"]
                logger.info(f"Reusing existing page target: {target_id}")
            else:
                # No page target available, create new one
                logger.info("No existing page target found, creating new page...")
                create_response = await self._cdp_client.send.Target.createTarget(params={"url": "about:blank"})
                target_id = create_response["targetId"]
                self._created_target_id = target_id  # Track for cleanup on exit
                logger.info(f"Created new page target: {target_id}")

        # Attach to target to obtain CDP session ID
        logger.info(f"Attaching to target {target_id}...")
        attach_response = await self._cdp_client.send.Target.attachToTarget(
            params={"targetId": target_id, "flatten": True}
        )
        session_id = attach_response["sessionId"]

        if session_id is None:  # pragma: no cover
            logger.error("Failed to obtain session ID from target attachment")
            raise ValueError("Failed to get session ID from target attachment")

        logger.info(f"Attached to target, session_id: {session_id}")

        # Create browser session with CDP session ID as page reference
        self._browser_session = BrowserSession(
            cdp_client=self._cdp_client,
            page=session_id,  # CDP session ID stored as page identifier
        )
        logger.info("BrowserSession created successfully")
        logger.debug(f"Session details - page: {session_id}, viewport: {self._browser_session.viewport}")

        # Build all tools with active session context
        logger.info(f"Building {len(ALL_TOOLS)} browser tools...")
        self._tools = [
            build_tool(
                self._browser_session,
                tool,
                max_retries=self.max_retries,
                prefix=self.prefix,
            )
            for tool in ALL_TOOLS
        ]
        logger.info("All tools built and ready")
        logger.debug(f"Tool names: {[tool.tool_def.name for tool in self._tools]}")

        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        """Cleanup CDP connection and close created page targets."""
        logger.info("Cleaning up BrowserUseToolset context")

        # Close page target if we created one (not reused) and auto_cleanup_page is enabled
        if self._created_target_id and self._cdp_client and self.auto_cleanup_page:
            try:
                logger.info(f"Closing created page target: {self._created_target_id}")
                await self._cdp_client.send.Target.closeTarget(params={"targetId": self._created_target_id})
                logger.info(f"Successfully closed page target: {self._created_target_id}")
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed to close page target {self._created_target_id}: {e}")
            finally:
                self._created_target_id = None
        elif self._created_target_id and not self.auto_cleanup_page:
            logger.info(f"Skipping cleanup for created page target {self._created_target_id} (auto_cleanup_page=False)")

        # Close CDP client connection
        if self._cdp_client:
            logger.info("Closing CDP client connection...")
            await self._cdp_client.__aexit__(*args)
            self._cdp_client = None
            logger.info("CDP client connection closed")

        # Dispose browser session and clear tools
        if self._browser_session:
            logger.info("Disposing browser session...")
            logger.debug(
                f"Session state before disposal - URL: {self._browser_session.current_url}, History: {len(self._browser_session.navigation_history)} entries"
            )
            self._browser_session.dispose()
            logger.info("Browser session disposed")
        self._tools = None
        return None

    async def get_tools(  # type: ignore[override]
        self, ctx: RunContext[AgentDepsT]
    ) -> dict[str, BrowserUseTool[AgentDepsT]]:
        """Return all available browser automation tools."""
        if self._tools is None:
            return {}
        return {
            tool.name: BrowserUseTool(
                toolset=self,
                tool_def=tool.tool_def,
                max_retries=tool.max_retries or 3,
                args_validator=tool.function_schema.validator,
                call_func=tool.function_schema.call,
            )
            for tool in self._tools
        }

    async def call_tool(  # type: ignore[override]
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: BrowserUseTool[AgentDepsT],
    ) -> Any:
        """Execute browser tool with provided arguments.

        Args:
            name: Tool name.
            tool_args: Tool arguments.
            ctx: Run context.
            tool: Tool definition from get_tools().
        """
        return await tool.call_func(tool_args, ctx)
