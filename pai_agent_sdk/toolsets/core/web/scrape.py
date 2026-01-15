"""Web scraping tool using Firecrawl with MarkItDown fallback."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Annotated, Any

import anyio.to_thread
from markitdown import MarkItDown
from pydantic import Field
from pydantic_ai import RunContext

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.base import BaseTool
from pai_agent_sdk.toolsets.core.web._http_client import ForbiddenUrlError, verify_url

logger = get_logger(__name__)

CONTENT_TRUNCATE_THRESHOLD = 60000
_PROMPTS_DIR = Path(__file__).parent / "prompts"


@cache
def _load_instruction() -> str:
    return (_PROMPTS_DIR / "scrape.md").read_text()


class ScrapeTool(BaseTool):
    """Web scraping tool that converts websites to Markdown."""

    name = "scrape"
    description = "Convert websites to Markdown format for content analysis."

    def __init__(self) -> None:
        self._md = MarkItDown(enable_plugins=True)

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        return _load_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        url: Annotated[str, Field(description="URL of the web page to scrape. e.g. https://example.com")],
    ) -> dict[str, Any]:
        """Scrape webpage and return content as Markdown."""
        skip_verification = ctx.deps.tool_config.skip_url_verification

        # Verify URL security
        if not skip_verification:
            try:
                verify_url(url)
            except ForbiddenUrlError as e:
                logger.warning(f"URL access forbidden: {url} - {e}")
                return {"success": False, "error": f"URL access forbidden - {e}"}

        cfg = ctx.deps.tool_config

        # Try Firecrawl first if available
        if cfg.firecrawl_api_key:
            try:
                from firecrawl import AsyncFirecrawlApp

                logger.info(f"Scraping webpage with Firecrawl: {url}")
                app = AsyncFirecrawlApp(api_key=cfg.firecrawl_api_key)
                result = await app.scrape(url=url, formats=["markdown"])

                if result.markdown:
                    return self._build_success_response(result.markdown)
                logger.warning(f"Firecrawl returned empty result for {url}, falling back")
            except Exception:
                logger.exception(f"Firecrawl failed for {url}, falling back")

        # Fallback to MarkItDown
        return await self._fallback_scrape(url)

    async def _fallback_scrape(self, url: str) -> dict[str, Any]:
        """Fallback scraping using MarkItDown."""
        try:
            result = await anyio.to_thread.run_sync(self._md.convert, url)
            return self._build_success_response(result.text_content)
        except Exception:
            logger.exception(f"Fallback scrape failed for {url}")
            return {"success": False, "error": "Failed to scrape webpage"}

    def _build_success_response(self, content: str) -> dict[str, Any]:
        """Build success response with optional truncation."""
        tips = "All content is returned."
        truncated = False
        total_length = len(content)

        if len(content) > CONTENT_TRUNCATE_THRESHOLD:
            content = content[:CONTENT_TRUNCATE_THRESHOLD] + "\n\n... (truncated)"
            tips = "Content truncated. Consider using `download` to save the full source."
            truncated = True

        return {
            "success": True,
            "markdown_content": content,
            "truncated": truncated,
            "total_length": total_length,
            "tips": tips,
        }
