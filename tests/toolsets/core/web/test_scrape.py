"""Tests for pai_agent_sdk.toolsets.core.web.scrape module."""

from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

from inline_snapshot import snapshot
from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext, ToolConfig
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.toolsets.core.web.scrape import CONTENT_TRUNCATE_THRESHOLD, ScrapeTool


def test_scrape_tool_attributes() -> None:
    """Should have correct name and description."""
    assert ScrapeTool.name == "scrape"
    assert "Markdown" in ScrapeTool.description


async def test_scrape_tool_forbidden_url(tmp_path: Path) -> None:
    """Should return error for forbidden URLs when verification enabled."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(
            AgentContext(
                env=env,
                tool_config=ToolConfig(skip_url_verification=False),
            )
        )
        tool = ScrapeTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, url="http://192.168.1.1/page")
        assert result == snapshot({
            "success": False,
            "error": "URL access forbidden - Access to private IP range is forbidden: 192.168.1.1",
        })


async def test_scrape_tool_fallback_to_markitdown(tmp_path: Path) -> None:
    """Should fallback to MarkItDown when Firecrawl not configured."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(
            AgentContext(
                env=env,
                tool_config=ToolConfig(),  # No firecrawl key
            )
        )
        tool = ScrapeTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        # Mock MarkItDown.convert to avoid actual HTTP request
        mock_result = MagicMock()
        mock_result.text_content = "# Test Page\n\nThis is the content."

        with patch.object(tool._md, "convert", return_value=mock_result):
            result = await tool.call(mock_run_ctx, url="https://example.com")

        assert result["success"] is True
        assert "markdown_content" in result
        assert "tips" in result


async def test_scrape_tool_truncates_long_content(tmp_path: Path) -> None:
    """Should truncate content exceeding threshold."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(
            AgentContext(
                env=env,
                tool_config=ToolConfig(firecrawl_api_key=None),  # Force MarkItDown fallback
            )
        )
        tool = ScrapeTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        # Create long content that exceeds threshold
        long_content = "x" * (CONTENT_TRUNCATE_THRESHOLD + 10000)
        mock_result = MagicMock()
        mock_result.text_content = long_content

        with patch.object(tool._md, "convert", return_value=mock_result):
            result = await tool.call(mock_run_ctx, url="https://example.com")

        assert result["success"] is True
        assert result["truncated"] is True
        # Content is truncated to threshold + suffix "\n\n... (truncated)"
        assert len(result["markdown_content"]) <= CONTENT_TRUNCATE_THRESHOLD + 20
