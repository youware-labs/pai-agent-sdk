"""Tests for pai_agent_sdk.toolsets.core.web.search module."""

from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import MagicMock

import httpx
from inline_snapshot import snapshot
from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext, ToolConfig
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.toolsets.core.web.search import (
    SearchImageTool,
    SearchStockImageTool,
    SearchTool,
)

# =============================================================================
# SearchTool tests
# =============================================================================


def test_search_tool_attributes() -> None:
    """Should have correct name and description."""
    assert SearchTool.name == "search"
    assert "Search the web" in SearchTool.description


async def test_search_tool_is_available_with_google(tmp_path: Path) -> None:
    """Should be available when Google keys are configured."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(
            AgentContext(
                file_operator=env.file_operator,
                shell=env.shell,
                tool_config=ToolConfig(google_search_api_key="test-key", google_search_cx="test-cx"),
            )
        )
        tool = SearchTool(ctx)
        assert tool.is_available() is True


async def test_search_tool_is_available_with_tavily(tmp_path: Path) -> None:
    """Should be available when Tavily key is configured."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(
            AgentContext(
                file_operator=env.file_operator,
                shell=env.shell,
                tool_config=ToolConfig(tavily_api_key="test-key"),
            )
        )
        tool = SearchTool(ctx)
        assert tool.is_available() is True


async def test_search_tool_not_available_without_keys(tmp_path: Path) -> None:
    """Should not be available when no keys configured."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(
            AgentContext(
                file_operator=env.file_operator,
                shell=env.shell,
                # Explicitly pass None for all API keys to override any env defaults
                tool_config=ToolConfig(
                    google_search_api_key=None,
                    google_search_cx=None,
                    tavily_api_key=None,
                ),
            )
        )
        tool = SearchTool(ctx)
        assert tool.is_available() is False


async def test_search_tool_google_search(tmp_path: Path, httpx_mock) -> None:
    """Should use Google when both keys configured."""
    httpx_mock.add_response(
        url="https://www.googleapis.com/customsearch/v1?q=test+query&num=10&key=test-key&cx=test-cx",
        json={
            "items": [
                {"title": "Result 1", "link": "https://example.com/1"},
                {"title": "Result 2", "link": "https://example.com/2"},
            ]
        },
    )

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(
            AgentContext(
                file_operator=env.file_operator,
                shell=env.shell,
                tool_config=ToolConfig(google_search_api_key="test-key", google_search_cx="test-cx"),
            )
        )
        tool = SearchTool(ctx)

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, query="test query")
        assert result == snapshot({
            "items": [
                {"title": "Result 1", "link": "https://example.com/1"},
                {"title": "Result 2", "link": "https://example.com/2"},
            ]
        })


# =============================================================================
# SearchStockImageTool tests
# =============================================================================


def test_search_stock_image_tool_attributes() -> None:
    """Should have correct name and description."""
    assert SearchStockImageTool.name == "search_stock_image"
    assert "Pixabay" in SearchStockImageTool.description


async def test_search_stock_image_tool_is_available(tmp_path: Path) -> None:
    """Should be available when Pixabay key is configured."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(
            AgentContext(
                file_operator=env.file_operator,
                shell=env.shell,
                tool_config=ToolConfig(pixabay_api_key="test-key"),
            )
        )
        tool = SearchStockImageTool(ctx)
        assert tool.is_available() is True


async def test_search_stock_image_tool_search(tmp_path: Path, httpx_mock) -> None:
    """Should search Pixabay and validate URLs."""

    # Mock all requests with a callback
    def mock_callback(request: httpx.Request) -> httpx.Response:
        if "pixabay.com/api" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "total": 1,
                    "hits": [
                        {
                            "id": 123,
                            "tags": "nature, landscape",
                            "webformatURL": "https://pixabay.com/image.jpg",
                            "previewURL": "https://pixabay.com/preview.jpg",
                        }
                    ],
                },
            )
        # HEAD/GET for URL validation
        return httpx.Response(200)

    httpx_mock.add_callback(mock_callback, is_reusable=True)

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(
            AgentContext(
                file_operator=env.file_operator,
                shell=env.shell,
                tool_config=ToolConfig(pixabay_api_key="test-key"),
            )
        )
        tool = SearchStockImageTool(ctx)

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, query="nature")
        assert result["total"] == 1
        assert len(result["hits"]) == 1
        assert "system-reminder" in result


# =============================================================================
# SearchImageTool tests
# =============================================================================


def test_search_image_tool_attributes() -> None:
    """Should have correct name and description."""
    assert SearchImageTool.name == "search_image"
    assert "RapidAPI" in SearchImageTool.description


async def test_search_image_tool_is_available(tmp_path: Path) -> None:
    """Should be available when RapidAPI key is configured."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(
            AgentContext(
                file_operator=env.file_operator,
                shell=env.shell,
                tool_config=ToolConfig(rapidapi_api_key="test-key"),
            )
        )
        tool = SearchImageTool(ctx)
        assert tool.is_available() is True


async def test_search_image_tool_search(tmp_path: Path, httpx_mock) -> None:
    """Should search RapidAPI and validate URLs."""

    # Mock all requests with a callback
    def mock_callback(request: httpx.Request) -> httpx.Response:
        if "rapidapi.com" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "status": "OK",
                    "data": [
                        {"id": 1, "url": "https://example.com/image1.jpg", "title": "Image 1"},
                        {"id": 2, "url": "https://example.com/image2.jpg", "title": "Image 2"},
                    ],
                },
            )
        # HEAD/GET for URL validation
        return httpx.Response(200)

    httpx_mock.add_callback(mock_callback, is_reusable=True)

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(
            AgentContext(
                file_operator=env.file_operator,
                shell=env.shell,
                tool_config=ToolConfig(rapidapi_api_key="test-key"),
            )
        )
        tool = SearchImageTool(ctx)

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, query="test")
        assert result["status"] == "OK"
        assert len(result["data"]) == 2
        assert "system-reminder" in result
