"""Tests for pai_agent_sdk.toolsets.core.web.download module."""

from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import MagicMock

from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.toolsets.core.web.download import DownloadTool


def test_download_tool_attributes() -> None:
    """Should have correct name and description."""
    assert DownloadTool.name == "download"
    assert "Download" in DownloadTool.description


async def test_download_tool_single_file(tmp_path: Path, httpx_mock) -> None:
    """Should download a single file."""
    httpx_mock.add_response(
        url="https://example.com/file.txt",
        content=b"Hello, World!",
        headers={"Content-Type": "text/plain"},
    )

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = DownloadTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(
            mock_run_ctx,
            urls=["https://example.com/file.txt"],
            save_dir="downloads",
        )

        assert len(result) == 1
        assert result[0]["success"] is True
        assert result[0]["content_type"] == "text/plain"
        assert result[0]["size"] == 13

        # Verify file was created
        save_path = tmp_path / result[0]["save_path"]
        assert save_path.exists()
        assert save_path.read_bytes() == b"Hello, World!"


async def test_download_tool_multiple_files(tmp_path: Path, httpx_mock) -> None:
    """Should download multiple files in parallel."""
    httpx_mock.add_response(
        url="https://example.com/file1.txt",
        content=b"File 1",
        headers={"Content-Type": "text/plain"},
    )
    httpx_mock.add_response(
        url="https://example.com/file2.txt",
        content=b"File 2",
        headers={"Content-Type": "text/plain"},
    )

    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = DownloadTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(
            mock_run_ctx,
            urls=["https://example.com/file1.txt", "https://example.com/file2.txt"],
            save_dir="downloads",
        )

        assert len(result) == 2
        assert all(r["success"] for r in result)


async def test_download_tool_forbidden_url(tmp_path: Path) -> None:
    """Should return error for forbidden URLs when verification enabled."""
    from pai_agent_sdk.context import ToolConfig

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
        tool = DownloadTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(
            mock_run_ctx,
            urls=["http://192.168.1.1/secret.txt"],
            save_dir="downloads",
        )

        assert len(result) == 1
        assert result[0]["success"] is False
        assert "forbidden" in result[0]["error"].lower()
