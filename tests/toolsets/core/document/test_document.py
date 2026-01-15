"""Tests for document conversion tools (PDF and Office)."""

import shutil
from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from inline_snapshot import snapshot
from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment

# Test file directory
TEST_FILES_DIR = Path(__file__).parent


@pytest.fixture
def pdf_file(tmp_path: Path) -> Path:
    """Copy dummy.pdf to tmp_path for testing."""
    src = TEST_FILES_DIR / "dummy.pdf"
    if src.exists():
        dst = tmp_path / "dummy.pdf"
        shutil.copy(src, dst)
        return dst
    pytest.skip("dummy.pdf not found")


@pytest.fixture
def docx_file(tmp_path: Path) -> Path:
    """Copy dummy.docx to tmp_path for testing."""
    src = TEST_FILES_DIR / "dummy.docx"
    if src.exists():
        dst = tmp_path / "dummy.docx"
        shutil.copy(src, dst)
        return dst
    pytest.skip("dummy.docx not found")


# --- PDF Convert Tool Tests ---

# Check if PDF tools are available
try:
    from pai_agent_sdk.toolsets.core.document.pdf import PdfConvertTool

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


@pytest.mark.skipif(not PDF_AVAILABLE, reason="pymupdf not installed")
def test_pdf_convert_tool_attributes(agent_context: AgentContext) -> None:
    """Should have correct name and description."""
    tool = PdfConvertTool()
    assert tool.name == "pdf_convert"
    assert "PDF" in tool.description or "pdf" in tool.description.lower()


@pytest.mark.skipif(not PDF_AVAILABLE, reason="pymupdf not installed")
async def test_pdf_convert_file_not_found(tmp_path: Path) -> None:
    """Should return error when file not found."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = PdfConvertTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="nonexistent.pdf")
        assert result["success"] is False
        assert "not found" in result["error"].lower()


@pytest.mark.skipif(not PDF_AVAILABLE, reason="pymupdf not installed")
async def test_pdf_convert_not_pdf_file(tmp_path: Path) -> None:
    """Should return error when file is not PDF."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = PdfConvertTool()

        # Create a non-PDF file
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a pdf")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="test.txt")
        assert result["success"] is False
        assert "not a pdf" in result["error"].lower()


@pytest.mark.skipif(not PDF_AVAILABLE, reason="pymupdf not installed")
async def test_pdf_convert_success(tmp_path: Path, pdf_file: Path) -> None:
    """Should successfully convert PDF to markdown."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = PdfConvertTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="dummy.pdf")
        assert result["success"] is True
        assert result == snapshot({
            "success": True,
            "export_path": "export_dummy",
            "markdown_path": "export_dummy/dummy.md",
            "total_pages": 1,
            "converted_pages": 1,
            "page_range": "1-1",
        })

        # Verify export directory was created
        export_dir = tmp_path / result["export_path"]
        assert export_dir.exists()
        assert (tmp_path / result["markdown_path"]).exists()


@pytest.mark.skipif(not PDF_AVAILABLE, reason="pymupdf not installed")
async def test_pdf_convert_page_range(tmp_path: Path, pdf_file: Path) -> None:
    """Should respect page range parameters."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = PdfConvertTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="dummy.pdf", page_start=1, page_end=1)
        assert result["success"] is True
        assert result["converted_pages"] == 1
        assert result == snapshot({
            "success": True,
            "export_path": "export_dummy",
            "markdown_path": "export_dummy/dummy.md",
            "total_pages": 1,
            "converted_pages": 1,
            "page_range": "1-1",
        })

        # Verify export directory was created
        export_dir = tmp_path / result["export_path"]
        assert export_dir.exists()
        assert (tmp_path / result["markdown_path"]).exists()
        assert (tmp_path / result["markdown_path"]).read_text() == snapshot("""\
**Dummy PDF file** \n\

""")


# --- Office Convert Tool Tests ---

# Check if Office tools are available
try:
    from pai_agent_sdk.toolsets.core.document.office import OfficeConvertTool

    OFFICE_AVAILABLE = True
except ImportError:
    OFFICE_AVAILABLE = False


@pytest.mark.skipif(not OFFICE_AVAILABLE, reason="markitdown not installed")
def test_office_convert_tool_attributes(agent_context: AgentContext) -> None:
    """Should have correct name and description."""
    tool = OfficeConvertTool()
    assert tool.name == "office_to_markdown"
    assert "Office" in tool.description or "Word" in tool.description


@pytest.mark.skipif(not OFFICE_AVAILABLE, reason="markitdown not installed")
async def test_office_convert_file_not_found(tmp_path: Path) -> None:
    """Should return error when file not found."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = OfficeConvertTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="nonexistent.docx")
        assert result["success"] is False
        assert "not found" in result["error"].lower()


@pytest.mark.skipif(not OFFICE_AVAILABLE, reason="markitdown not installed")
async def test_office_convert_unsupported_format(tmp_path: Path) -> None:
    """Should return error for unsupported formats."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = OfficeConvertTool()

        # Create an unsupported file
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not an office doc")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="test.txt")
        assert result["success"] is False
        assert "unsupported" in result["error"].lower()


@pytest.mark.skipif(not OFFICE_AVAILABLE, reason="markitdown not installed")
async def test_office_convert_docx_success(tmp_path: Path, docx_file: Path) -> None:
    """Should successfully convert DOCX to markdown."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = OfficeConvertTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, file_path="dummy.docx")
        assert result["success"] is True
        assert result == snapshot({
            "success": True,
            "export_path": "export_dummy",
            "markdown_path": "export_dummy/dummy.md",
        })

        # Verify export directory was created
        export_dir = tmp_path / result["export_path"]
        assert export_dir.exists()
        assert (tmp_path / result["markdown_path"]).exists()
        assert "这是一首简单的小情歌" in (tmp_path / result["markdown_path"]).read_text()
        assert "images/" in (tmp_path / result["markdown_path"]).read_text()
