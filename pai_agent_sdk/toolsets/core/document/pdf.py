"""PDF to Markdown conversion tool.

Converts PDF files to markdown format with embedded images extracted.
Requires optional dependencies: pymupdf, pymupdf4llm.

Install with: pip install pai-agent-sdk[document]
"""

from functools import cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field
from pydantic_ai import RunContext

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.base import BaseTool

logger = get_logger(__name__)

# Optional dependency check
try:
    import pymupdf
    import pymupdf4llm
except ImportError as e:
    raise ImportError(
        "The 'pymupdf' and 'pymupdf4llm' packages are required for PdfConvertTool. "
        "Install with: pip install pai-agent-sdk[document]"
    ) from e

_PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_MAX_PAGES = 20


@cache
def _load_instruction() -> str:
    """Load PDF convert instruction from prompts/pdf.md."""
    prompt_file = _PROMPTS_DIR / "pdf.md"
    return prompt_file.read_text()


async def _run_in_threadpool(func, *args, **kwargs):
    """Run a sync function in a thread pool."""
    import functools

    import anyio.to_thread

    return await anyio.to_thread.run_sync(functools.partial(func, *args, **kwargs))


def _resolve_pdf_path(file_op, file_path: str) -> Path:
    """Resolve PDF path to absolute path."""
    pdf_path = file_op._default_path / file_path
    if not pdf_path.exists():
        pdf_path = Path(file_path)
        if not pdf_path.is_absolute():
            pdf_path = file_op._default_path / file_path
    return pdf_path


def _validate_page_params(
    page_start: int | None, page_end: int | None, total_pages: int
) -> tuple[int, int, str | None]:
    """Validate and calculate page range.

    Returns:
        (start_page, end_page, error_message) - error_message is None if valid
    """
    # Validate inputs
    if page_start is not None and page_start <= 0:
        return 0, 0, f"Invalid page_start: {page_start}. Must be >= 1."
    if page_end is not None and page_end != -1 and page_end <= 0:
        return 0, 0, f"Invalid page_end: {page_end}. Must be >= 1 or -1."

    # Calculate page range (convert to 0-based for pymupdf)
    start_page = (page_start - 1) if page_start else 0
    if page_end == -1:
        end_page = total_pages - 1
    elif page_end:
        end_page = page_end - 1
    else:
        end_page = min(start_page + DEFAULT_MAX_PAGES - 1, total_pages - 1)

    # Validate range against PDF size
    if start_page >= total_pages:
        return 0, 0, f"Invalid page_start: PDF has only {total_pages} pages."
    if end_page < start_page:
        return 0, 0, "Invalid range: page_end must be >= page_start."

    return start_page, min(end_page, total_pages - 1), None


class PdfConvertTool(BaseTool):
    """Tool for converting PDF files to markdown."""

    name = "pdf_convert"
    description = "Convert PDF to markdown with image extraction."

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str:
        """Load instruction from prompts/pdf.md."""
        return _load_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        file_path: Annotated[str, Field(description="Path to the PDF file to convert.")],
        page_start: Annotated[
            int | None,
            Field(description="Starting page number (1-based). Default: 1."),
        ] = None,
        page_end: Annotated[
            int | None,
            Field(description="Ending page number (1-based, inclusive). Default: 20. Use -1 for all pages."),
        ] = None,
    ) -> dict[str, Any]:
        file_op = ctx.deps.file_operator

        # Check file exists
        if not await file_op.exists(file_path):
            return {"error": f"File not found: {file_path}", "success": False}

        # Resolve absolute path for pymupdf
        pdf_path = _resolve_pdf_path(file_op, file_path)

        if pdf_path.suffix.lower() != ".pdf":
            return {"error": f"Not a PDF file: {file_path}", "success": False}

        # Create export directory
        export_dir_name = f"export_{pdf_path.stem}"
        export_dir = pdf_path.parent / export_dir_name
        images_dir = export_dir / "images"

        try:
            export_dir.mkdir(exist_ok=True)
            images_dir.mkdir(exist_ok=True)
        except Exception as e:
            return {"error": f"Failed to create export directory: {e}", "success": False}

        # Get total page count
        try:

            def get_page_count(path):
                with pymupdf.open(path) as doc:
                    return len(doc)

            total_pages = await _run_in_threadpool(get_page_count, pdf_path)
        except Exception as e:
            logger.exception("Failed to read PDF file")
            return {"error": f"Failed to read PDF file: {e}", "success": False}

        # Validate and calculate page range
        start_page, actual_end_page, error = _validate_page_params(page_start, page_end, total_pages)
        if error:
            return {"error": error, "success": False}

        converted_pages = actual_end_page - start_page + 1

        # Convert PDF to markdown
        try:
            content = await _run_in_threadpool(
                pymupdf4llm.to_markdown,  # type: ignore[union-attr]
                str(pdf_path),
                write_images=True,
                image_path=str(images_dir),
                pages=list(range(start_page, actual_end_page + 1)),
            )
        except Exception as e:
            return {"error": f"Failed to convert PDF: {e}", "success": False}

        # Fix image paths in markdown (pymupdf4llm uses absolute paths)
        content = content.replace(str(images_dir) + "/", "./images/")

        # Write markdown file
        md_filename = f"{pdf_path.stem}.md"
        md_path = export_dir / md_filename

        try:
            md_path.write_text(content, encoding="utf-8")
        except Exception as e:
            return {"error": f"Failed to write markdown: {e}", "success": False}

        # Get relative path for response
        try:
            rel_export_dir = export_dir.relative_to(file_op._default_path)
            rel_md_path = md_path.relative_to(file_op._default_path)
        except ValueError:
            rel_export_dir = export_dir
            rel_md_path = md_path

        return {
            "success": True,
            "export_path": str(rel_export_dir),
            "markdown_path": str(rel_md_path),
            "total_pages": total_pages,
            "converted_pages": converted_pages,
            "page_range": f"{start_page + 1}-{actual_end_page + 1}",
        }
