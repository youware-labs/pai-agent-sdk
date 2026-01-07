"""Office/EPub to Markdown conversion tool.

Converts Office documents (Word, PowerPoint, Excel) and EPub files to markdown.
Requires optional dependency: markitdown.

Install with: pip install pai-agent-sdk[document]
"""

import base64
import re
import uuid
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
    from markitdown import MarkItDown
except ImportError as e:
    raise ImportError(
        "The 'markitdown' package is required for OfficeConvertTool. Install with: pip install pai-agent-sdk[document]"
    ) from e

_PROMPTS_DIR = Path(__file__).parent / "prompts"

SUPPORTED_EXTENSIONS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".epub"}


@cache
def _load_instruction() -> str:
    """Load office convert instruction from prompts/office.md."""
    prompt_file = _PROMPTS_DIR / "office.md"
    return prompt_file.read_text()


async def _run_in_threadpool(func, *args, **kwargs):
    """Run a sync function in a thread pool."""
    import functools

    import anyio.to_thread

    return await anyio.to_thread.run_sync(functools.partial(func, *args, **kwargs))


class OfficeConvertTool(BaseTool):
    """Tool for converting Office documents and EPub to markdown."""

    name = "office_to_markdown"
    description = "Convert Office documents (Word, PowerPoint, Excel) and EPub to markdown."

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str:
        """Load instruction from prompts/office.md."""
        return _load_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        file_path: Annotated[str, Field(description="Path to the document file to convert.")],
    ) -> dict[str, Any]:
        file_op = ctx.deps.file_operator

        # Check file exists
        if not await file_op.exists(file_path):
            return {"error": f"File not found: {file_path}", "success": False}

        # Resolve absolute path
        doc_path = file_op._default_path / file_path
        if not doc_path.exists():
            doc_path = Path(file_path)
            if not doc_path.is_absolute():
                doc_path = file_op._default_path / file_path

        # Check extension
        if doc_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return {
                "error": f"Unsupported format: {doc_path.suffix}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
                "success": False,
            }

        # Create export directory
        export_dir_name = f"export_{doc_path.stem}"
        export_dir = doc_path.parent / export_dir_name
        images_dir = export_dir / "images"

        try:
            export_dir.mkdir(exist_ok=True)
            images_dir.mkdir(exist_ok=True)
        except Exception as e:
            return {"error": f"Failed to create export directory: {e}", "success": False}

        # Convert document
        try:
            md = MarkItDown(enable_plugins=True)
            result = await _run_in_threadpool(md.convert, f"file://{doc_path.as_posix()}", keep_data_uris=True)
            content = result.text_content
        except Exception as e:
            return {"error": f"Failed to convert document: {e}", "success": False}

        # Extract base64 images and save to files
        content = self._extract_images(content, images_dir, doc_path)

        # Write markdown file
        md_filename = f"{doc_path.stem}.md"
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
        }

    def _extract_images(self, content: str, images_dir: Path, doc_path: Path) -> str:
        """Extract base64 image data URIs and save as files.

        Args:
            content: Markdown content with base64 images.
            images_dir: Directory to save extracted images.
            doc_path: Original document path (for relative path calculation).

        Returns:
            Modified markdown with file paths instead of data URIs.
        """
        pattern = r"!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)"
        prefix = uuid.uuid4().hex[:8]
        counter = [0]  # Use list for closure mutation

        def replace_image(match):
            alt_text, image_format, base64_data = match.groups()
            counter[0] += 1

            try:
                image_data = base64.b64decode(base64_data)
                ext = f".{image_format.lower()}" if image_format else ".png"
                filename = f"{prefix}_{counter[0]}{ext}"
                image_file = images_dir / filename

                with open(image_file, "wb") as f:
                    f.write(image_data)

                return f"![{alt_text}](./images/{filename})"
            except Exception:
                return match.group(0)

        return re.sub(pattern, replace_image, content)
