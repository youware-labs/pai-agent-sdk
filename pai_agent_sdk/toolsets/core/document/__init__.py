"""Document processing tools.

Tools for converting PDF and Office documents to markdown format.
Requires optional dependencies - install with: pip install pai-agent-sdk[document]
"""

from pai_agent_sdk.toolsets.core.base import BaseTool

tools: list[type[BaseTool]] = []
__all__ = ["tools"]

# PDF conversion tool (requires pymupdf, pymupdf4llm)
try:
    from pai_agent_sdk.toolsets.core.document.pdf import PdfConvertTool

    tools.append(PdfConvertTool)
    __all__.append("PdfConvertTool")
except ImportError:
    pass

# Office/EPub conversion tool (requires markitdown)
try:
    from pai_agent_sdk.toolsets.core.document.office import OfficeConvertTool

    tools.append(OfficeConvertTool)
    __all__.append("OfficeConvertTool")
except ImportError:
    pass
