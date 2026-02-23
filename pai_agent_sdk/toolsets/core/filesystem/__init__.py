"""Filesystem-related tools.

Tools for file and directory operations.
"""

from pai_agent_sdk.toolsets.core.base import BaseTool
from pai_agent_sdk.toolsets.core.filesystem._types import (
    BatchMkdirResponse,
    CopyResult,
    EditItem,
    FileInfo,
    FileInfoWithStats,
    GrepMatch,
    MkdirResult,
    MkdirSummary,
    MoveResult,
    PathPair,
    ViewMetadata,
    ViewReadingParams,
    ViewSegment,
    ViewTruncationInfo,
)
from pai_agent_sdk.toolsets.core.filesystem.edit import EditTool, MultiEditTool
from pai_agent_sdk.toolsets.core.filesystem.glob import GlobTool
from pai_agent_sdk.toolsets.core.filesystem.grep import GrepTool
from pai_agent_sdk.toolsets.core.filesystem.ls import ListTool
from pai_agent_sdk.toolsets.core.filesystem.mkdir import MkdirTool
from pai_agent_sdk.toolsets.core.filesystem.move_copy import CopyTool, MoveTool
from pai_agent_sdk.toolsets.core.filesystem.view import ViewTool
from pai_agent_sdk.toolsets.core.filesystem.write import WriteTool

tools: list[type[BaseTool]] = [
    GlobTool,
    GrepTool,
    ListTool,
    ViewTool,
    EditTool,
    MultiEditTool,
    WriteTool,
    MkdirTool,
    MoveTool,
    CopyTool,
]

__all__ = [
    "BatchMkdirResponse",
    "CopyResult",
    "CopyTool",
    "EditItem",
    "EditTool",
    "FileInfo",
    "FileInfoWithStats",
    "GlobTool",
    "GrepMatch",
    "GrepTool",
    "ListTool",
    "MkdirResult",
    "MkdirSummary",
    "MkdirTool",
    "MoveResult",
    "MoveTool",
    "MultiEditTool",
    "PathPair",
    "ViewMetadata",
    "ViewReadingParams",
    "ViewSegment",
    "ViewTool",
    "ViewTruncationInfo",
    "WriteTool",
    "tools",
]
