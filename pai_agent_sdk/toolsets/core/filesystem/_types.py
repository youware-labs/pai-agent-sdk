"""Type definitions for filesystem tools.

This module provides shared type definitions used across filesystem tools.
"""

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class PathPair(TypedDict):
    """Source and destination path pair for batch operations."""

    src: str
    """Source path."""
    dst: str
    """Destination path."""


class EditItem(BaseModel):
    """Single edit operation for multi-edit tool."""

    old_string: str = Field(
        ...,
        description="Text to replace (must match file contents exactly, including all whitespace and indentation)",
    )
    new_string: str = Field(..., description="New text to replace the old text with")
    replace_all: bool = Field(
        default=False,
        description="Replace all occurrences of old_string. Defaults to false (replace only first occurrence).",
    )


class FileInfo(TypedDict):
    """File information returned by ls tool."""

    name: str
    """File or directory name."""
    path: str
    """Relative path from working directory."""
    type: str
    """'file' or 'directory'."""


class FileInfoWithStats(FileInfo, total=False):
    """File information with optional stats (for files only)."""

    size: int
    """File size in bytes (files only)."""
    modified: float
    """Modification time as Unix timestamp (files only)."""
    error: str
    """Error message if stats retrieval failed."""


class GrepMatch(TypedDict):
    """Single grep match result."""

    file_path: str
    """Path to the file containing the match."""
    line_number: int
    """Line number of the match (1-indexed)."""
    matching_line: str
    """The line containing the match."""
    context: str
    """Context lines around the match."""
    context_start_line: int
    """Starting line number of the context."""


class ViewMetadata(TypedDict):
    """Metadata for text file viewing."""

    file_path: str
    """Name of the file."""
    total_lines: int
    """Total number of lines in the file."""
    total_characters: int
    """Total number of characters in the file."""
    file_size_bytes: int
    """File size in bytes."""
    current_segment: "ViewSegment"
    """Information about the current segment being viewed."""
    reading_parameters: "ViewReadingParams"
    """Parameters used for reading."""
    truncation_info: "ViewTruncationInfo"
    """Information about any truncation applied."""


class ViewSegment(TypedDict):
    """Information about the current viewed segment."""

    start_line: int
    """Starting line number (1-indexed)."""
    end_line: int
    """Ending line number (1-indexed)."""
    lines_to_show: int
    """Number of lines in this segment."""
    has_more_content: bool
    """True if there is more content after this segment."""


class ViewReadingParams(TypedDict):
    """Reading parameters for view tool."""

    line_offset: int | None
    """Line offset used (None if default)."""
    line_limit: int
    """Line limit used."""


class ViewTruncationInfo(TypedDict):
    """Truncation information for view tool."""

    lines_truncated: bool
    """True if any lines were truncated."""
    content_truncated: bool
    """True if overall content was truncated."""
    max_line_length: int
    """Maximum line length used."""


class MkdirResult(TypedDict):
    """Result of mkdir operation for a single path."""

    path: str
    """Path that was created or failed."""
    success: bool
    """True if directory was created successfully."""
    message: str
    """Success or error message."""


class MkdirSummary(TypedDict):
    """Summary of batch mkdir operation."""

    total: int
    """Total number of paths processed."""
    successful: int
    """Number of paths successfully created."""
    failed: int
    """Number of paths that failed."""


class BatchMkdirResponse(TypedDict):
    """Response from batch mkdir operation."""

    success: bool
    """True if all paths were created successfully."""
    message: str
    """Summary message."""
    results: list[MkdirResult]
    """Individual results for each path."""
    summary: MkdirSummary
    """Summary statistics."""


class MoveResult(TypedDict):
    """Result of move operation for a single path pair."""

    src: str
    """Source path."""
    dst: str
    """Destination path."""
    success: bool
    """True if move was successful."""
    message: str
    """Success or error message."""


class CopyResult(TypedDict):
    """Result of copy operation for a single path pair."""

    src: str
    """Source path."""
    dst: str
    """Destination path."""
    success: bool
    """True if copy was successful."""
    message: str
    """Success or error message."""


__all__ = [
    "BatchMkdirResponse",
    "CopyResult",
    "EditItem",
    "FileInfo",
    "FileInfoWithStats",
    "GrepMatch",
    "MkdirResult",
    "MkdirSummary",
    "MoveResult",
    "PathPair",
    "ViewMetadata",
    "ViewReadingParams",
    "ViewSegment",
    "ViewTruncationInfo",
]
