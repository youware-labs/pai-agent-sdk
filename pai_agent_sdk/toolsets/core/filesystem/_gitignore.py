"""Gitignore helper for filesystem tools.

Provides utilities to filter files based on .gitignore patterns.
"""

from collections import Counter
from dataclasses import dataclass, field

import anyio
import pathspec
from agent_environment import FileOperator


def _parse_gitignore_content(content: str) -> pathspec.PathSpec:
    """Parse gitignore content into a PathSpec.

    Args:
        content: Content of .gitignore file

    Returns:
        PathSpec object for matching
    """
    return pathspec.PathSpec.from_lines("gitignore", content.splitlines())


def _get_top_level_dir(path: str) -> str:
    """Extract top-level directory from a path.

    Examples:
        "node_modules/foo/bar.js" -> "node_modules/"
        "src/main.py" -> "src/"
        "file.txt" -> "file.txt"
    """
    parts = path.split("/")
    if len(parts) > 1:
        return parts[0] + "/"
    return path


@dataclass
class GitignoreFilterResult:
    """Result of gitignore filtering with summary information."""

    kept: list[str]
    ignored: list[str] = field(default_factory=list)

    def get_ignored_summary(self, max_items: int = 5) -> list[str]:
        """Generate a summary of ignored paths grouped by top-level directory.

        Args:
            max_items: Maximum number of summary items to return (must be > 0)

        Returns:
            List of summary strings, e.g., ["node_modules/ (1234 files)", ".venv/ (567 files)"]
        """
        if not self.ignored or max_items <= 0:
            return []

        # Count files by top-level directory
        dir_counts: Counter[str] = Counter()
        for path in self.ignored:
            top_dir = _get_top_level_dir(path)
            dir_counts[top_dir] += 1

        # Sort by count (descending) and format
        sorted_dirs = dir_counts.most_common()
        summary: list[str] = []

        for i, (dir_name, count) in enumerate(sorted_dirs):
            if i >= max_items:
                remaining = len(sorted_dirs) - max_items
                summary.append(f"... and {remaining} more patterns")
                break
            unit = "file" if count == 1 else "files"
            summary.append(f"{dir_name} ({count} {unit})")

        return summary


async def filter_gitignored(
    files: list[str],
    file_operator: FileOperator,
) -> GitignoreFilterResult:
    """Filter out files that match gitignore patterns.

    Args:
        files: List of file paths (relative paths)
        file_operator: FileOperator to read .gitignore

    Returns:
        GitignoreFilterResult containing kept files and ignored files
    """
    # Try to load .gitignore
    try:
        if not await file_operator.exists(".gitignore"):
            return GitignoreFilterResult(kept=files, ignored=[])
        content = await file_operator.read_file(".gitignore")
        spec = _parse_gitignore_content(content)
    except Exception:
        # If we can't read .gitignore, return all files
        return GitignoreFilterResult(kept=files, ignored=[])

    # CPU-intensive operation: run in thread pool to avoid blocking event loop
    def _compute_filter() -> tuple[list[str], list[str]]:
        # pathspec.match_files returns files that MATCH the patterns (i.e., ignored files)
        ignored_set = set(spec.match_files(files))
        kept = [f for f in files if f not in ignored_set]
        ignored = [f for f in files if f in ignored_set]
        return kept, ignored

    kept, ignored = await anyio.to_thread.run_sync(_compute_filter)  # type: ignore[arg-type]
    return GitignoreFilterResult(kept=kept, ignored=ignored)


__all__ = ["GitignoreFilterResult", "filter_gitignored"]
