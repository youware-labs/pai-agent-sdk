"""Gitignore helper for filesystem tools.

Provides utilities to filter files based on .gitignore patterns.
"""

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


async def filter_gitignored(
    files: list[str],
    file_operator: FileOperator,
) -> list[str]:
    """Filter out files that match gitignore patterns.

    Args:
        files: List of file paths (relative paths)
        file_operator: FileOperator to read .gitignore

    Returns:
        List of files that are NOT gitignored
    """
    # Try to load .gitignore
    try:
        if not await file_operator.exists(".gitignore"):
            return files
        content = await file_operator.read_file(".gitignore")
        spec = _parse_gitignore_content(content)
    except Exception:
        # If we can't read .gitignore, return all files
        return files

    # pathspec.match_files returns files that MATCH the patterns (i.e., ignored files)
    # We want files that DON'T match (i.e., not ignored)
    ignored_set = set(spec.match_files(files))
    return [f for f in files if f not in ignored_set]


__all__ = ["filter_gitignored"]
