"""Grep tool for content search."""

import json
import re
from functools import cache
from pathlib import Path
from typing import Annotated, Any, cast

from agent_environment import FileOperator
from pydantic import Field
from pydantic_ai import RunContext

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.base import BaseTool
from pai_agent_sdk.toolsets.core.filesystem._gitignore import filter_gitignored
from pai_agent_sdk.toolsets.core.filesystem._types import GrepMatch

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@cache
def _load_instruction() -> str:
    """Load grep instruction from prompts/grep.md."""
    prompt_file = _PROMPTS_DIR / "grep.md"
    return prompt_file.read_text()


class GrepTool(BaseTool):
    """Tool for searching file contents using regular expressions."""

    name = "grep"
    description = "Search file contents using regex patterns. Returns matches with context lines."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available (requires file_operator)."""
        if ctx.deps.file_operator is None:
            logger.debug("GrepTool unavailable: file_operator is not configured")
            return False
        return True

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Load instruction from prompts/grep.md."""
        return _load_instruction()

    def _search_file_for_matches(
        self,
        file_path: str,
        content: str,
        regex_pattern: re.Pattern[str],
        context_lines: int,
        max_matches_per_file: int,
    ) -> dict[str, GrepMatch]:
        """Search a single file for regex matches."""
        matches: dict[str, GrepMatch] = {}
        file_matches = 0

        lines = content.splitlines(keepends=True)

        for i, line in enumerate(lines):
            if regex_pattern.search(line):
                if max_matches_per_file > 0 and file_matches >= max_matches_per_file:
                    break

                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)

                context = "".join(lines[start:end])

                match_data = GrepMatch(
                    file_path=file_path,
                    line_number=i + 1,
                    matching_line=line.rstrip("\n"),
                    context=context,
                    context_start_line=start + 1,
                )

                matches[f"{file_path}:{i + 1}"] = match_data
                file_matches += 1

        return matches

    async def call(
        self,
        ctx: RunContext[AgentContext],
        pattern: Annotated[str, Field(description="Regular expression pattern to search for")],
        include: Annotated[
            str,
            Field(description="Glob pattern to filter files (default: **/*)", default="**/*"),
        ] = "**/*",
        context_lines: Annotated[
            int,
            Field(description="Context lines before/after matches (default: 2)", default=2),
        ] = 2,
        max_results: Annotated[
            int,
            Field(description="Max total matches (default: 100, -1 for unlimited)", default=100),
        ] = 100,
        max_matches_per_file: Annotated[
            int,
            Field(description="Max matches per file (default: 20, -1 for unlimited)", default=20),
        ] = 20,
        max_files: Annotated[
            int,
            Field(description="Max files to search (default: 50, -1 for unlimited)", default=50),
        ] = 50,
        include_ignored: Annotated[
            bool,
            Field(description="Include files ignored by .gitignore (default: false)", default=False),
        ] = False,
    ) -> dict[str, Any] | str:
        """Search file contents using regular expressions."""
        file_operator = cast(FileOperator, ctx.deps.file_operator)

        try:
            compiled_pattern = re.compile(pattern, re.UNICODE)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        files = await file_operator.glob(include)

        # Filter out gitignored files by default
        if not include_ignored:
            files = await filter_gitignored(files, file_operator)

        files_to_search = files[:max_files] if max_files > 0 else files

        results: dict[str, Any] = {}
        total_matches_found = 0
        truncated_by_total_limit = False

        for file_path in files_to_search:
            if await file_operator.is_dir(file_path):
                continue

            try:
                content = await file_operator.read_file(file_path)
            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {e}")
                continue

            file_matches = self._search_file_for_matches(
                file_path,
                content,
                compiled_pattern,
                context_lines,
                max_matches_per_file,
            )

            for match_key, match_data in file_matches.items():
                if max_results > 0 and total_matches_found >= max_results:
                    truncated_by_total_limit = True
                    results["<system>"] = f"Hit global limit: {max_results} matches"
                    break

                results[match_key] = match_data
                total_matches_found += 1

            if truncated_by_total_limit:
                break

        logger.info(f"Total matches found: {total_matches_found}")

        if len(json.dumps(results, default=str)) > 60000:
            logger.info("Results too long, dropping context")
            truncated_results: dict[str, Any] = {
                match: {
                    "line_number": match_data["line_number"],
                    "matching_line": match_data["matching_line"],
                    "context_start_line": match_data["context_start_line"],
                }
                for match, match_data in results.items()
                if isinstance(match_data, dict)
            }
            truncated_results["<system>"] = "Results truncated. Use `view` to read specific files."
            return truncated_results

        return results


__all__ = ["GrepTool"]
