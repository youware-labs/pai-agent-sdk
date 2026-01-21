"""Tests for pai_agent_sdk.toolsets.core.filesystem.grep module."""

from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import MagicMock

from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.toolsets.core.filesystem.grep import GrepTool


def test_grep_attributes(agent_context: AgentContext) -> None:
    """Should have correct name and description."""
    assert GrepTool.name == "grep"
    assert "regex" in GrepTool.description
    tool = GrepTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    instruction = tool.get_instruction(mock_run_ctx)
    assert instruction is not None


async def test_grep_find_pattern(tmp_path: Path) -> None:
    """Should find lines matching pattern."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool()

        (tmp_path / "test.py").write_text("def hello():\n    print('hello')\n\ndef world():\n    pass")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="def \\w+")
        assert isinstance(result, dict)
        assert len(result) == 2


async def test_grep_invalid_regex(tmp_path: Path) -> None:
    """Should return error for invalid regex."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool()

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="[invalid")
        assert "Error: Invalid regex" in result


async def test_grep_with_include_filter(tmp_path: Path) -> None:
    """Should filter files by include pattern."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool()

        (tmp_path / "test.py").write_text("hello world")
        (tmp_path / "test.txt").write_text("hello universe")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="hello", include="*.py")
        assert isinstance(result, dict)
        assert all("test.py" in key for key in result)


async def test_grep_context_lines(tmp_path: Path) -> None:
    """Should include context lines around matches."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool()

        content = "line1\nline2\nMATCH\nline4\nline5"
        (tmp_path / "test.txt").write_text(content)

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="MATCH", context_lines=2)
        assert isinstance(result, dict)
        match_data = next(iter(result.values()))
        assert "line2" in match_data["context"]
        assert "line4" in match_data["context"]


async def test_grep_max_results_limit(tmp_path: Path) -> None:
    """Should limit total matches."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool()

        # Create file with many matches
        content = "\n".join([f"match{i}" for i in range(20)])
        (tmp_path / "test.txt").write_text(content)

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="match", max_results=5)
        assert isinstance(result, dict)
        # Should have 5 matches + possible system message
        match_count = len([k for k in result if k != "<system>"])
        assert match_count == 5


async def test_grep_max_matches_per_file(tmp_path: Path) -> None:
    """Should limit matches per file."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool()

        content = "\n".join([f"match{i}" for i in range(10)])
        (tmp_path / "test.txt").write_text(content)

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="match", max_matches_per_file=3, max_results=100)
        assert isinstance(result, dict)
        match_count = len([k for k in result if k != "<system>"])
        assert match_count == 3


async def test_grep_no_matches(tmp_path: Path) -> None:
    """Should return empty dict when no matches."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool()

        (tmp_path / "test.txt").write_text("no matches here")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="xyz123")
        assert result == {}


async def test_grep_match_data_structure(tmp_path: Path) -> None:
    """Should return correct match data structure."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool()

        (tmp_path / "test.txt").write_text("line with pattern")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="pattern")
        assert isinstance(result, dict)
        match_data = next(iter(result.values()))
        assert "file_path" in match_data
        assert "line_number" in match_data
        assert "matching_line" in match_data
        assert "context" in match_data
        assert "context_start_line" in match_data


async def test_grep_max_files_limit(tmp_path: Path) -> None:
    """Should limit number of files searched."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool()

        # Create many files with matches
        for i in range(10):
            (tmp_path / f"file{i}.txt").write_text(f"match in file {i}")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        # Limit to 3 files
        result = await tool.call(mock_run_ctx, pattern="match", max_files=3)
        assert isinstance(result, dict)
        # Should only have matches from at most 3 files
        matched_files = {v["file_path"] for v in result.values() if isinstance(v, dict)}
        assert len(matched_files) <= 3


async def test_grep_multiple_files(tmp_path: Path) -> None:
    """Should search across multiple files and aggregate results."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool()

        (tmp_path / "file1.py").write_text("def foo(): pass")
        (tmp_path / "file2.py").write_text("def bar(): pass")
        (tmp_path / "file3.py").write_text("class Baz: pass")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="def \\w+", include="*.py")
        assert isinstance(result, dict)
        # Should find matches in file1 and file2
        matched_files = {v["file_path"] for v in result.values() if isinstance(v, dict)}
        assert "file1.py" in matched_files
        assert "file2.py" in matched_files
        assert "file3.py" not in matched_files


async def test_grep_skips_directories(tmp_path: Path) -> None:
    """Should skip directories without error."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool()

        # Create a directory and a file
        (tmp_path / "subdir").mkdir()
        (tmp_path / "test.txt").write_text("searchable content")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        # Should not raise error when glob includes directory
        result = await tool.call(mock_run_ctx, pattern="searchable", include="*")
        assert isinstance(result, dict)
        # Should find the file match
        assert len(result) == 1


async def test_grep_large_result_truncation(tmp_path: Path) -> None:
    """Should truncate results when they exceed 60000 characters."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool()

        # Create file with many matches that will produce large output
        # Each match with context will be substantial
        lines = [f"match_{i}_" + "x" * 500 for i in range(200)]
        (tmp_path / "large.txt").write_text("\n".join(lines))

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(
            mock_run_ctx,
            pattern="match_\\d+",
            max_results=-1,  # unlimited
            max_matches_per_file=-1,  # unlimited
            context_lines=5,
        )
        assert isinstance(result, dict)
        # When truncated, context is dropped and system message added
        if "<system>" in result:
            assert "truncated" in result["<system>"].lower()
            # Truncated results should not have 'context' field
            for key, val in result.items():
                if key != "<system>" and isinstance(val, dict):
                    assert "context" not in val


async def test_grep_unreadable_file_handling(tmp_path: Path) -> None:
    """Should gracefully handle files that cannot be read."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool()

        # Create a valid file and a binary file that may cause decode errors
        (tmp_path / "valid.txt").write_text("searchable text")
        (tmp_path / "binary.bin").write_bytes(b"\x00\x01\x02\xff\xfe")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        # Should not raise, should skip unreadable files
        result = await tool.call(mock_run_ctx, pattern="searchable", include="*")
        assert isinstance(result, dict)
        # Should find match in valid.txt
        assert any("valid.txt" in k for k in result)


async def test_grep_excludes_gitignored_files(tmp_path: Path) -> None:
    """Should exclude files matching .gitignore patterns by default."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool()

        # Create .gitignore
        (tmp_path / ".gitignore").write_text("node_modules/\n")

        # Create files
        (tmp_path / "main.py").write_text("hello world")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "pkg.js").write_text("hello from node")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="hello")
        assert isinstance(result, dict)

        # Should have gitignore info
        assert "<gitignore_excluded>" in result
        assert "<note>" in result

        # Should find match only in main.py
        match_keys = [k for k in result if not k.startswith("<")]
        assert len(match_keys) == 1
        assert "main.py" in match_keys[0]

        # Summary should mention excluded paths
        summary = result["<gitignore_excluded>"]
        assert any("node_modules/" in s for s in summary)


async def test_grep_include_ignored_flag(tmp_path: Path) -> None:
    """Should include gitignored files when include_ignored=True."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool()

        # Create .gitignore
        (tmp_path / ".gitignore").write_text("*.log\n")
        (tmp_path / "app.py").write_text("hello app")
        (tmp_path / "debug.log").write_text("hello debug")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="hello", include_ignored=True)
        assert isinstance(result, dict)

        # Should not have gitignore info
        assert "<gitignore_excluded>" not in result

        # Should find matches in both files
        match_keys = [k for k in result if not k.startswith("<")]
        assert len(match_keys) == 2
        assert any("app.py" in k for k in match_keys)
        assert any("debug.log" in k for k in match_keys)


async def test_grep_no_gitignore_no_excluded_info(tmp_path: Path) -> None:
    """Should not include gitignore info when no .gitignore exists."""
    async with AsyncExitStack() as stack:
        env = await stack.enter_async_context(
            LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
        )
        ctx = await stack.enter_async_context(AgentContext(env=env))
        tool = GrepTool()

        (tmp_path / "file.py").write_text("hello world")

        mock_run_ctx = MagicMock(spec=RunContext)
        mock_run_ctx.deps = ctx

        result = await tool.call(mock_run_ctx, pattern="hello")
        assert isinstance(result, dict)

        # No .gitignore means no excluded info
        assert "<gitignore_excluded>" not in result
        assert "<note>" not in result
