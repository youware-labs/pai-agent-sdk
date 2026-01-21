"""Tests for gitignore filtering functionality."""

import pytest

from pai_agent_sdk.toolsets.core.filesystem._gitignore import (
    GitignoreFilterResult,
    _get_top_level_dir,
    _parse_gitignore_content,
    filter_gitignored,
)


def test_parse_gitignore_content_simple_patterns():
    """Test parsing simple gitignore patterns."""
    content = """
# Comment
*.pyc
__pycache__/
.venv/
node_modules/
"""
    spec = _parse_gitignore_content(content)

    # Should match ignored files
    assert spec.match_file("test.pyc")
    assert spec.match_file("__pycache__/cache.py")
    assert spec.match_file(".venv/lib/python/site.py")
    assert spec.match_file("node_modules/package/index.js")

    # Should not match non-ignored files
    assert not spec.match_file("test.py")
    assert not spec.match_file("src/main.py")


def test_parse_gitignore_content_negation():
    """Test parsing gitignore with negation patterns."""
    content = """
*.log
!important.log
"""
    spec = _parse_gitignore_content(content)

    assert spec.match_file("debug.log")
    # Negation should exclude from ignore
    assert not spec.match_file("important.log")


def test_parse_gitignore_content_directory_patterns():
    """Test parsing directory-specific patterns."""
    content = """
build/
dist/
"""
    spec = _parse_gitignore_content(content)

    assert spec.match_file("build/output.js")
    assert spec.match_file("dist/bundle.js")
    assert not spec.match_file("src/build.py")


def test_get_top_level_dir():
    """Test extracting top-level directory from paths."""
    assert _get_top_level_dir("node_modules/foo/bar.js") == "node_modules/"
    assert _get_top_level_dir("src/main.py") == "src/"
    assert _get_top_level_dir(".venv/lib/site.py") == ".venv/"
    assert _get_top_level_dir("file.txt") == "file.txt"
    assert _get_top_level_dir("deep/nested/path/file.py") == "deep/"


def test_gitignore_filter_result_get_ignored_summary():
    """Test GitignoreFilterResult.get_ignored_summary."""
    result = GitignoreFilterResult(
        kept=["src/main.py"],
        ignored=[
            "node_modules/a.js",
            "node_modules/b.js",
            "node_modules/sub/c.js",
            ".venv/lib/x.py",
            "build/output.js",
        ],
    )

    summary = result.get_ignored_summary()
    assert len(summary) == 3
    assert "node_modules/ (3 files)" in summary
    assert ".venv/ (1 file)" in summary
    assert "build/ (1 file)" in summary


def test_gitignore_filter_result_get_ignored_summary_max_items():
    """Test GitignoreFilterResult.get_ignored_summary with max_items."""
    result = GitignoreFilterResult(
        kept=[],
        ignored=[
            "dir1/a.py",
            "dir2/b.py",
            "dir3/c.py",
            "dir4/d.py",
            "dir5/e.py",
            "dir6/f.py",
        ],
    )

    summary = result.get_ignored_summary(max_items=3)
    assert len(summary) == 4  # 3 items + "... and X more patterns"
    assert "... and 3 more patterns" in summary


def test_gitignore_filter_result_get_ignored_summary_empty():
    """Test GitignoreFilterResult.get_ignored_summary with no ignored files."""
    result = GitignoreFilterResult(kept=["src/main.py"], ignored=[])
    summary = result.get_ignored_summary()
    assert summary == []


def test_gitignore_filter_result_get_ignored_summary_invalid_max_items():
    """Test GitignoreFilterResult.get_ignored_summary with invalid max_items."""
    result = GitignoreFilterResult(
        kept=["src/main.py"],
        ignored=["node_modules/a.js", ".venv/lib/x.py"],
    )
    # max_items <= 0 should return empty list
    assert result.get_ignored_summary(max_items=0) == []
    assert result.get_ignored_summary(max_items=-1) == []


@pytest.fixture
def mock_file_operator(tmp_path):
    """Create a mock file operator for testing."""
    from pai_agent_sdk.environment.local import LocalFileOperator

    return LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])


async def test_filter_gitignored_no_gitignore(mock_file_operator):
    """Test filter_gitignored when no .gitignore exists."""
    files = ["src/main.py", "test.py", ".venv/lib/site.py"]

    result = await filter_gitignored(files, mock_file_operator)

    # All files should be returned when no .gitignore
    assert result.kept == files
    assert result.ignored == []


async def test_filter_gitignored_with_gitignore(mock_file_operator, tmp_path):
    """Test filter_gitignored with .gitignore file."""
    # Create .gitignore
    gitignore_path = tmp_path / ".gitignore"
    gitignore_path.write_text(".venv/\n__pycache__/\n*.pyc\n")

    files = [
        "src/main.py",
        "test.py",
        ".venv/lib/site.py",
        "__pycache__/cache.pyc",
        "module.pyc",
    ]

    result = await filter_gitignored(files, mock_file_operator)

    # Only non-ignored files should be in kept
    assert "src/main.py" in result.kept
    assert "test.py" in result.kept
    assert ".venv/lib/site.py" not in result.kept
    assert "__pycache__/cache.pyc" not in result.kept
    assert "module.pyc" not in result.kept

    # Ignored files should be in ignored list
    assert ".venv/lib/site.py" in result.ignored
    assert "__pycache__/cache.pyc" in result.ignored
    assert "module.pyc" in result.ignored


async def test_filter_gitignored_gitignore_changes(mock_file_operator, tmp_path):
    """Test that filter_gitignored picks up .gitignore changes (no caching)."""
    gitignore_path = tmp_path / ".gitignore"

    files = ["src/main.py", ".venv/lib/site.py", "node_modules/pkg/index.js"]

    # First: ignore .venv
    gitignore_path.write_text(".venv/\n")
    result1 = await filter_gitignored(files, mock_file_operator)
    assert ".venv/lib/site.py" not in result1.kept
    assert "node_modules/pkg/index.js" in result1.kept

    # Second: change to ignore node_modules instead
    gitignore_path.write_text("node_modules/\n")
    result2 = await filter_gitignored(files, mock_file_operator)
    assert ".venv/lib/site.py" in result2.kept
    assert "node_modules/pkg/index.js" not in result2.kept


async def test_filter_gitignored_empty_gitignore(mock_file_operator, tmp_path):
    """Test filter_gitignored with empty .gitignore."""
    gitignore_path = tmp_path / ".gitignore"
    gitignore_path.write_text("")

    files = ["src/main.py", ".venv/lib/site.py"]

    result = await filter_gitignored(files, mock_file_operator)

    # All files should be returned with empty .gitignore
    assert result.kept == files
    assert result.ignored == []


async def test_filter_gitignored_preserves_order(mock_file_operator, tmp_path):
    """Test that filter_gitignored preserves file order."""
    gitignore_path = tmp_path / ".gitignore"
    gitignore_path.write_text("*.pyc\n")

    files = ["z.py", "a.py", "m.py", "b.pyc"]

    result = await filter_gitignored(files, mock_file_operator)

    # Order should be preserved
    assert result.kept == ["z.py", "a.py", "m.py"]


async def test_filter_gitignored_returns_ignored_summary(mock_file_operator, tmp_path):
    """Test that filter_gitignored result can generate summary."""
    gitignore_path = tmp_path / ".gitignore"
    gitignore_path.write_text("node_modules/\n.venv/\n")

    files = [
        "src/main.py",
        "node_modules/a.js",
        "node_modules/b.js",
        ".venv/lib/site.py",
    ]

    result = await filter_gitignored(files, mock_file_operator)

    summary = result.get_ignored_summary()
    assert "node_modules/ (2 files)" in summary
    assert ".venv/ (1 file)" in summary
