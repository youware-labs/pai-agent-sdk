"""Tests for gitignore filtering functionality."""

import pytest

from pai_agent_sdk.toolsets.core.filesystem._gitignore import (
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
    assert result == files


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

    # Only non-ignored files should be returned
    assert "src/main.py" in result
    assert "test.py" in result
    assert ".venv/lib/site.py" not in result
    assert "__pycache__/cache.pyc" not in result
    assert "module.pyc" not in result


async def test_filter_gitignored_gitignore_changes(mock_file_operator, tmp_path):
    """Test that filter_gitignored picks up .gitignore changes (no caching)."""
    gitignore_path = tmp_path / ".gitignore"

    files = ["src/main.py", ".venv/lib/site.py", "node_modules/pkg/index.js"]

    # First: ignore .venv
    gitignore_path.write_text(".venv/\n")
    result1 = await filter_gitignored(files, mock_file_operator)
    assert ".venv/lib/site.py" not in result1
    assert "node_modules/pkg/index.js" in result1

    # Second: change to ignore node_modules instead
    gitignore_path.write_text("node_modules/\n")
    result2 = await filter_gitignored(files, mock_file_operator)
    assert ".venv/lib/site.py" in result2
    assert "node_modules/pkg/index.js" not in result2


async def test_filter_gitignored_empty_gitignore(mock_file_operator, tmp_path):
    """Test filter_gitignored with empty .gitignore."""
    gitignore_path = tmp_path / ".gitignore"
    gitignore_path.write_text("")

    files = ["src/main.py", ".venv/lib/site.py"]

    result = await filter_gitignored(files, mock_file_operator)

    # All files should be returned with empty .gitignore
    assert result == files


async def test_filter_gitignored_preserves_order(mock_file_operator, tmp_path):
    """Test that filter_gitignored preserves file order."""
    gitignore_path = tmp_path / ".gitignore"
    gitignore_path.write_text("*.pyc\n")

    files = ["z.py", "a.py", "m.py", "b.pyc"]

    result = await filter_gitignored(files, mock_file_operator)

    # Order should be preserved
    assert result == ["z.py", "a.py", "m.py"]
