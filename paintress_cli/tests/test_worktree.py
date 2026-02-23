"""Tests for paintress_cli --worktree functionality."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from paintress_cli.cli import _create_worktree, _get_git_root, _project_hash, cli

# =============================================================================
# _get_git_root Tests
# =============================================================================


def test_get_git_root_in_repo(tmp_path: Path) -> None:
    """Test _get_git_root returns root path inside a git repo."""
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, check=True)

    # Get expected root
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(tmp_path),
    )
    expected = Path(result.stdout.strip())

    # Capture real subprocess.run before patching
    real_run = subprocess.run

    def patched_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("cwd", str(tmp_path))
        return real_run(*args, **kwargs)

    with patch("paintress_cli.cli.subprocess.run", side_effect=patched_run):
        root = _get_git_root()

    assert root is not None
    assert root == expected


def test_get_git_root_not_in_repo(tmp_path: Path) -> None:
    """Test _get_git_root returns None when not in a git repo."""
    # Use a directory that's definitely not a git repo
    with patch(
        "paintress_cli.cli.subprocess.run",
        side_effect=subprocess.CalledProcessError(128, "git"),
    ):
        root = _get_git_root()

    assert root is None


def test_get_git_root_git_not_installed() -> None:
    """Test _get_git_root returns None when git is not installed."""
    with patch("paintress_cli.cli.subprocess.run", side_effect=FileNotFoundError):
        root = _get_git_root()

    assert root is None


# =============================================================================
# _create_worktree Tests
# =============================================================================


def test_create_worktree_auto_branch(tmp_path: Path) -> None:
    """Test _create_worktree with auto-generated branch name."""
    # Create a git repo with an initial commit
    repo = tmp_path / "repo"
    repo.mkdir()
    config_dir = tmp_path / "config"
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), capture_output=True, check=True)
    (repo / "README.md").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True, check=True)

    with (
        patch("paintress_cli.cli._get_git_root", return_value=repo),
        patch("paintress_cli.cli.ConfigManager.DEFAULT_CONFIG_DIR", config_dir),
    ):
        worktree_dir, branch_name, is_resume = _create_worktree(None)

    assert worktree_dir.exists()
    assert not is_resume
    assert branch_name.startswith("paintress/")
    assert worktree_dir.parent == config_dir / "worktrees" / _project_hash(repo)

    # Verify it's a valid git worktree
    result = subprocess.run(
        ["git", "worktree", "list"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    assert str(worktree_dir) in result.stdout

    # Cleanup
    subprocess.run(["git", "worktree", "remove", str(worktree_dir)], cwd=str(repo), capture_output=True, check=True)


def test_create_worktree_custom_branch(tmp_path: Path) -> None:
    """Test _create_worktree with a custom branch name."""
    repo = tmp_path / "repo"
    repo.mkdir()
    config_dir = tmp_path / "config"
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), capture_output=True, check=True)
    (repo / "README.md").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True, check=True)

    with (
        patch("paintress_cli.cli._get_git_root", return_value=repo),
        patch("paintress_cli.cli.ConfigManager.DEFAULT_CONFIG_DIR", config_dir),
    ):
        worktree_dir, branch_name, is_resume = _create_worktree("my-feature")

    assert worktree_dir.exists()
    assert not is_resume
    assert branch_name == "my-feature"
    assert worktree_dir.name == "my-feature"

    # Verify metadata file was created with correct structure
    metadata_file = worktree_dir.parent / "metadata.json"
    assert metadata_file.exists()
    metadata = json.loads(metadata_file.read_text())
    assert metadata["git_root"] == str(repo.resolve())
    assert "created_at" in metadata

    # Cleanup
    subprocess.run(["git", "worktree", "remove", str(worktree_dir)], cwd=str(repo), capture_output=True, check=True)


def test_create_worktree_not_in_git_repo() -> None:
    """Test _create_worktree raises error when not in a git repo."""
    import click

    with patch("paintress_cli.cli._get_git_root", return_value=None):
        with pytest.raises(click.ClickException, match="requires a git repository"):
            _create_worktree(None)


def test_create_worktree_resume_existing(tmp_path: Path) -> None:
    """Test _create_worktree resumes when worktree dir already exists."""
    repo = tmp_path / "repo"
    repo.mkdir()
    config_dir = tmp_path / "config"

    # Pre-create the worktree directory (simulating a previous run)
    proj_hash = _project_hash(repo)
    worktrees_base = config_dir / "worktrees" / proj_hash
    worktrees_base.mkdir(parents=True)
    existing = worktrees_base / "my-branch"
    existing.mkdir()

    with (
        patch("paintress_cli.cli._get_git_root", return_value=repo),
        patch("paintress_cli.cli.ConfigManager.DEFAULT_CONFIG_DIR", config_dir),
    ):
        worktree_dir, branch_name, is_resume = _create_worktree("my-branch")

    assert worktree_dir == existing
    assert branch_name == "my-branch"
    assert is_resume


def test_create_worktree_git_failure(tmp_path: Path) -> None:
    """Test _create_worktree raises error when git command fails."""
    import click

    repo = tmp_path / "repo"
    repo.mkdir()

    error = subprocess.CalledProcessError(128, "git", stderr="fatal: invalid reference: HEAD")

    with (
        patch("paintress_cli.cli._get_git_root", return_value=repo),
        patch("paintress_cli.cli.subprocess.run", side_effect=error),
    ):
        with pytest.raises(click.ClickException, match="Failed to create git worktree"):
            _create_worktree("test-branch")


# =============================================================================
# CLI --worktree Option Tests
# =============================================================================


def test_cli_worktree_option_exists() -> None:
    """Test that --worktree and --branch options are recognized by the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "--worktree" in result.output
    assert "--branch" in result.output


def test_cli_worktree_not_in_repo(tmp_path: Path) -> None:
    """Test CLI exits with error when --worktree used outside git repo."""
    runner = CliRunner()
    with (
        patch("paintress_cli.cli._get_git_root", return_value=None),
        patch("paintress_cli.cli.ConfigManager.load", return_value=MagicMock(is_configured=True, env={})),
        patch("paintress_cli.cli.ensure_builtin_assets"),
        patch("paintress_cli.cli.load_env_from_config"),
    ):
        result = runner.invoke(cli, ["--worktree"])
    assert result.exit_code != 0
    assert "requires a git repository" in result.output


def test_cli_branch_implies_worktree(tmp_path: Path) -> None:
    """Test --branch implies --worktree."""
    runner = CliRunner()
    with (
        patch("paintress_cli.cli._get_git_root", return_value=None),
        patch("paintress_cli.cli.ConfigManager.load", return_value=MagicMock(is_configured=True, env={})),
        patch("paintress_cli.cli.ensure_builtin_assets"),
        patch("paintress_cli.cli.load_env_from_config"),
    ):
        result = runner.invoke(cli, ["--branch", "my-feature"])
    assert result.exit_code != 0
    assert "requires a git repository" in result.output
