"""Tests for session management (save/load/prune/list)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

from paintress_cli.config import ConfigManager

# =============================================================================
# ConfigManager.get_sessions_dir Tests
# =============================================================================


def test_get_sessions_dir(tmp_path: Path) -> None:
    """Test get_sessions_dir returns correct path."""
    config_dir = tmp_path / ".paintress_cli"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text('model = "anthropic:test"\n')

    mgr = ConfigManager(config_dir=config_dir, project_dir=tmp_path)
    sessions_dir = mgr.get_sessions_dir()

    assert sessions_dir == config_dir / "sessions"


# =============================================================================
# Session ID Tests
# =============================================================================


def test_session_id_is_12_char_hex() -> None:
    """Test that session_id is a 12-character hex string."""
    from paintress_cli.app.tui import TUIApp

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = MagicMock(exists=lambda: False)

    app = TUIApp(config=config, config_manager=config_manager)

    assert len(app.session_id) == 12
    assert re.match(r"^[0-9a-f]{12}$", app.session_id)


def test_session_id_unique_per_instance() -> None:
    """Test that each TUIApp instance gets a unique session_id."""
    from paintress_cli.app.tui import TUIApp

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = MagicMock(exists=lambda: False)

    app1 = TUIApp(config=config, config_manager=config_manager)
    app2 = TUIApp(config=config, config_manager=config_manager)

    assert app1.session_id != app2.session_id


# =============================================================================
# Prune Sessions Tests
# =============================================================================


def test_prune_sessions_no_op_under_limit(tmp_path: Path) -> None:
    """Test prune does nothing when under limit."""
    from paintress_cli.app.tui import TUIApp

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Create 3 sessions
    for i in range(3):
        d = sessions_dir / f"session{i:04d}"
        d.mkdir()
        (d / "metadata.json").write_text(
            json.dumps({
                "session_id": f"session{i:04d}",
                "updated_at": f"2026-01-{i + 1:02d}T00:00:00+00:00",
            })
        )

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = sessions_dir

    app = TUIApp(config=config, config_manager=config_manager)
    app._prune_sessions(sessions_dir, max_sessions=5)

    assert len(list(sessions_dir.iterdir())) == 3


def test_prune_sessions_removes_oldest(tmp_path: Path) -> None:
    """Test prune removes oldest sessions when over limit."""
    from paintress_cli.app.tui import TUIApp

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Create 5 sessions with different timestamps
    for i in range(5):
        d = sessions_dir / f"session{i:04d}"
        d.mkdir()
        (d / "metadata.json").write_text(
            json.dumps({
                "session_id": f"session{i:04d}",
                "updated_at": f"2026-01-{i + 1:02d}T00:00:00+00:00",
            })
        )

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = sessions_dir

    app = TUIApp(config=config, config_manager=config_manager)
    app._prune_sessions(sessions_dir, max_sessions=3)

    remaining = sorted(d.name for d in sessions_dir.iterdir())
    assert len(remaining) == 3
    # Oldest (session0000, session0001) should be removed
    assert remaining == ["session0002", "session0003", "session0004"]


def test_prune_sessions_handles_missing_metadata(tmp_path: Path) -> None:
    """Test prune handles sessions without metadata.json."""
    from paintress_cli.app.tui import TUIApp

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Create sessions - some with metadata, some without
    for i in range(4):
        d = sessions_dir / f"session{i:04d}"
        d.mkdir()
        if i >= 2:  # Only newer sessions have metadata
            (d / "metadata.json").write_text(
                json.dumps({
                    "session_id": f"session{i:04d}",
                    "updated_at": f"2026-02-{i + 1:02d}T00:00:00+00:00",
                })
            )

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = sessions_dir

    app = TUIApp(config=config, config_manager=config_manager)
    app._prune_sessions(sessions_dir, max_sessions=2)

    remaining = sorted(d.name for d in sessions_dir.iterdir())
    assert len(remaining) == 2


def test_prune_sessions_nonexistent_dir(tmp_path: Path) -> None:
    """Test prune handles non-existent sessions directory."""
    from paintress_cli.app.tui import TUIApp

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()

    app = TUIApp(config=config, config_manager=config_manager)
    # Should not raise
    app._prune_sessions(tmp_path / "nonexistent")


# =============================================================================
# Load Session Tests
# =============================================================================


def test_load_session_exact_match(tmp_path: Path) -> None:
    """Test loading session with exact ID match."""
    from paintress_cli.app.tui import TUIApp

    sessions_dir = tmp_path / "sessions"
    session_dir = sessions_dir / "abc123def456"
    session_dir.mkdir(parents=True)
    (session_dir / "message_history.json").write_text("[]")
    (session_dir / "context_state.json").write_text("{}")

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = sessions_dir

    app = TUIApp(config=config, config_manager=config_manager)
    original_id = app.session_id

    with patch.object(app, "_load_history") as mock_load:
        app._load_session("abc123def456")

    mock_load.assert_called_once_with(str(session_dir))
    assert app.session_id == "abc123def456"
    assert app.session_id != original_id


def test_load_session_prefix_match(tmp_path: Path) -> None:
    """Test loading session with prefix match."""
    from paintress_cli.app.tui import TUIApp

    sessions_dir = tmp_path / "sessions"
    session_dir = sessions_dir / "abc123def456"
    session_dir.mkdir(parents=True)
    (session_dir / "message_history.json").write_text("[]")

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = sessions_dir

    app = TUIApp(config=config, config_manager=config_manager)

    with patch.object(app, "_load_history") as mock_load:
        app._load_session("abc1")

    mock_load.assert_called_once_with(str(session_dir))
    assert app.session_id == "abc123def456"


def test_load_session_ambiguous(tmp_path: Path) -> None:
    """Test loading session with ambiguous prefix."""
    from paintress_cli.app.tui import TUIApp

    sessions_dir = tmp_path / "sessions"
    (sessions_dir / "abc123aaaaaa").mkdir(parents=True)
    (sessions_dir / "abc123bbbbbb").mkdir(parents=True)

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = sessions_dir

    app = TUIApp(config=config, config_manager=config_manager)

    with patch.object(app, "_append_system_output") as mock_output:
        app._load_session("abc")

    # Should mention "Ambiguous"
    calls = [str(c) for c in mock_output.call_args_list]
    assert any("Ambiguous" in c for c in calls)


def test_load_session_not_found(tmp_path: Path) -> None:
    """Test loading session that doesn't exist."""
    from paintress_cli.app.tui import TUIApp

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = sessions_dir

    app = TUIApp(config=config, config_manager=config_manager)

    with patch.object(app, "_append_system_output") as mock_output:
        app._load_session("nonexistent")

    calls = [str(c) for c in mock_output.call_args_list]
    assert any("not found" in c for c in calls)
