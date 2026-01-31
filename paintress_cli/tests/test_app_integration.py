"""Integration tests for TUIApp.

Tests core logic that requires mocking the TUI environment.
Focus on testable components and state transitions.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

# Import the components we're testing
from paintress_cli.app import TUIApp, TUIMode, TUIState


@dataclass
class MockConfig:
    """Minimal mock config for testing."""

    general: Any = field(
        default_factory=lambda: MagicMock(
            max_requests=10,
            mode="act",
        )
    )
    display: Any = field(
        default_factory=lambda: MagicMock(
            max_lines=500,
            mouse=True,
        )
    )
    browser: Any = field(
        default_factory=lambda: MagicMock(
            mode="disabled",
            url=None,
        )
    )

    def get_commands(self) -> dict:
        return {}


@dataclass
class MockConfigManager:
    """Minimal mock config manager for testing."""

    global_config_dir: Any = field(default_factory=lambda: MagicMock())
    project_config_dir: Any = field(default_factory=lambda: MagicMock())

    def get_auto_save_dir(self) -> Any:
        return MagicMock(exists=lambda: False)

    def get_mcp_config(self) -> None:
        return None

    def load_custom_commands(self) -> dict:
        return {}


# =============================================================================
# TUIMode/TUIState Tests
# =============================================================================


def test_tui_mode_values():
    """Test TUIMode enum values."""
    assert TUIMode.ACT.value == "act"
    assert TUIMode.PLAN.value == "plan"


def test_tui_state_values():
    """Test TUIState enum values."""
    assert TUIState.IDLE.value == "idle"
    assert TUIState.RUNNING.value == "running"


def test_tui_mode_is_string():
    """Test that TUIMode values can be used as strings."""
    # TUIMode inherits from str, so .value gives the string
    assert TUIMode.ACT.value == "act"
    assert TUIMode.PLAN.value == "plan"
    # Can compare with string due to str inheritance
    assert TUIMode.ACT == "act"
    assert TUIMode.PLAN == "plan"


# =============================================================================
# TUIApp Initialization Tests
# =============================================================================


def test_tui_app_initial_state():
    """Test TUIApp initial state."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Check initial state
    assert app.mode == TUIMode.ACT
    assert app.state == TUIState.IDLE
    assert app._agent_phase == "idle"


def test_tui_app_mode_switching():
    """Test mode switching."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Initial mode
    assert app.mode == TUIMode.ACT

    # Switch to PLAN
    app.switch_mode(TUIMode.PLAN)
    assert app.mode == TUIMode.PLAN

    # Switch back to ACT
    app.switch_mode(TUIMode.ACT)
    assert app.mode == TUIMode.ACT


def test_tui_app_mode_switch_no_change_when_same():
    """Test mode switch does nothing when already in that mode."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Already in ACT mode
    app.switch_mode(TUIMode.ACT)
    assert app.mode == TUIMode.ACT


# =============================================================================
# Output Management Tests
# =============================================================================


def test_tui_app_output_cache_invalidation():
    """Test output cache is invalidated properly."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Initial state - cache is valid (empty)
    assert app._output_cache_valid is True

    # Invalidate cache
    app._invalidate_output_cache()
    assert app._output_cache_valid is False


def test_tui_app_append_output():
    """Test appending output lines."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Append some lines
    app._append_output("Line 1")
    app._append_output("Line 2")

    assert len(app._output_lines) == 2
    assert app._output_lines[0] == "Line 1"
    assert app._output_lines[1] == "Line 2"
    assert app._output_cache_valid is False


def test_tui_app_output_line_limit():
    """Test output line trimming at max limit."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    app._max_output_lines = 10  # Set low limit for testing

    # Add more lines than limit
    for i in range(15):
        app._append_output(f"Line {i}")

    # Should be trimmed to max_output_lines
    assert len(app._output_lines) == 10
    # Oldest lines should be removed
    assert app._output_lines[0] == "Line 5"
    assert app._output_lines[-1] == "Line 14"


# =============================================================================
# Streaming Text Tests
# =============================================================================


def test_tui_app_streaming_text_lifecycle():
    """Test streaming text start/update/finalize."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    # Mock the prompt_toolkit app with proper output size
    mock_output = MagicMock()
    mock_output.get_size.return_value = MagicMock(columns=80, rows=24)
    app._app = MagicMock(output=mock_output)

    # Start streaming
    app._start_streaming_text("Hello")
    assert app._streaming_text == "Hello"
    assert app._streaming_line_index == 0
    assert len(app._output_lines) == 1

    # Update streaming - this renders markdown so needs proper width
    app._update_streaming_text(" World")
    assert app._streaming_text == "Hello World"

    # Finalize
    app._finalize_streaming_text()
    assert app._streaming_text == ""
    assert app._streaming_line_index is None


def test_tui_app_streaming_thinking_lifecycle():
    """Test streaming thinking start/update/finalize."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    # Mock the prompt_toolkit app with proper output size
    mock_output = MagicMock()
    mock_output.get_size.return_value = MagicMock(columns=80, rows=24)
    app._app = MagicMock(output=mock_output)

    # Start streaming thinking
    app._start_streaming_thinking("Thinking...")
    assert app._streaming_thinking == "Thinking..."
    assert app._streaming_thinking_line_index == 0

    # Update
    app._update_streaming_thinking(" more")
    assert app._streaming_thinking == "Thinking... more"

    # Finalize
    app._finalize_streaming_thinking()
    assert app._streaming_thinking == ""
    assert app._streaming_thinking_line_index is None


# =============================================================================
# HITL State Tests
# =============================================================================


def test_tui_app_hitl_initial_state():
    """Test HITL initial state."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    assert app._hitl_pending is False
    assert app._approval_event is None
    assert app._approval_result is None
    assert len(app._pending_approvals) == 0
    assert app._current_approval_index == 0


def test_tui_app_hitl_reset():
    """Test HITL state reset."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Set some HITL state
    app._hitl_pending = True
    app._pending_approvals = [MagicMock(), MagicMock()]
    app._current_approval_index = 1
    app._approval_result = True
    # Don't set _approval_event for this test

    # Reset
    app._reset_hitl_state()

    assert app._hitl_pending is False
    assert len(app._pending_approvals) == 0
    assert app._current_approval_index == 0
    # When no event exists, result remains unchanged after reset


def test_tui_app_hitl_reset_with_event():
    """Test HITL state reset when event exists."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Set HITL state with an event
    app._hitl_pending = True
    app._approval_event = asyncio.Event()
    app._approval_result = True

    # Reset should set result to False and set the event
    app._reset_hitl_state()

    assert app._hitl_pending is False
    assert app._approval_result is False
    assert app._approval_reason == "Cancelled"
    assert app._approval_event is None  # Cleared after reset


# =============================================================================
# Steering Message Tests
# =============================================================================


def test_tui_app_steering_add():
    """Test adding steering messages."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    app._app = MagicMock()
    app._state = TUIState.RUNNING  # Only add when running

    app._add_steering_message("Do this instead")

    assert len(app._steering_items) == 1
    _, text, status = app._steering_items[0]
    assert text == "Do this instead"
    assert status == "pending"


def test_tui_app_steering_ack():
    """Test acknowledging steering messages."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    app._app = MagicMock()
    app._state = TUIState.RUNNING

    # Add a message
    app._add_steering_message("Do this")

    # Acknowledge it - content_preview must contain the original text
    app._ack_steering_by_content("Please Do this instead")

    _, _, status = app._steering_items[0]
    assert status == "acked"


# =============================================================================
# Subagent State Tests
# =============================================================================


def test_tui_app_subagent_state_tracking():
    """Test subagent state tracking."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Initially empty
    assert len(app._subagent_states) == 0

    # Add subagent state
    app._subagent_states["sub-1"] = {
        "line_index": 0,
        "tool_names": ["search", "view"],
    }

    assert "sub-1" in app._subagent_states
    assert app._subagent_states["sub-1"]["tool_names"] == ["search", "view"]


# =============================================================================
# Tool Message Tests
# =============================================================================


def test_tui_app_tool_message_tracking():
    """Test tool message tracking."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Initially empty
    assert len(app._tool_messages) == 0
    assert len(app._printed_tool_calls) == 0


# =============================================================================
# History Tests
# =============================================================================


def test_tui_app_prompt_history():
    """Test prompt history tracking."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Initially empty
    assert len(app._prompt_history) == 0
    assert app._history_index == -1

    # Add to history
    app._prompt_history.append("First prompt")
    app._prompt_history.append("Second prompt")

    assert len(app._prompt_history) == 2
    assert app._prompt_history[0] == "First prompt"


# =============================================================================
# Session Usage Tests
# =============================================================================


def test_tui_app_session_usage_tracking():
    """Test session usage tracking."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Initial state
    assert app._session_usage.is_empty()


def test_tui_app_context_token_tracking():
    """Test context token tracking."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Initial state
    assert app._current_context_tokens == 0
    assert app._context_window_size == 200000

    # Update tokens
    app._current_context_tokens = 5000
    assert app._current_context_tokens == 5000


# =============================================================================
# UI State Tests
# =============================================================================


def test_tui_app_input_mode():
    """Test input mode tracking."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Default mode
    assert app._input_mode == "send"


def test_tui_app_mouse_enabled():
    """Test mouse mode tracking."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Default enabled
    assert app._mouse_enabled is True


def test_tui_app_ctrl_c_handling():
    """Test double Ctrl+C exit tracking."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    assert app._last_ctrl_c_time == 0.0
    assert app._ctrl_c_exit_timeout == 2.0
