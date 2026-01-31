"""TUI application module.

This module provides the core TUI application components:
- TUIApp: Main TUI application class
- TUIMode: Operating mode (ACT/PLAN)
- TUIState: Application state (IDLE/RUNNING)
- TUIPhase: Execution phase (for state machine)
- TUIStateMachine: State management
- CommandRegistry: Slash command handling
"""

from __future__ import annotations

from paintress_cli.app.commands import (
    BUILTIN_COMMANDS,
    Command,
    CommandContext,
    CommandRegistry,
    create_default_registry,
)
from paintress_cli.app.state import (
    VALID_TRANSITIONS,
    TUIMode,
    TUIPhase,
    TUIStateMachine,
)
from paintress_cli.app.tui import TUIApp, TUIState

__all__ = [
    "BUILTIN_COMMANDS",
    "VALID_TRANSITIONS",
    "Command",
    "CommandContext",
    "CommandRegistry",
    "TUIApp",
    "TUIMode",
    "TUIPhase",
    "TUIState",
    "TUIStateMachine",
    "create_default_registry",
]
