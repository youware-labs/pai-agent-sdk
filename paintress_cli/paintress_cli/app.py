"""TUI Application for paintress-cli.

This module provides the main TUI application entry point.
Orchestrates environment, context, agent runtime and UI components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from paintress_cli.config import ConfigManager, PaintressConfig
from paintress_cli.logging import get_logger

if TYPE_CHECKING:
    from pai_agent_sdk.agents.main import AgentRuntime
    from paintress_cli.environment import TUIEnvironment
    from paintress_cli.session import TUIContext

logger = get_logger(__name__)


class TUIMode(str, Enum):
    """Agent operating mode."""

    ACT = "act"
    PLAN = "plan"


class TUIState(str, Enum):
    """TUI application state."""

    IDLE = "idle"
    RUNNING = "running"


@dataclass
class TUIApplication:
    """Main TUI application class."""

    config: PaintressConfig
    config_manager: ConfigManager

    # Runtime state
    _mode: TUIMode = field(default=TUIMode.ACT, init=False)
    _state: TUIState = field(default=TUIState.IDLE, init=False)
    _env: TUIEnvironment | None = field(default=None, init=False)
    _ctx: TUIContext | None = field(default=None, init=False)
    _runtime: AgentRuntime | None = field(default=None, init=False)

    async def run(self) -> None:
        """Run the TUI application.

        Initialization flow:
        1. Create TUIEnvironment
        2. Create TUIContext
        3. Create AgentRuntime
        4. Start UI loop
        """
        # TODO: Implement full run loop
        logger.info("TUIApplication.run() called")
        logger.info("Config model: %s", self.config.general.model)

        # Placeholder - just print for now
        print(f"Paintress CLI initialized with model: {self.config.general.model}")
        print("TUI not yet implemented. Exiting.")
