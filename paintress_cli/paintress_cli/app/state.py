"""TUI state management.

Provides explicit state machine for TUI application state.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto


class TUIMode(str, Enum):
    """Agent operating mode."""

    ACT = "act"
    PLAN = "plan"


class TUIPhase(Enum):
    """TUI execution phase.

    Represents the current phase of agent execution:
    - IDLE: Waiting for user input
    - THINKING: Model is generating response
    - TOOL_CALLING: Executing tool calls
    - AWAITING_APPROVAL: Waiting for HITL approval
    - STREAMING_OUTPUT: Streaming text output
    """

    IDLE = auto()
    THINKING = auto()
    TOOL_CALLING = auto()
    AWAITING_APPROVAL = auto()
    STREAMING_OUTPUT = auto()


# Valid state transitions
VALID_TRANSITIONS: dict[TUIPhase, set[TUIPhase]] = {
    TUIPhase.IDLE: {TUIPhase.THINKING},
    TUIPhase.THINKING: {TUIPhase.TOOL_CALLING, TUIPhase.STREAMING_OUTPUT, TUIPhase.IDLE},
    TUIPhase.TOOL_CALLING: {TUIPhase.AWAITING_APPROVAL, TUIPhase.THINKING, TUIPhase.IDLE},
    TUIPhase.AWAITING_APPROVAL: {TUIPhase.TOOL_CALLING, TUIPhase.IDLE},
    TUIPhase.STREAMING_OUTPUT: {TUIPhase.THINKING, TUIPhase.TOOL_CALLING, TUIPhase.IDLE},
}


class TUIStateMachine:
    """Explicit state machine for TUI application.

    Manages state transitions and notifies observers when state changes.
    Invalid transitions are logged but not blocked to maintain robustness.
    """

    def __init__(self, initial_mode: TUIMode = TUIMode.ACT) -> None:
        """Initialize state machine.

        Args:
            initial_mode: Initial operating mode.
        """
        self._mode = initial_mode
        self._phase = TUIPhase.IDLE
        self._observers: list[Callable[[TUIPhase, TUIPhase], None]] = []
        self._mode_observers: list[Callable[[TUIMode, TUIMode], None]] = []

    @property
    def mode(self) -> TUIMode:
        """Get current operating mode."""
        return self._mode

    @property
    def phase(self) -> TUIPhase:
        """Get current execution phase."""
        return self._phase

    @property
    def is_idle(self) -> bool:
        """Check if in idle state."""
        return self._phase == TUIPhase.IDLE

    @property
    def is_running(self) -> bool:
        """Check if agent is running (not idle)."""
        return self._phase != TUIPhase.IDLE

    @property
    def is_awaiting_approval(self) -> bool:
        """Check if waiting for HITL approval."""
        return self._phase == TUIPhase.AWAITING_APPROVAL

    def add_observer(self, callback: Callable[[TUIPhase, TUIPhase], None]) -> None:
        """Add phase change observer.

        Args:
            callback: Function called with (old_phase, new_phase).
        """
        self._observers.append(callback)

    def add_mode_observer(self, callback: Callable[[TUIMode, TUIMode], None]) -> None:
        """Add mode change observer.

        Args:
            callback: Function called with (old_mode, new_mode).
        """
        self._mode_observers.append(callback)

    def remove_observer(self, callback: Callable[[TUIPhase, TUIPhase], None]) -> None:
        """Remove phase change observer."""
        if callback in self._observers:
            self._observers.remove(callback)

    def transition(self, new_phase: TUIPhase) -> bool:
        """Transition to a new phase.

        Args:
            new_phase: Target phase.

        Returns:
            True if transition was valid, False otherwise.
        """
        if self._phase == new_phase:
            return True

        is_valid = self._is_valid_transition(new_phase)
        old_phase = self._phase
        self._phase = new_phase

        # Notify observers
        for observer in self._observers:
            observer(old_phase, new_phase)

        return is_valid

    def switch_mode(self, new_mode: TUIMode) -> bool:
        """Switch operating mode.

        Args:
            new_mode: Target mode.

        Returns:
            True if switch was allowed (only when idle), False otherwise.
        """
        if not self.is_idle:
            return False

        if self._mode == new_mode:
            return True

        old_mode = self._mode
        self._mode = new_mode

        # Notify observers
        for observer in self._mode_observers:
            observer(old_mode, new_mode)

        return True

    def reset(self) -> None:
        """Reset to idle state."""
        if self._phase != TUIPhase.IDLE:
            self.transition(TUIPhase.IDLE)

    def _is_valid_transition(self, new_phase: TUIPhase) -> bool:
        """Check if transition is valid.

        Args:
            new_phase: Target phase.

        Returns:
            True if transition is allowed.
        """
        valid_targets = VALID_TRANSITIONS.get(self._phase, set())
        return new_phase in valid_targets

    # Convenience methods for common transitions

    def start_thinking(self) -> bool:
        """Transition to thinking phase."""
        return self.transition(TUIPhase.THINKING)

    def start_tools(self) -> bool:
        """Transition to tool calling phase."""
        return self.transition(TUIPhase.TOOL_CALLING)

    def start_approval(self) -> bool:
        """Transition to awaiting approval phase."""
        return self.transition(TUIPhase.AWAITING_APPROVAL)

    def start_streaming(self) -> bool:
        """Transition to streaming output phase."""
        return self.transition(TUIPhase.STREAMING_OUTPUT)

    def finish(self) -> bool:
        """Transition back to idle."""
        return self.transition(TUIPhase.IDLE)

    def get_status_text(self) -> str:
        """Get human-readable status text for current phase."""
        status_map = {
            TUIPhase.IDLE: "Idle",
            TUIPhase.THINKING: "Thinking...",
            TUIPhase.TOOL_CALLING: "Running tools...",
            TUIPhase.AWAITING_APPROVAL: "Awaiting approval...",
            TUIPhase.STREAMING_OUTPUT: "Generating...",
        }
        return status_map.get(self._phase, "Unknown")
