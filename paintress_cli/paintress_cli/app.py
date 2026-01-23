"""TUI Application for paintress-cli.

This module provides the main TUI application with:
- prompt_toolkit based UI with dual-pane layout
- Agent execution with streaming output
- Steering message injection during execution
- Mode switching (ACT/PLAN) via /act and /plan slash commands
- Scrollable output with keyboard and mouse support
- Input mode switching (send/edit) with Tab key
- Double Ctrl+C exit confirmation

Example:
    from paintress_cli.app import TUIApp

    async with TUIApp(config, config_manager) as app:
        await app.run()

"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

from prompt_toolkit import Application
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, ScrollablePane, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea
from pydantic_ai import DeferredToolRequests, DeferredToolResults, ToolDenied, UsageLimits
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessagesTypeAdapter,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
)
from pydantic_ai.models import Model
from rich.text import Text

from pai_agent_sdk.agents.main import AgentRuntime, stream_agent
from pai_agent_sdk.context import ResumableState, StreamEvent
from pai_agent_sdk.events import (
    CompactCompleteEvent,
    CompactFailedEvent,
    CompactStartEvent,
    HandoffCompleteEvent,
    HandoffFailedEvent,
    HandoffStartEvent,
    SubagentCompleteEvent,
    SubagentStartEvent,
)
from pai_agent_sdk.utils import get_latest_request_usage
from paintress_cli.browser import BrowserManager
from paintress_cli.config import ConfigManager, PaintressConfig
from paintress_cli.display import EventRenderer, RichRenderer, ToolMessage
from paintress_cli.events import (
    AgentPhaseEvent,
    ContextUpdateEvent,
    SteeringInjectedEvent,
)
from paintress_cli.hooks import emit_context_update
from paintress_cli.logging import configure_tui_logging, get_logger
from paintress_cli.runtime import create_tui_runtime
from paintress_cli.session import TUIContext
from paintress_cli.usage import SessionUsage

if TYPE_CHECKING:
    from prompt_toolkit.key_binding import KeyPressEvent

logger = get_logger(__name__)


# =============================================================================
# Enums
# =============================================================================


class TUIMode(str, Enum):
    """Agent operating mode."""

    ACT = "act"
    PLAN = "plan"


class TUIState(str, Enum):
    """TUI application state."""

    IDLE = "idle"
    RUNNING = "running"


# =============================================================================
# TUI Application
# =============================================================================


@dataclass
class TUIApp:
    """Main TUI application class.

    Manages the lifecycle of:
    - BrowserManager (optional)
    - AgentRuntime (env + ctx + agent)
    - prompt_toolkit Application

    Usage:
        async with TUIApp(config, config_manager) as app:
            await app.run()
    """

    config: PaintressConfig
    config_manager: ConfigManager
    verbose: bool = False

    # Runtime state
    _mode: TUIMode = field(default=TUIMode.ACT, init=False)
    _state: TUIState = field(default=TUIState.IDLE, init=False)

    # Resources (initialized in __aenter__)
    _exit_stack: AsyncExitStack | None = field(default=None, init=False, repr=False)
    _browser: BrowserManager | None = field(default=None, init=False)
    _runtime: AgentRuntime[TUIContext, str | DeferredToolRequests] | None = field(default=None, init=False)

    # UI components
    _app: Application[None] | None = field(default=None, init=False, repr=False)
    _output_lines: list[str] = field(default_factory=list, init=False)
    _max_output_lines: int = field(default=500, init=False)  # Overridden from config.display
    _renderer: RichRenderer = field(default_factory=RichRenderer, init=False)
    _event_renderer: EventRenderer = field(default_factory=EventRenderer, init=False)

    # Agent execution
    _agent_task: asyncio.Task[None] | None = field(default=None, init=False)
    _last_run: Any | None = field(default=None, init=False)  # AgentRun from last execution
    _message_history: list[Any] | None = field(default=None, init=False)  # Conversation history

    # Tool tracking
    _tool_messages: dict[str, ToolMessage] = field(default_factory=dict, init=False)
    _printed_tool_calls: set[str] = field(default_factory=set, init=False)

    # Subagent state tracking: agent_id -> {"line_index": int, "tool_names": list[str]}
    _subagent_states: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    # Steering pane
    _steering_items: list[tuple[str, str, str]] = field(default_factory=list, init=False)
    _max_steering_lines: int = field(default=5, init=False)

    # Input mode: "send" (Enter sends) or "edit" (Enter inserts newline)
    _input_mode: str = field(default="send", init=False)

    # Mouse support mode
    _mouse_enabled: bool = field(default=True, init=False)

    # Double Ctrl+C exit
    _last_ctrl_c_time: float = field(default=0.0, init=False)
    _ctrl_c_exit_timeout: float = field(default=2.0, init=False)

    # UI component references (for scroll support)
    _output_window: ScrollablePane | None = field(default=None, init=False)

    # Prompt history for up/down navigation
    _prompt_history: list[str] = field(default_factory=list, init=False)
    _history_index: int = field(default=-1, init=False)
    _current_input_backup: str = field(default="", init=False)

    # Streaming text tracking for markdown rendering
    _streaming_text: str = field(default="", init=False)
    _streaming_line_index: int | None = field(default=None, init=False)

    # Streaming thinking tracking for extended thinking display
    _streaming_thinking: str = field(default="", init=False)
    _streaming_thinking_line_index: int | None = field(default=None, init=False)

    # Real-time context usage tracking
    _current_context_tokens: int = field(default=0, init=False)
    _context_window_size: int = field(default=200000, init=False)

    # Session-level usage tracking
    _session_usage: SessionUsage = field(default_factory=SessionUsage, init=False)

    # UI refresh throttling
    _last_invalidate_time: float = field(default=0.0, init=False)
    _invalidate_interval: float = field(default=0.016, init=False)  # ~60fps max

    # HITL (Human-in-the-Loop) approval state
    _hitl_pending: bool = field(default=False, init=False)
    _approval_event: asyncio.Event | None = field(default=None, init=False)
    _approval_result: bool | None = field(default=None, init=False)  # True=approve, False=reject
    _approval_reason: str | None = field(default=None, init=False)
    _pending_approvals: list[ToolCallPart] = field(default_factory=list, init=False)
    _current_approval_index: int = field(default=0, init=False)

    @property
    def mode(self) -> TUIMode:
        """Current agent mode."""
        return self._mode

    @property
    def state(self) -> TUIState:
        """Current application state."""
        return self._state

    @property
    def runtime(self) -> AgentRuntime[TUIContext, str | DeferredToolRequests]:
        """Get agent runtime (must be entered first)."""
        if self._runtime is None:
            raise RuntimeError("TUIApp not entered. Use 'async with app:' first.")
        return self._runtime

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def __aenter__(self) -> TUIApp:
        """Initialize resources."""
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        # Configure TUI logging (queue for internal use, file for verbose mode)
        log_queue: asyncio.Queue[object] = asyncio.Queue()

        # Start browser manager (optional)
        self._browser = BrowserManager(self.config.browser)
        await self._exit_stack.enter_async_context(self._browser)

        # Load MCP config
        mcp_config = self.config_manager.load_mcp_config()

        # Create runtime
        self._runtime = create_tui_runtime(
            config=self.config,
            mcp_config=mcp_config,
            browser_manager=self._browser,
            working_dir=Path.cwd(),
        )
        await self._exit_stack.enter_async_context(self._runtime)

        # Initialize context window size from model config
        if self._runtime.ctx.model_cfg.context_window:
            self._context_window_size = self._runtime.ctx.model_cfg.context_window

        # Apply display config
        self._max_output_lines = self.config.display.max_output_lines

        logger.info("TUIApp initialized")
        configure_tui_logging(log_queue, verbose=self.verbose)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Cleanup resources."""
        # Cancel any running agent task
        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._agent_task

        # Give event loop a chance to process pending cleanups
        await asyncio.sleep(0)

        if self._exit_stack:
            try:
                result = await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
                self._exit_stack = None
                return result
            except (RuntimeError, GeneratorExit, BaseExceptionGroup) as e:
                # Suppress MCP stdio client cleanup errors
                # These occur due to async generator lifecycle issues in pydantic-ai/mcp
                logger.debug("Suppressed cleanup error: %s", e)
                self._exit_stack = None
                return None
        return None

    # =========================================================================
    # Mode Management
    # =========================================================================

    def switch_mode(self, mode: TUIMode) -> None:
        """Switch agent operating mode."""
        if self._state == TUIState.RUNNING:
            self._append_output("[Cannot switch mode while agent is running]")
            return

        if self._mode != mode:
            old_mode = self._mode
            self._mode = mode
            self._append_output(f"[Mode switched: {old_mode.value} -> {mode.value}]")
            if self._app:
                self._app.invalidate()

    # =========================================================================
    # Output Management
    # =========================================================================

    def _throttled_invalidate(self) -> None:
        """Invalidate UI with throttling to prevent excessive redraws."""
        if not self._app:
            return
        now = time.time()
        if now - self._last_invalidate_time >= self._invalidate_interval:
            self._last_invalidate_time = now
            self._app.invalidate()

    def _append_output(self, text: str) -> None:
        """Append text to output buffer with auto-scroll when running."""
        self._output_lines.append(text)
        # Trim old lines to prevent memory issues
        if len(self._output_lines) > self._max_output_lines:
            trim_count = len(self._output_lines) - self._max_output_lines
            self._output_lines = self._output_lines[trim_count:]
        # Auto-scroll to bottom when agent is running
        if self._state == TUIState.RUNNING:
            self._scroll_to_bottom()
        # Invalidate app to refresh display (throttled during streaming)
        self._throttled_invalidate()

    def _scroll_to_bottom(self) -> None:
        """Scroll output pane to bottom.

        Calculates actual content height and scrolls to show bottom content.
        """
        if not self._output_window:
            return
        try:
            # Count total lines in output content
            total_lines = 0
            for line in self._output_lines:
                total_lines += line.count("\n") + 1

            # Get visible height from terminal
            if self._app and self._app.output:
                terminal_size = self._app.output.get_size()
                # Reserve: status bar (2) + steering (dynamic) + input area (5) + margins
                visible_height = max(5, terminal_size.rows - 9)
            else:
                visible_height = 20

            # Calculate scroll needed to show bottom content with padding
            bottom_padding = 4
            if total_lines > visible_height:
                scroll_to = total_lines - visible_height + bottom_padding
                self._output_window.vertical_scroll = max(0, scroll_to)
            else:
                self._output_window.vertical_scroll = 0
        except Exception:  # noqa: S110
            pass

    def _get_output_text(self) -> ANSI:
        """Get formatted output for display."""
        return ANSI("\n".join(self._output_lines))

    def _get_max_scroll(self) -> int:
        """Calculate maximum scroll position."""
        total_lines = sum(line.count("\n") + 1 for line in self._output_lines)
        if self._app and self._app.output:
            terminal_size = self._app.output.get_size()
            visible_height = max(5, terminal_size.rows - 10)
        else:
            visible_height = 20
        return max(0, total_lines - visible_height)

    def _get_terminal_width(self) -> int:
        """Get current terminal width for Rich rendering."""
        if self._app and self._app.output:
            return self._app.output.get_size().columns
        return 120

    def _get_code_theme(self) -> str:
        """Get the code theme for markdown/syntax highlighting."""
        # TODO: Make configurable via config
        return "monokai"

    def _start_streaming_text(self, initial_content: str = "") -> None:
        """Start tracking a new streaming text block."""
        self._streaming_text = initial_content
        self._streaming_line_index = len(self._output_lines)
        # Add placeholder that will be updated
        self._output_lines.append(initial_content)

    def _update_streaming_text(self, delta: str) -> None:
        """Update the current streaming text block with delta."""
        self._streaming_text += delta
        # Re-render markdown for the complete text so far with dynamic width
        if self._streaming_line_index is not None and self._streaming_line_index < len(self._output_lines):
            rendered = self._renderer.render_markdown(
                self._streaming_text,
                code_theme=self._get_code_theme(),
                width=self._get_terminal_width(),
            ).rstrip("\n")
            self._output_lines[self._streaming_line_index] = rendered
            if self._state == TUIState.RUNNING:
                self._scroll_to_bottom()
            self._throttled_invalidate()

    def _finalize_streaming_text(self) -> None:
        """Finalize the current streaming text block."""
        if self._streaming_text and self._streaming_line_index is not None:
            # Final render with complete text and dynamic width
            rendered = self._renderer.render_markdown(
                self._streaming_text,
                code_theme=self._get_code_theme(),
                width=self._get_terminal_width(),
            ).rstrip("\n")
            if self._streaming_line_index < len(self._output_lines):
                self._output_lines[self._streaming_line_index] = rendered
        self._streaming_text = ""
        self._streaming_line_index = None

    def _start_streaming_thinking(self, initial_content: str = "") -> None:
        """Start tracking a new streaming thinking block."""
        self._streaming_thinking = initial_content
        self._streaming_thinking_line_index = len(self._output_lines)
        # Render initial content with thinking style
        rendered = self._event_renderer.render_thinking(initial_content, width=self._get_terminal_width()).rstrip("\n")
        self._output_lines.append(rendered)
        self._throttled_invalidate()

    def _update_streaming_thinking(self, delta: str) -> None:
        """Update current streaming thinking with delta."""
        self._streaming_thinking += delta
        # Re-render thinking for the complete text so far
        if self._streaming_thinking_line_index is not None and self._streaming_thinking_line_index < len(
            self._output_lines
        ):
            rendered = self._event_renderer.render_thinking(
                self._streaming_thinking,
                width=self._get_terminal_width(),
            ).rstrip("\n")
            self._output_lines[self._streaming_thinking_line_index] = rendered
            if self._state == TUIState.RUNNING:
                self._scroll_to_bottom()
            self._throttled_invalidate()

    def _finalize_streaming_thinking(self) -> None:
        """Finalize the current streaming thinking block."""
        if self._streaming_thinking and self._streaming_thinking_line_index is not None:
            # Final render
            rendered = self._event_renderer.render_thinking(
                self._streaming_thinking,
                width=self._get_terminal_width(),
            ).rstrip("\n")
            if self._streaming_thinking_line_index < len(self._output_lines):
                self._output_lines[self._streaming_thinking_line_index] = rendered
        self._streaming_thinking = ""
        self._streaming_thinking_line_index = None

    def _append_user_input(self, text: str) -> None:
        """Render user input with styled prompt indicator and word wrap."""
        width = self._get_terminal_width()
        # Use Rich Text for proper word wrapping
        from rich.text import Text as RichText

        user_text = RichText()
        user_text.append("> ", style="bold green")
        user_text.append(text)
        rendered = self._renderer.render(user_text, width=width).rstrip("\n")
        self._append_output(rendered)

    def _append_error_output(self, e: BaseException) -> None:
        """Render error message with proper word wrapping to fit terminal width."""
        width = self._get_terminal_width()
        from rich.text import Text as RichText

        error_type = type(e).__name__
        error_msg = str(e)

        self._append_output("")

        # Error header
        header = RichText()
        header.append("[ERROR] ", style="bold red")
        header.append(error_type, style="bold red")
        self._append_output(self._renderer.render(header, width=width).rstrip("\n"))

        # Error message (with word wrap)
        msg_text = RichText()
        msg_text.append("  ", style="dim")
        msg_text.append(error_msg)
        self._append_output(self._renderer.render(msg_text, width=width).rstrip("\n"))

        # Cause if available
        if e.__cause__:
            cause_text = RichText()
            cause_text.append("  Caused by: ", style="dim red")
            cause_text.append(f"{type(e.__cause__).__name__}: {e.__cause__}")
            self._append_output(self._renderer.render(cause_text, width=width).rstrip("\n"))

        self._append_output("")

        # Hint
        hint = RichText()
        hint.append("(History not saved - you can retry the same message)", style="dim")
        self._append_output(self._renderer.render(hint, width=width).rstrip("\n"))

    # =========================================================================
    # Steering Pane
    # =========================================================================

    def _get_steering_text(self) -> ANSI:
        """Get formatted steering messages for the steering pane."""
        if not self._steering_items:
            return ANSI(" [Steering messages will appear here during agent execution]")

        lines = []
        for _, text, status in reversed(self._steering_items[-self._max_steering_lines :]):
            if status == "acked":
                lines.append(f"[v] {text}")
            else:
                lines.append(f">>> {text}")

        return ANSI("\n".join(lines))

    def _get_steering_height(self) -> int:
        """Get dynamic height for steering pane."""
        if not self._steering_items:
            return 1
        return min(len(self._steering_items), self._max_steering_lines)

    def _add_steering_message(self, message: str) -> None:
        """Add a steering message to UI and enqueue to steering manager.

        This method:
        1. Adds the message to UI list with 'pending' status
        2. Schedules async enqueue to steering manager

        The UI status will be updated to 'acked' when SteeringInjectedEvent
        is received (event-driven UI update).
        """
        # Add to UI list with pending status (use content as key for matching)
        self._steering_items.append((message, message, "pending"))
        if self._app:
            self._app.invalidate()

        # Schedule async enqueue to steering manager
        asyncio.create_task(self._enqueue_steering(message))  # noqa: RUF006

    async def _enqueue_steering(self, message: str) -> None:
        """Enqueue steering message to the steering manager."""
        try:
            await self.runtime.ctx.steering_manager.enqueue(message)
            logger.debug("Steering message enqueued: %s", message[:50])
        except Exception:
            logger.exception("Failed to enqueue steering message")

    def _ack_steering_by_content(self, content: str) -> None:
        """Mark steering messages as acknowledged by matching content.

        Called when SteeringInjectedEvent is received. Matches messages
        by content since the event contains the full injected content.
        """
        # Split content into individual messages and match
        injected_messages = {line.strip() for line in content.split("\n") if line.strip()}

        for i, (key, text, status) in enumerate(self._steering_items):
            if status == "pending" and text.strip() in injected_messages:
                self._steering_items[i] = (key, text, "acked")

        if self._app:
            self._app.invalidate()

    # =========================================================================
    # Status Bar
    # =========================================================================

    def _get_status_text(self) -> list[tuple[str, str]]:
        """Get formatted status bar text."""
        mode_style = f"class:status-bar.mode-{self._mode.value}"
        state_text = "RUNNING" if self._state == TUIState.RUNNING else "IDLE"

        # Calculate context usage percentage
        if self._current_context_tokens > 0 and self._context_window_size > 0:
            context_pct = f"{self._current_context_tokens / self._context_window_size * 100:.0f}"
        else:
            context_pct = "--"

        # Build status based on state
        if self._state == TUIState.RUNNING:
            # Check if waiting for HITL approval
            if self._hitl_pending:
                approval_progress = f"{self._current_approval_index + 1}/{len(self._pending_approvals)}"
                return [
                    (mode_style, f" {self._mode.value.upper()} "),
                    ("class:status-bar", " | "),
                    ("class:status-bar.warning", f"Approval: {approval_progress}"),
                    ("class:status-bar", " | "),
                    ("class:status-bar", f"Context: {context_pct}%"),
                    ("class:status-bar", " | "),
                    ("class:status-bar", "Enter/Y: Approve | Text: Reject | Ctrl+C: Cancel"),
                ]
            else:
                return [
                    (mode_style, f" {self._mode.value.upper()} "),
                    ("class:status-bar", " | "),
                    ("class:status-bar", f"State: {state_text}"),
                    ("class:status-bar", " | "),
                    ("class:status-bar", f"Context: {context_pct}%"),
                    ("class:status-bar", " | "),
                    ("class:status-bar", "Ctrl+C: Interrupt "),
                ]
        else:
            # IDLE: show input mode and scroll hint
            if self._input_mode == "send":
                input_mode_text = "Enter:Send | Tab:Multiline"
            else:
                input_mode_text = "Enter:Newline | Tab:Send"

            scroll_hint = "Shift+Up/Down: Scroll" if sys.platform == "darwin" else "Ctrl+Up/Down: Scroll"

            return [
                (mode_style, f" {self._mode.value.upper()} "),
                ("class:status-bar", " | "),
                ("class:status-bar", f"State: {state_text}"),
                ("class:status-bar", " | "),
                ("class:status-bar", f"Context: {context_pct}%"),
                ("class:status-bar", " | "),
                ("class:status-bar", input_mode_text),
                ("class:status-bar", " | "),
                ("class:status-bar", scroll_hint),
                ("class:status-bar", " | "),
                ("class:status-bar", "Ctrl+C: Exit "),
            ]

    def _get_prompt(self) -> str:
        """Get the input prompt based on current state."""
        state_indicator = "*" if self._state == TUIState.RUNNING else ">"
        mouse_mode = "scroll" if self._mouse_enabled else "select"
        return f"[{mouse_mode}] {state_indicator} "

    # =========================================================================
    # Agent Execution
    # =========================================================================

    def _load_guidance_files(self) -> tuple[str | None, str | None]:
        """Load project guidance (AGENTS.md) and user rules (RULES.md).

        Returns:
            Tuple of (project_guidance, user_rules), each can be None if not found.
        """
        project_guidance = None
        user_rules = None

        # Load AGENTS.md from working directory
        agents_path = Path.cwd() / "AGENTS.md"
        if agents_path.exists() and agents_path.is_file():
            try:
                content = agents_path.read_text(encoding="utf-8")
                if content.strip():
                    project_guidance = f"<project-guidance name={agents_path.name}>\n{content}\n</project-guidance>"
                    logger.debug(f"Loaded project guidance from {agents_path}")
            except Exception as e:
                logger.warning(f"Failed to read {agents_path}: {e}")

        # Load RULES.md from user config directory
        rules_path = self.config_manager.config_dir / "RULES.md"
        if rules_path.exists() and rules_path.is_file():
            try:
                content = rules_path.read_text(encoding="utf-8")
                if content.strip():
                    user_rules = f"<user-rules location={rules_path.absolute().as_posix()}>\n{content}\n</user-rules>"
                    logger.debug(f"Loaded user rules from {rules_path}")
            except Exception as e:
                logger.warning(f"Failed to read {rules_path}: {e}")

        return project_guidance, user_rules

    def _build_user_prompt(self, user_input: str) -> str | list[str]:
        """Build the full user prompt with optional guidance files and mode reminder.

        Args:
            user_input: The user's input text.

        Returns:
            Either the plain user_input string, or a list of
            [user_input, project_guidance, user_rules, mode_reminder] if guidance files exist
            or plan mode is active.
        """
        project_guidance, user_rules = self._load_guidance_files()

        # Build mode reminder for plan mode
        mode_reminder: str | None = None
        if self._mode == TUIMode.PLAN:
            mode_reminder = (
                "<mode-reminder>\n"
                "You are currently in PLAN mode. In this mode:\n"
                "- Do NOT make any modifications to existing code or files\n"
                "- Do NOT execute commands that change system state\n"
                "- Focus on analysis, discussion, and planning\n"
                "- You MAY create new draft files (e.g., in .drafts/ or .handoff/) to save context, "
                "discussion results, plans, or design documents\n"
                "You can ask user to switch back to ACT mode by typing '/act'\n"
                "</mode-reminder>"
            )

        # If no guidance files and not in plan mode, return plain string
        if not project_guidance and not user_rules and not mode_reminder:
            return user_input

        # Build list with non-None items
        parts = [user_input]
        if project_guidance:
            parts.append(project_guidance)
        if user_rules:
            parts.append(user_rules)
        if mode_reminder:
            parts.append(mode_reminder)

        return parts

    def _on_agent_task_done(self, task: asyncio.Task[None]) -> None:
        """Callback when agent task completes - handles uncaught exceptions."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            # Exception was not caught in _run_agent - display it
            logger.error("Uncaught exception in agent task: %s: %s", type(exc).__name__, str(exc))
            self._append_error_output(exc)
            self._state = TUIState.IDLE
            if self._app:
                self._app.invalidate()

    async def _run_agent(self, user_input: str) -> None:
        """Execute agent with HITL inner loop for tool approvals."""
        self._state = TUIState.RUNNING
        self._tool_messages.clear()
        self._printed_tool_calls.clear()
        self._subagent_states.clear()
        self._event_renderer.clear()

        try:
            # Initial agent execution
            result = await self._execute_stream(user_input)

            # HITL inner loop: keep processing until we get final str output
            while result and isinstance(result.output, DeferredToolRequests):
                deferred = result.output
                if not deferred.approvals:
                    # No approvals needed, just continue
                    break

                # Collect user approval decisions
                user_response = await self._request_user_action(deferred)

                # Resume agent with approval results
                result = await self._execute_stream(user_response)

            # Agent completed successfully

        except asyncio.CancelledError:
            self._finalize_streaming_text()
            self._finalize_streaming_thinking()
            self._append_output("[Cancelled]")
            # Don't update message history on cancel - allows retry
        except Exception as e:
            self._finalize_streaming_text()
            self._finalize_streaming_thinking()
            # Display error in output pane - don't save history so user can retry
            self._append_error_output(e)
            logger.exception("Agent execution failed")
        finally:
            # Finalize any remaining streaming text/thinking
            self._finalize_streaming_text()
            self._finalize_streaming_thinking()
            # Reset all HITL state
            self._reset_hitl_state()
            self._state = TUIState.IDLE
            if self._app:
                self._app.invalidate()

    def _reset_hitl_state(self) -> None:
        """Reset all HITL-related state variables.

        Called after agent execution completes (success, error, or cancel)
        to ensure clean state for next execution.
        """
        self._hitl_pending = False
        self._pending_approvals.clear()
        self._current_approval_index = 0
        self._approval_result = None
        self._approval_reason = None
        # Don't set _approval_event to None here as it may still be awaited
        # Instead, set it if it exists to unblock any waiting coroutine
        if self._approval_event and not self._approval_event.is_set():
            # Signal cancellation by setting result to False
            self._approval_result = False
            self._approval_reason = "Cancelled"
            self._approval_event.set()
        self._approval_event = None

    async def _execute_stream(
        self,
        prompt: str | DeferredToolResults,
    ) -> Any:
        """Execute a single agent stream and return the result.

        Args:
            prompt: User prompt string or DeferredToolResults from approval.

        Returns:
            AgentRunResult with output (str or DeferredToolRequests).
        """
        # Clear tracking for new stream
        self._tool_messages.clear()
        self._printed_tool_calls.clear()
        self._subagent_states.clear()

        # Build user prompt if string input
        if isinstance(prompt, str):
            user_prompt = self._build_user_prompt(prompt)
            deferred_results = None
        else:
            user_prompt = ""
            deferred_results = prompt

        async with stream_agent(
            self.runtime,  # type: ignore[arg-type]
            user_prompt=user_prompt if user_prompt else None,
            message_history=self._message_history,
            deferred_tool_results=deferred_results,
            usage_limits=UsageLimits(request_limit=self.config.general.max_requests),
            post_node_hook=emit_context_update,
        ) as stream:
            async for event in stream:
                self._handle_stream_event(event)

            stream.raise_if_exception()
            # Save run and update message history for next conversation
            self._last_run = stream.run
            if stream.run:
                self._message_history = list(stream.run.all_messages())
                # Update context usage from run
                usage = stream.run.usage()
                latest_usage = get_latest_request_usage(self._message_history)
                self._current_context_tokens = latest_usage.total_tokens if latest_usage else usage.total_tokens

                # Accumulate session usage
                model_id = cast(Model, self.runtime.agent.model).model_name
                self._session_usage.add("main", model_id, usage)

                # Also accumulate extra_usages (subagents, image_understanding, etc.)
                ctx = self.runtime.ctx
                for record in ctx.extra_usages:
                    self._session_usage.add(record.agent, record.model_id, record.usage)

                # Auto-save session after each run (before clearing extra_usages)
                self._auto_save_history()

                # Clear extra_usages after saving to avoid double counting on next run
                ctx.extra_usages.clear()

            return stream.run.result if stream.run else None

    async def _request_user_action(
        self,
        deferred: DeferredToolRequests,
    ) -> DeferredToolResults:
        """Collect approval decisions from user for pending tool calls.

        Args:
            deferred: DeferredToolRequests containing tools needing approval.

        Returns:
            DeferredToolResults with approval decisions.
        """
        results = DeferredToolResults()

        if not deferred.approvals:
            return results

        self._hitl_pending = True
        self._pending_approvals = list(deferred.approvals)
        self._current_approval_index = 0

        self._append_output("")
        self._append_output(f"[Tool approval required: {len(deferred.approvals)} tool(s)]")

        for idx, tool_call in enumerate(deferred.approvals):
            self._current_approval_index = idx
            # Display approval panel
            self._display_approval_panel(tool_call, idx + 1, len(deferred.approvals))

            # Wait for user decision (blocking with asyncio.Event)
            approved, reason = await self._wait_for_approval_input()

            if approved:
                results.approvals[tool_call.tool_call_id] = True
                self._append_output(f"  [Approved: {tool_call.tool_name}]")
            else:
                results.approvals[tool_call.tool_call_id] = ToolDenied(reason or "User rejected")
                self._append_output(f"  [Rejected: {tool_call.tool_name} - {reason or 'User rejected'}]")

        self._hitl_pending = False
        self._pending_approvals.clear()
        self._append_output("")

        return results

    async def _wait_for_approval_input(self) -> tuple[bool, str | None]:
        """Wait for user's approval decision.

        Returns:
            Tuple of (approved: bool, reason: str | None).
        """
        self._approval_event = asyncio.Event()
        self._approval_result = None
        self._approval_reason = None

        if self._app:
            self._app.invalidate()

        # Wait for key handler to set the event
        await self._approval_event.wait()

        approved = self._approval_result if self._approval_result is not None else False
        reason = f"User not approved with response: `{self._approval_reason}`"

        self._approval_event = None
        return approved, reason

    def _format_args_for_display(self, args: Any, max_str_len: int = 500, max_lines: int = 30) -> str:
        """Format tool arguments for display with smart truncation.

        Args:
            args: Tool arguments (can be dict, JSON string, or any object)
            max_str_len: Maximum length for string values before truncation
            max_lines: Maximum number of lines in output

        Returns:
            Formatted JSON string or fallback representation
        """

        def truncate_strings(obj: Any, max_len: int) -> Any:
            """Recursively truncate long strings in nested structures."""
            if isinstance(obj, str):
                if len(obj) > max_len:
                    return obj[:max_len] + f"... ({len(obj) - max_len} more chars)"
                return obj
            elif isinstance(obj, dict):
                return {k: truncate_strings(v, max_len) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [truncate_strings(item, max_len) for item in obj]
            return obj

        try:
            # If args is a string, try to parse it as JSON first
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                    args = parsed
                except json.JSONDecodeError:
                    # Not valid JSON, treat as plain string
                    if len(args) > max_str_len:
                        return args[:max_str_len] + f"\n... ({len(args) - max_str_len} more chars)"
                    return args

            # Truncate long strings in the structure
            truncated = truncate_strings(args, max_str_len)

            # Format as pretty JSON
            formatted = json.dumps(truncated, indent=2, ensure_ascii=False)

            # Limit total lines
            lines = formatted.split("\n")
            if len(lines) > max_lines:
                formatted = "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"

            return formatted

        except Exception:
            # Ultimate fallback: convert to string
            result = str(args)
            if len(result) > max_str_len:
                result = result[:max_str_len] + f"... ({len(result) - max_str_len} more chars)"
            return result

    def _display_approval_panel(
        self,
        tool_call: ToolCallPart,
        index: int,
        total: int,
    ) -> None:
        """Display approval panel for a tool call."""
        from rich.console import Group
        from rich.panel import Panel
        from rich.syntax import Syntax

        content_parts: list[Any] = [
            Text(f"Tool {index} of {total}", style="bold cyan"),
            Text(""),
            Text(f"Tool: {tool_call.tool_name}", style="bold yellow"),
        ]

        if tool_call.args:
            content_parts.append(Text(""))
            content_parts.append(Text("Arguments:", style="bold cyan"))
            formatted_args = self._format_args_for_display(tool_call.args)
            code_theme = self.config.display.code_theme or "monokai"
            # Determine if it looks like JSON for syntax highlighting
            is_json_like = formatted_args.strip().startswith(("{", "["))
            syntax = Syntax(formatted_args, "json" if is_json_like else "text", theme=code_theme)
            content_parts.append(syntax)

        panel = Panel(
            Group(*content_parts),
            title="[yellow]Tool Approval Required[/yellow]",
            subtitle="[dim]Enter/Y: Approve | Any text: Reject with reason | Ctrl+C: Cancel[/dim]",
            border_style="yellow",
            padding=(1, 2),
        )

        # Render panel to ANSI and append
        rendered = self._renderer.render(panel, width=self._get_terminal_width())
        self._append_output(rendered.rstrip())

    # =========================================================================
    # Subagent Event Handling
    # =========================================================================

    def _handle_subagent_start(self, event: SubagentStartEvent) -> None:
        """Handle subagent start event - create progress line."""
        agent_id = event.agent_id
        agent_name = event.agent_name

        # Create progress line
        text = Text()
        text.append(f"[{agent_id}] ", style="cyan")
        text.append("Running...", style="dim")
        rendered = self._renderer.render(text, width=self._get_terminal_width())

        # Track state with line index for later update
        line_index = len(self._output_lines)
        self._subagent_states[agent_id] = {
            "line_index": line_index,
            "tool_names": [],
            "agent_name": agent_name,
        }
        self._output_lines.append(rendered.rstrip())
        self._throttled_invalidate()

    def _handle_subagent_complete(self, event: SubagentCompleteEvent) -> None:
        """Handle subagent complete event - update progress line to summary."""
        agent_id = event.agent_id

        if agent_id not in self._subagent_states:
            # Start event was missed, just show completion
            text = Text()
            if event.success:
                text.append(f"[{agent_id}] ", style="cyan")
                text.append("Done ", style="bold green")
                text.append(f"({event.duration_seconds:.1f}s)", style="dim")
                if event.request_count > 0:
                    text.append(f" | {event.request_count} reqs", style="dim")
            else:
                text.append(f"[{agent_id}] ", style="cyan")
                text.append("Failed ", style="bold red")
                text.append(f"({event.duration_seconds:.1f}s)", style="dim")
                if event.error:
                    text.append(f" | {event.error[:50]}", style="dim red")
            rendered = self._renderer.render(text, width=self._get_terminal_width())
            self._append_output(rendered.rstrip())
            return

        state = self._subagent_states[agent_id]
        line_index = state["line_index"]

        # Build summary line
        text = Text()
        if event.success:
            text.append(f"[{agent_id}] ", style="cyan")
            text.append("Done ", style="bold green")
            text.append(f"({event.duration_seconds:.1f}s)", style="dim")
            if event.request_count > 0:
                text.append(f" | {event.request_count} reqs", style="dim")
            if event.result_preview:
                # Truncate result preview
                preview = event.result_preview.replace("\n", " ")[:60]
                if len(event.result_preview) > 60:
                    preview += "..."
                text.append(f' | "{preview}"', style="dim italic")
        else:
            text.append(f"[{agent_id}] ", style="cyan")
            text.append("Failed ", style="bold red")
            text.append(f"({event.duration_seconds:.1f}s)", style="dim")
            if event.error:
                error_preview = event.error[:50]
                text.append(f" | {error_preview}", style="dim red")

        rendered = self._renderer.render(text, width=self._get_terminal_width())

        # Update the line in place
        if line_index < len(self._output_lines):
            self._output_lines[line_index] = rendered.rstrip()

        # Clean up state
        del self._subagent_states[agent_id]
        self._throttled_invalidate()

    def _update_subagent_progress_line(self, agent_id: str) -> None:
        """Update subagent progress line with current tool list."""
        if agent_id not in self._subagent_states:
            return

        state = self._subagent_states[agent_id]
        line_index = state["line_index"]
        tool_names = state["tool_names"]

        # Build progress line
        text = Text()
        text.append(f"[{agent_id}] ", style="cyan")
        text.append("Running... ", style="dim")

        if tool_names:
            # Show last few tools
            recent_tools = tool_names[-3:]  # Last 3 tools
            tools_str = ", ".join(recent_tools)
            if len(tool_names) > 3:
                tools_str = f"...{tools_str}"
            text.append(tools_str, style="dim yellow")
            text.append(f" ({len(tool_names)} tools)", style="dim")

        rendered = self._renderer.render(text, width=self._get_terminal_width())

        # Update the line in place
        if line_index < len(self._output_lines):
            self._output_lines[line_index] = rendered.rstrip()
            self._throttled_invalidate()

    def _handle_stream_event(self, event: StreamEvent) -> None:
        """Handle a stream event from agent execution."""
        message_event = event.event
        agent_id = event.agent_id

        # Handle subagent lifecycle events (from any agent)
        if isinstance(message_event, SubagentStartEvent):
            self._handle_subagent_start(message_event)
            return

        if isinstance(message_event, SubagentCompleteEvent):
            self._handle_subagent_complete(message_event)
            return

        # For subagent events (not main), only track tool calls silently
        if agent_id != "main" and agent_id in self._subagent_states:
            if isinstance(message_event, FunctionToolCallEvent):
                # Track tool name for progress display
                tool_name = message_event.part.tool_name
                self._subagent_states[agent_id]["tool_names"].append(tool_name)
                self._update_subagent_progress_line(agent_id)
            # Ignore all other subagent events (text streaming, tool results, etc.)
            return

        # Main agent events - normal processing
        if isinstance(message_event, PartStartEvent) and isinstance(message_event.part, TextPart):
            # Start new streaming text block
            self._finalize_streaming_text()  # Finalize any previous
            self._finalize_streaming_thinking()  # Finalize any thinking
            self._start_streaming_text(message_event.part.content)

        elif isinstance(message_event, PartStartEvent) and isinstance(message_event.part, ThinkingPart):
            # Start new streaming thinking block (extended thinking from model)
            self._finalize_streaming_thinking()  # Finalize any previous
            self._start_streaming_thinking(message_event.part.content)

        elif isinstance(message_event, PartDeltaEvent) and isinstance(message_event.delta, TextPartDelta):
            # Update streaming text with delta
            if self._streaming_line_index is not None:
                self._update_streaming_text(message_event.delta.content_delta)
            else:
                # Fallback if no streaming started
                self._start_streaming_text(message_event.delta.content_delta)

        elif isinstance(message_event, PartDeltaEvent) and isinstance(message_event.delta, ThinkingPartDelta):
            # Update streaming thinking with delta
            if message_event.delta.content_delta:
                if self._streaming_thinking_line_index is not None:
                    self._update_streaming_thinking(message_event.delta.content_delta)
                else:
                    # Fallback if no streaming started
                    self._start_streaming_thinking(message_event.delta.content_delta)

        elif isinstance(message_event, FunctionToolCallEvent):
            # Finalize any streaming text before tool call
            self._finalize_streaming_text()
            self._finalize_streaming_thinking()

            tool_call_id = message_event.part.tool_call_id
            tool_name = message_event.part.tool_name
            self._tool_messages[tool_call_id] = ToolMessage(
                tool_call_id=tool_call_id,
                name=tool_name,
                args=message_event.part.args,
            )
            self._event_renderer.tracker.start_call(tool_call_id, tool_name, message_event.part.args)
            rendered = self._event_renderer.render_tool_call_start(tool_name, tool_call_id)
            self._append_output(rendered.rstrip())

        elif isinstance(message_event, FunctionToolResultEvent):
            tool_call_id = message_event.tool_call_id
            if tool_call_id in self._tool_messages:
                tool_msg = self._tool_messages[tool_call_id]
                result_content = self._extract_tool_result(message_event)
                tool_msg.content = result_content
                self._event_renderer.tracker.complete_call(tool_call_id, result_content)

                if tool_call_id not in self._printed_tool_calls:
                    # Get duration from tracker
                    duration = 0.0
                    if tool_call_id in self._event_renderer.tracker.tool_calls:
                        duration = self._event_renderer.tracker.tool_calls[tool_call_id].duration()
                    rendered = self._event_renderer.render_tool_call_complete(
                        tool_msg, duration=duration, width=self._get_terminal_width()
                    )
                    self._append_output(rendered.rstrip())
                    self._printed_tool_calls.add(tool_call_id)

        # Handle SDK events (compact, handoff)
        elif isinstance(message_event, CompactStartEvent):
            self._finalize_streaming_text()
            self._finalize_streaming_thinking()
            rendered = self._event_renderer.render_compact_start(message_event.message_count)
            self._append_output(rendered.rstrip())

        elif isinstance(message_event, CompactCompleteEvent):
            rendered = self._event_renderer.render_compact_complete(
                message_event.original_message_count,
                message_event.compacted_message_count,
                message_event.summary_markdown,
            )
            self._append_output(rendered.rstrip())

        elif isinstance(message_event, CompactFailedEvent):
            rendered = self._event_renderer.render_compact_failed(message_event.error)
            self._append_output(rendered.rstrip())

        elif isinstance(message_event, HandoffStartEvent):
            self._finalize_streaming_text()
            self._finalize_streaming_thinking()
            rendered = self._event_renderer.render_handoff_start(message_event.message_count)
            self._append_output(rendered.rstrip())

        elif isinstance(message_event, HandoffCompleteEvent):
            rendered = self._event_renderer.render_handoff_complete(message_event.handoff_content)
            self._append_output(rendered.rstrip())

        elif isinstance(message_event, HandoffFailedEvent):
            rendered = self._event_renderer.render_handoff_failed(message_event.error)
            self._append_output(rendered.rstrip())

        # Handle TUI-specific events
        elif isinstance(message_event, ContextUpdateEvent):
            self._current_context_tokens = message_event.total_tokens
            if message_event.context_window_size > 0:
                self._context_window_size = message_event.context_window_size

        elif isinstance(message_event, AgentPhaseEvent):
            # Update status based on phase (handled by status bar)
            pass

        elif isinstance(message_event, SteeringInjectedEvent):
            # Update UI status to acked for matched steering messages
            self._ack_steering_by_content(message_event.content)

            rendered = self._event_renderer.render_steering_injected(
                message_event.message_count,
                message_event.content,
            )
            self._append_output(rendered.rstrip())

        if self._app:
            self._app.invalidate()

    def _extract_tool_result(self, event: FunctionToolResultEvent) -> str:
        """Extract result content from tool result event."""
        try:
            result = event.result
            if hasattr(result, "content"):
                content = result.content
                if isinstance(content, str):
                    return content
                rv = getattr(content, "return_value", None)
                if rv is not None:
                    return str(rv)
                return str(content)
            return str(result)
        except Exception:
            return "<result>"

    # =========================================================================
    # UI Setup
    # =========================================================================

    def _setup_keybindings(self, input_area: TextArea) -> KeyBindings:
        """Set up keyboard bindings."""
        kb = KeyBindings()

        @kb.add("c-c")
        def handle_ctrl_c(event: KeyPressEvent) -> None:
            """Handle Ctrl+C - cancel running task or double-press to exit."""
            current_time = time.time()

            if self._state == TUIState.RUNNING:
                # Running: request cancellation (state change handled by _run_agent finally)
                if self._agent_task and not self._agent_task.done():
                    self._agent_task.cancel()
                    self._append_output("[Cancelling...]")
            else:
                # Idle: double-press to exit, single-press to clear input
                if current_time - self._last_ctrl_c_time < self._ctrl_c_exit_timeout:
                    event.app.exit()
                else:
                    self._append_output("[Press Ctrl+C again to exit, or Ctrl+D to exit immediately]")
                    self._last_ctrl_c_time = current_time
                    # Clear input area on first Ctrl+C
                    input_area.buffer.reset()

        @kb.add("c-d")
        def handle_ctrl_d(event: KeyPressEvent) -> None:
            """Handle Ctrl+D - exit."""
            event.app.exit()

        # Scroll functions
        def _scroll_up(event: KeyPressEvent) -> None:
            """Scroll output up."""
            if self._output_window:
                with contextlib.suppress(Exception):
                    self._output_window.vertical_scroll = max(0, self._output_window.vertical_scroll - 10)

        def _scroll_down(event: KeyPressEvent) -> None:
            """Scroll output down."""
            if self._output_window:
                with contextlib.suppress(Exception):
                    max_scroll = self._get_max_scroll()
                    new_scroll = self._output_window.vertical_scroll + 10
                    self._output_window.vertical_scroll = min(new_scroll, max_scroll)

        # Register scroll keybindings
        kb.add("pageup")(_scroll_up)
        kb.add("pagedown")(_scroll_down)
        if sys.platform == "darwin":
            kb.add("s-up")(_scroll_up)
            kb.add("s-down")(_scroll_down)
        else:
            kb.add("c-up")(_scroll_up)
            kb.add("c-down")(_scroll_down)

        @kb.add("c-l")
        def handle_ctrl_l(event: KeyPressEvent) -> None:
            """Scroll to bottom of output."""
            self._scroll_to_bottom()

        @kb.add("c-u")
        def handle_ctrl_u(event: KeyPressEvent) -> None:
            """Clear input line."""
            input_area.buffer.reset()
            self._history_index = -1

        @kb.add("up")
        def handle_up(event: KeyPressEvent) -> None:
            """Navigate to previous prompt in history."""
            if not self._prompt_history:
                return
            # First time pressing up: backup current input
            if self._history_index == -1:
                self._current_input_backup = input_area.buffer.text
                self._history_index = len(self._prompt_history)
            # Move to previous item
            if self._history_index > 0:
                self._history_index -= 1
                input_area.buffer.text = self._prompt_history[self._history_index]
                input_area.buffer.cursor_position = len(input_area.buffer.text)

        @kb.add("down")
        def handle_down(event: KeyPressEvent) -> None:
            """Navigate to next prompt in history."""
            if self._history_index == -1:
                return
            # Move to next item
            self._history_index += 1
            if self._history_index >= len(self._prompt_history):
                # Reached end, restore original input
                self._history_index = -1
                input_area.buffer.text = self._current_input_backup
            else:
                input_area.buffer.text = self._prompt_history[self._history_index]
            input_area.buffer.cursor_position = len(input_area.buffer.text)

        @kb.add("escape")
        def handle_escape(event: KeyPressEvent) -> None:
            """Toggle mouse support mode."""
            self._mouse_enabled = not self._mouse_enabled
            if self._app and self._app.output:
                if self._mouse_enabled:
                    self._app.output.enable_mouse_support()
                else:
                    self._app.output.disable_mouse_support()

        @kb.add("enter")
        def handle_enter(event: KeyPressEvent) -> None:
            """Handle Enter based on current input mode."""
            if self._input_mode == "send":
                text = input_area.buffer.text.strip()

                # Check if waiting for HITL approval input
                if self._hitl_pending and self._approval_event and not self._approval_event.is_set():
                    input_area.buffer.reset()
                    # Approve if empty, y, Y, yes, YES
                    if text.lower() in ("", "y", "yes"):
                        self._approval_result = True
                        self._approval_reason = None
                    else:
                        self._approval_result = False
                        self._approval_reason = text if text else None
                    self._approval_event.set()
                    return

                if text:
                    # Reset history navigation
                    self._history_index = -1
                    self._current_input_backup = ""

                    # Save to prompt history (avoid duplicates)
                    if not self._prompt_history or self._prompt_history[-1] != text:
                        self._prompt_history.append(text)

                    if self._state == TUIState.RUNNING:
                        # Add steering message and enqueue to steering manager
                        self._add_steering_message(text)
                        input_area.buffer.reset()
                    else:
                        input_area.buffer.reset()
                        # Handle slash commands
                        if text.startswith("/"):
                            asyncio.create_task(self._handle_command(text))  # noqa: RUF006
                        # Handle shell commands
                        elif text.startswith("!"):
                            asyncio.create_task(self._execute_shell_command(text[1:]))  # noqa: RUF006
                        else:
                            self._append_user_input(text)
                            self._agent_task = asyncio.create_task(self._run_agent(text))
                            self._agent_task.add_done_callback(self._on_agent_task_done)
                else:
                    # Empty input - also handle HITL approval (approve with Enter)
                    if self._hitl_pending and self._approval_event and not self._approval_event.is_set():
                        self._approval_result = True
                        self._approval_reason = None
                        self._approval_event.set()
                    input_area.buffer.reset()
            else:
                input_area.buffer.insert_text("\n")

        @kb.add("tab")
        def handle_tab(event: KeyPressEvent) -> None:
            """Toggle input mode between send and edit."""
            if self._input_mode == "send":
                self._input_mode = "edit"
            else:
                self._input_mode = "send"
            if self._app:
                self._app.invalidate()

        @kb.add("c-o")
        def handle_newline(event: KeyPressEvent) -> None:
            """Insert newline with Ctrl+O (works in both modes)."""
            input_area.buffer.insert_text("\n")

        # Word navigation (Option+Arrow on macOS)
        @kb.add("escape", "b")
        def handle_word_left(event: KeyPressEvent) -> None:
            """Move cursor to previous word."""
            buff = input_area.buffer
            pos = buff.document.find_previous_word_beginning(count=1)
            if pos:
                buff.cursor_position += pos

        @kb.add("escape", "f")
        def handle_word_right(event: KeyPressEvent) -> None:
            """Move cursor to next word."""
            buff = input_area.buffer
            pos = buff.document.find_next_word_ending(count=1)
            if pos:
                buff.cursor_position += pos

        return kb

    def _setup_style(self) -> Style:
        """Set up UI styles."""
        return Style.from_dict({
            "status-bar": "bg:ansiblue fg:white",
            "status-bar.mode-act": "bg:ansigreen fg:black bold",
            "status-bar.mode-plan": "bg:ansiblue fg:white bold",
            "steering-pane": "bg:ansibrightblack fg:ansicyan",
            "input-area": "",
        })

    # =========================================================================
    # Command Handling
    # =========================================================================

    async def _handle_command(self, command: str) -> None:
        """Handle slash commands."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Built-in system commands (cannot be overridden)
        match cmd:
            case "/help":
                self._append_user_input(command)
                self._show_help()
            case "/clear":
                self._append_user_input(command)
                self._clear_session()
            case "/cost":
                self._append_user_input(command)
                self._show_cost()
            case "/dump":
                self._append_user_input(command)
                self._dump_history(args.strip() if args else None)
            case "/load":
                self._append_user_input(command)
                if not args.strip():
                    # Load from auto-save directory
                    auto_save_dir = self.config_manager.get_auto_save_dir()
                    if auto_save_dir.exists():
                        self._load_history(str(auto_save_dir))
                    else:
                        self._append_system_output("No auto-saved session found for this project")
                        self._append_system_output(f"Expected: {auto_save_dir}")
                else:
                    self._load_history(args.strip())
            case "/exit":
                self._append_user_input(command)
                if self._app:
                    self._app.exit()
            case "/act":
                self._append_user_input(command)
                self.switch_mode(TUIMode.ACT)
                self._append_system_output("Mode changed to ACT")
            case "/plan":
                self._append_user_input(command)
                self.switch_mode(TUIMode.PLAN)
                self._append_system_output("Mode changed to PLAN")
            case _:
                # Check custom commands
                cmd_name = cmd[1:]  # Remove leading /
                commands = self.config.get_commands()
                if cmd_name in commands:
                    cmd_def = commands[cmd_name]
                    # Switch mode if specified
                    if cmd_def.mode:
                        new_mode = TUIMode.ACT if cmd_def.mode == "act" else TUIMode.PLAN
                        self.switch_mode(new_mode)
                    # Show expanded prompt instead of command name
                    self._append_user_input(cmd_def.prompt)
                    self._agent_task = asyncio.create_task(self._run_agent(cmd_def.prompt))
                    self._agent_task.add_done_callback(self._on_agent_task_done)
                else:
                    self._append_user_input(command)
                    self._append_system_output(f"Unknown command: {cmd}")

        if self._app:
            self._app.invalidate()

    async def _execute_shell_command(self, command_str: str) -> None:
        """Execute a shell command directly and display output."""
        import os

        if not command_str.strip():
            self._append_system_output("Usage: !<command>")
            return

        # Show command being executed
        cmd_text = Text()
        cmd_text.append("$ ", style="bold cyan")
        cmd_text.append(command_str, style="cyan")
        self._append_output(self._renderer.render(cmd_text).rstrip())

        start_time = time.time()

        try:
            process = await asyncio.create_subprocess_shell(
                command_str,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd(),
                env=os.environ.copy(),
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
            elapsed = time.time() - start_time

            # Display stdout (limit to 100 lines)
            if stdout:
                stdout_text = stdout.decode("utf-8", errors="replace").strip()
                if stdout_text:
                    lines = stdout_text.split("\n")
                    if len(lines) > 100:
                        lines = lines[:100]
                        lines.append(f"... ({len(stdout_text.split(chr(10))) - 100} more lines)")
                    self._append_output("\n".join(lines))

            # Display stderr in red (limit to 50 lines)
            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace").strip()
                if stderr_text:
                    lines = stderr_text.split("\n")
                    if len(lines) > 50:
                        lines = lines[:50]
                        lines.append("... (truncated)")
                    err_output = Text("\n".join(lines), style="red")
                    self._append_output(self._renderer.render(err_output).rstrip())

            # Show exit code if non-zero
            if process.returncode != 0:
                self._append_system_output(f"Exit code: {process.returncode}")

            # Show elapsed time
            self._append_output(f"({elapsed:.1f}s)")

        except TimeoutError:
            self._append_system_output("Command timed out (300s)")
        except Exception as e:
            self._append_system_output(f"Error: {type(e).__name__}: {e}")
        finally:
            if self._app:
                self._app.invalidate()

    def _show_help(self) -> None:
        """Display help text."""
        from rich.table import Table

        lines = []

        # Header
        header = Text("Available Commands", style="bold cyan")
        lines.append(self._renderer.render(header).rstrip())

        # System commands
        sys_table = Table(show_header=False, box=None, padding=(0, 2))
        sys_table.add_column("Command", style="green")
        sys_table.add_column("Description")
        sys_table.add_row("/help", "Show this help")
        sys_table.add_row("/clear", "Clear output and history")
        sys_table.add_row("/cost", "Show cost summary")
        sys_table.add_row("/dump [folder]", "Export session to folder")
        sys_table.add_row("/load <folder>", "Load session from folder")
        sys_table.add_row("/act", "Switch to ACT mode")
        sys_table.add_row("/plan", "Switch to PLAN mode")
        sys_table.add_row("/exit", "Exit TUI")
        lines.append(self._renderer.render(sys_table).rstrip())

        # Custom commands
        commands = self.config.get_commands()
        if commands:
            custom_header = Text("\nCustom Commands", style="bold cyan")
            lines.append(self._renderer.render(custom_header).rstrip())

            custom_table = Table(show_header=False, box=None, padding=(0, 2))
            custom_table.add_column("Command", style="yellow")
            custom_table.add_column("Description")
            for name, cmd_def in sorted(commands.items()):
                desc = cmd_def.description or "(no description)"
                custom_table.add_row(f"/{name}", desc)
            lines.append(self._renderer.render(custom_table).rstrip())

        # Shell
        shell_header = Text("\nShell", style="bold cyan")
        lines.append(self._renderer.render(shell_header).rstrip())
        lines.append("  !<cmd>         Execute shell command directly")

        # Key bindings
        kb_header = Text("\nKey Bindings", style="bold cyan")
        lines.append(self._renderer.render(kb_header).rstrip())

        kb_table = Table(show_header=False, box=None, padding=(0, 2))
        kb_table.add_column("Key", style="yellow")
        kb_table.add_column("Action")
        kb_table.add_row("Ctrl+C", "Cancel / double-press exit")
        kb_table.add_row("Ctrl+D", "Exit")
        kb_table.add_row("Tab", "Toggle input mode")
        kb_table.add_row("Escape", "Toggle mouse mode")
        kb_table.add_row("Up/Down", "Browse history")
        lines.append(self._renderer.render(kb_table).rstrip())

        self._append_output("\n".join(lines))

    def _clear_session(self) -> None:
        """Clear output and message history.

        Resets:
        - Output lines and streaming state
        - Conversation history
        - Status bar context percentage
        - Scroll position

        Preserves:
        - Session usage (token/cost tracking)
        """
        self._output_lines.clear()
        # Reset streaming state
        self._streaming_text = ""
        self._streaming_line_index = None
        self._streaming_thinking = ""
        self._streaming_thinking_line_index = None
        self._printed_tool_calls.clear()
        self._tool_messages.clear()
        self._subagent_states.clear()
        self._steering_items.clear()
        # Clear conversation history
        self._message_history = None
        self._last_run = None
        # Reset status bar context (but keep usage)
        self._current_context_tokens = 0
        # Reset scroll position to top
        if self._output_window:
            self._output_window.vertical_scroll = 0
        # Show help after clear
        self._show_help()

    def _show_cost(self) -> None:
        """Show token usage summary for the current session."""
        summary = self._session_usage.format_summary()
        self._append_system_output(summary)

    def _dump_history(self, folder_path: str | None) -> None:
        """Dump session state to a folder.

        Creates a folder containing:
        - message_history.json: The conversation history
        - context_state.json: The agent context state (subagent history, etc.)

        Args:
            folder_path: Target folder path. Defaults to ".paintress-session".
        """
        if not self._message_history:
            self._append_system_output("No conversation history to dump")
            return

        dump_dir = Path(folder_path or ".paintress-session").expanduser().resolve()
        try:
            # Create folder
            dump_dir.mkdir(parents=True, exist_ok=True)

            # Save message history
            history_file = dump_dir / "message_history.json"
            history_file.write_bytes(ModelMessagesTypeAdapter.dump_json(self._message_history, indent=2))

            # Save context state
            state_file = dump_dir / "context_state.json"
            state = self.runtime.ctx.export_state()
            state_file.write_text(state.model_dump_json(indent=2))

            self._append_system_output(f"Session dumped to {dump_dir}")
            self._append_system_output(f"  - message_history.json ({len(self._message_history)} messages)")
            self._append_system_output("  - context_state.json")
        except Exception as e:
            self._append_system_output(f"Error: {e}")

    def _load_history(self, folder_path: str) -> None:
        """Load session state from a folder.

        Loads from a folder containing:
        - message_history.json: The conversation history
        - context_state.json: The agent context state (optional)

        Args:
            folder_path: Source folder path.
        """
        load_dir = Path(folder_path).expanduser().resolve()

        if not load_dir.is_dir():
            self._append_system_output(f"Not a directory: {load_dir}")
            return

        history_file = load_dir / "message_history.json"
        state_file = load_dir / "context_state.json"

        if not history_file.exists():
            self._append_system_output(f"message_history.json not found in {load_dir}")
            return

        try:
            # Load message history
            history_data = history_file.read_bytes()
            history = ModelMessagesTypeAdapter.validate_json(history_data)
            self._message_history = history

            # Load context state if exists
            if state_file.exists():
                state_data = state_file.read_text()
                state = ResumableState.model_validate_json(state_data)
                state.restore(self.runtime.ctx)

                # Re-populate session usage from restored extra_usages
                for record in self.runtime.ctx.extra_usages:
                    self._session_usage.add(record.agent, record.model_id, record.usage)
                # Clear after populating to avoid double counting on next run
                self.runtime.ctx.extra_usages.clear()

                self._append_system_output(f"Session loaded from {load_dir}")
                self._append_system_output(f"  - message_history.json ({len(history)} messages)")
                self._append_system_output("  - context_state.json (restored)")
            else:
                self._append_system_output(f"Session loaded from {load_dir}")
                self._append_system_output(f"  - message_history.json ({len(history)} messages)")
                self._append_system_output("  - context_state.json (not found, skipped)")

            self._append_system_output("Next message will continue from loaded history.")
        except Exception as e:
            self._append_system_output(f"Error loading session: {e}")

    def _auto_save_history(self) -> None:
        """Auto-save session to project-specific directory.

        Saves message history and context state to:
        ~/.config/youware-labs/paintress-cli/message_history/{project_hash}/

        This is called automatically after each agent run completes.
        """
        if not self._message_history:
            return

        save_dir = self.config_manager.get_auto_save_dir()
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save message history
        history_file = save_dir / "message_history.json"
        history_file.write_bytes(ModelMessagesTypeAdapter.dump_json(self._message_history, indent=2))

        # Save context state
        state_file = save_dir / "context_state.json"
        state = self.runtime.ctx.export_state()
        state_file.write_text(state.model_dump_json(indent=2))

        logger.debug(f"Auto-saved session to {save_dir}")

    def _append_system_output(self, text: str) -> None:
        """Append system message to output."""
        sys_text = Text()
        sys_text.append("[SYS] ", style="bold yellow")
        sys_text.append(text)
        self._append_output(self._renderer.render(sys_text).rstrip())

    # =========================================================================
    # Main Run Loop
    # =========================================================================

    async def run(self) -> None:
        """Run the TUI application."""
        # Welcome message
        title = Text("Paintress CLI", style="bold magenta")
        self._append_output(self._renderer.render(title).rstrip())
        self._append_output(f"Model: {self.config.general.model}")
        self._append_output(f"Mode: {self._mode.value.upper()}")
        self._append_output(f"Config: {self.config_manager.config_dir}")
        self._append_output("")  # blank line before help
        self._show_help()

        # Hint about auto-saved session if available
        auto_save_dir = self.config_manager.get_auto_save_dir()
        if (auto_save_dir / "message_history.json").exists():
            self._append_output("")
            hint = Text("Previous session found. Use /load to restore.", style="dim")
            self._append_output(self._renderer.render(hint).rstrip())

        # Create scrollable FormattedTextControl with mouse support
        tui_ref = self

        class ScrollableFormattedTextControl(FormattedTextControl):
            """FormattedTextControl that handles mouse scroll events."""

            def mouse_handler(self, mouse_event: MouseEvent) -> object:
                """Handle mouse scroll events."""
                if tui_ref._output_window:
                    if mouse_event.event_type == MouseEventType.SCROLL_UP:
                        tui_ref._output_window.vertical_scroll = max(0, tui_ref._output_window.vertical_scroll - 3)
                        return None
                    elif mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                        max_scroll = tui_ref._get_max_scroll()
                        new_scroll = tui_ref._output_window.vertical_scroll + 3
                        tui_ref._output_window.vertical_scroll = min(new_scroll, max_scroll)
                        return None
                return super().mouse_handler(mouse_event)

        # Create output control and window
        output_control = ScrollableFormattedTextControl(self._get_output_text)
        output_inner_window = Window(
            content=output_control,
            wrap_lines=False,
        )
        self._output_window = ScrollablePane(output_inner_window)

        # Steering pane
        steering_control = FormattedTextControl(self._get_steering_text)
        steering_window = Window(
            content=steering_control,
            height=self._get_steering_height,
            style="class:steering-pane",
            wrap_lines=True,
        )

        # Status bar
        status_bar = Window(
            content=FormattedTextControl(self._get_status_text),
            height=2,
            style="class:status-bar",
            wrap_lines=True,
        )

        # Input area with mouse scroll support
        class ScrollableBufferControl(BufferControl):
            """BufferControl that handles mouse scroll events for input area."""

            def mouse_handler(self, mouse_event: MouseEvent) -> object:
                """Handle mouse scroll events to scroll input text."""
                # Get the window that contains this control
                if mouse_event.event_type == MouseEventType.SCROLL_UP:
                    # Move cursor up by 1 line to scroll content
                    buff = self.buffer
                    if buff:
                        doc = buff.document
                        if doc.cursor_position_row > 0:
                            buff.cursor_up()
                        return None
                elif mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                    buff = self.buffer
                    if buff:
                        doc = buff.document
                        if doc.cursor_position_row < doc.line_count - 1:
                            buff.cursor_down()
                        return None
                return super().mouse_handler(mouse_event)

        input_area = TextArea(
            multiline=True,
            prompt=self._get_prompt,
            style="class:input-area",
            focusable=True,
            height=5,
            scrollbar=True,
        )

        # Replace the buffer control with our scrollable version
        original_control = input_area.control
        scrollable_control = ScrollableBufferControl(
            buffer=original_control.buffer,
            input_processors=original_control.input_processors,
            include_default_input_processors=False,
            lexer=original_control.lexer,
            focus_on_click=original_control.focus_on_click,
        )
        input_area.window.content = scrollable_control
        input_area.control = scrollable_control

        # Layout: Output | Steering | Status | Input
        layout = Layout(
            HSplit([
                self._output_window,
                steering_window,
                status_bar,
                input_area,
            ]),
            focused_element=input_area,
        )

        # Key bindings
        kb = self._setup_keybindings(input_area)

        # Create application
        self._app = Application(
            layout=layout,
            key_bindings=kb,
            style=self._setup_style(),
            full_screen=True,
            mouse_support=True,
        )

        # Run with error handling
        try:
            await self._app.run_async()
        except Exception as e:
            # Re-raise to be caught by cli.py with proper error display
            raise RuntimeError(f"TUI crashed: {e}") from e
        finally:
            # Ensure agent task is fully cancelled and awaited before __aexit__
            # This prevents async generator cleanup issues with MCP servers
            if self._agent_task and not self._agent_task.done():
                self._agent_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._agent_task
                self._agent_task = None
