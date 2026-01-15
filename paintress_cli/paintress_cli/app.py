"""TUI Application for paintress-cli.

This module provides the main TUI application with:
- prompt_toolkit based UI with dual-pane layout
- Agent execution with streaming output
- Steering message injection during execution
- Mode switching (ACT/PLAN) with Ctrl+P/Ctrl+A keybindings

Example:
    from paintress_cli.app import TUIApp

    async with TUIApp(config, config_manager) as app:
        await app.run()
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

from prompt_toolkit import Application
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from rich.text import Text

from pai_agent_sdk.agents.main import AgentRuntime, stream_agent
from pai_agent_sdk.context import StreamEvent
from paintress_cli.browser import BrowserManager
from paintress_cli.config import ConfigManager, PaintressConfig
from paintress_cli.display import EventRenderer, RichRenderer, ToolMessage
from paintress_cli.logging import get_logger
from paintress_cli.runtime import create_tui_runtime
from paintress_cli.session import TUIContext

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

    # Runtime state
    _mode: TUIMode = field(default=TUIMode.ACT, init=False)
    _state: TUIState = field(default=TUIState.IDLE, init=False)

    # Resources (initialized in __aenter__)
    _exit_stack: AsyncExitStack | None = field(default=None, init=False, repr=False)
    _browser: BrowserManager | None = field(default=None, init=False)
    _runtime: AgentRuntime[TUIContext, str] | None = field(default=None, init=False)

    # UI components
    _app: Application[None] | None = field(default=None, init=False, repr=False)
    _output_lines: list[str] = field(default_factory=list, init=False)
    _renderer: RichRenderer = field(default_factory=RichRenderer, init=False)
    _event_renderer: EventRenderer = field(default_factory=EventRenderer, init=False)

    # Agent execution
    _agent_task: asyncio.Task[None] | None = field(default=None, init=False)

    # Tool tracking
    _tool_messages: dict[str, ToolMessage] = field(default_factory=dict, init=False)
    _printed_tool_calls: set[str] = field(default_factory=set, init=False)

    @property
    def mode(self) -> TUIMode:
        """Current agent mode."""
        return self._mode

    @property
    def state(self) -> TUIState:
        """Current application state."""
        return self._state

    @property
    def runtime(self) -> AgentRuntime[TUIContext, str]:
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

        logger.info("TUIApp initialized")
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

        if self._exit_stack:
            result = await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
            self._exit_stack = None
            return result
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

    def _append_output(self, text: str) -> None:
        """Append text to output buffer."""
        self._output_lines.append(text)

    def _get_output_text(self) -> ANSI:
        """Get formatted output for display."""
        return ANSI("\n".join(self._output_lines))

    # =========================================================================
    # Agent Execution
    # =========================================================================

    async def _run_agent(self, user_prompt: str) -> None:
        """Execute agent with user prompt."""
        self._state = TUIState.RUNNING
        self._tool_messages.clear()
        self._printed_tool_calls.clear()
        self._event_renderer.clear()

        try:
            async with stream_agent(
                self.runtime,
                user_prompt=user_prompt,
            ) as stream:
                async for event in stream:
                    self._handle_stream_event(event)

                stream.raise_if_exception()

        except asyncio.CancelledError:
            self._append_output("\n[Cancelled]")
            raise
        except Exception as e:
            self._append_output(f"\n[Error: {e}]")
            logger.exception("Agent execution failed")
        finally:
            self._state = TUIState.IDLE
            if self._app:
                self._app.invalidate()

    def _handle_stream_event(self, event: StreamEvent) -> None:
        """Handle a stream event from agent execution."""
        message_event = event.event

        if isinstance(message_event, PartStartEvent) and isinstance(message_event.part, TextPart):
            # Start of text - append new line
            self._append_output(message_event.part.content)

        elif isinstance(message_event, PartDeltaEvent) and isinstance(message_event.delta, TextPartDelta):
            # Streaming text delta - append to last line
            if self._output_lines:
                self._output_lines[-1] += message_event.delta.content_delta
            else:
                self._append_output(message_event.delta.content_delta)

        elif isinstance(message_event, FunctionToolCallEvent):
            # Tool call start
            tool_call_id = message_event.part.tool_call_id
            tool_name = message_event.part.tool_name
            self._tool_messages[tool_call_id] = ToolMessage(
                tool_call_id=tool_call_id,
                name=tool_name,
                args=message_event.part.args,
            )
            self._event_renderer.tracker.start_call(tool_call_id, tool_name, message_event.part.args)
            self._append_output(f"[Tool] {tool_name}...")

        elif isinstance(message_event, FunctionToolResultEvent):
            # Tool result
            tool_call_id = message_event.tool_call_id
            if tool_call_id in self._tool_messages:
                tool_msg = self._tool_messages[tool_call_id]
                # Extract result content
                result_content = self._extract_tool_result(message_event)
                tool_msg.content = result_content
                self._event_renderer.tracker.complete_call(tool_call_id, result_content)

                # Render completed tool call
                if tool_call_id not in self._printed_tool_calls:
                    rendered = self._event_renderer.render_tool_call_complete(tool_msg)
                    self._append_output(rendered.rstrip())
                    self._printed_tool_calls.add(tool_call_id)

        # Refresh UI
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

    def _setup_keybindings(self) -> KeyBindings:
        """Set up keyboard bindings."""
        kb = KeyBindings()

        @kb.add("c-c")
        def handle_ctrl_c(event: KeyPressEvent) -> None:
            """Handle Ctrl+C - cancel or exit."""
            if self._state == TUIState.RUNNING and self._agent_task:
                self._agent_task.cancel()
            else:
                event.app.exit()

        @kb.add("c-d")
        def handle_ctrl_d(event: KeyPressEvent) -> None:
            """Handle Ctrl+D - exit."""
            event.app.exit()

        @kb.add("c-p")
        def handle_ctrl_p(event: KeyPressEvent) -> None:
            """Handle Ctrl+P - switch to PLAN mode."""
            self.switch_mode(TUIMode.PLAN)

        @kb.add("c-a")
        def handle_ctrl_a(event: KeyPressEvent) -> None:
            """Handle Ctrl+A - switch to ACT mode."""
            self.switch_mode(TUIMode.ACT)

        return kb

    def _setup_style(self) -> Style:
        """Set up UI styles."""
        return Style.from_dict({
            "status-bar": "bg:#333333 #ffffff",
            "status-bar.mode": "bold",
            "status-bar.mode-act": "bg:#2e7d32 #ffffff bold",
            "status-bar.mode-plan": "bg:#1565c0 #ffffff bold",
        })

    def _get_status_bar_text(self) -> str:
        """Generate status bar text."""
        mode = self.mode.value.upper()
        state = "Running..." if self.state == TUIState.RUNNING else "Ready"
        model = self.config.general.model or "No model"
        browser = "[B]" if self._browser and self._browser.is_available else ""

        return f" [{mode}] | {model} | {state} {browser} | Ctrl+P: Plan | Ctrl+A: Act"

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
        self._append_output("Type your message and press Enter. Ctrl+C to cancel, Ctrl+D to exit.")
        self._append_output("Ctrl+P: Plan mode | Ctrl+A: Act mode\n")

        # Create input area
        input_area = TextArea(
            height=3,
            prompt="> ",
            multiline=False,
        )

        # Create output window
        output_window = Window(
            content=FormattedTextControl(self._get_output_text),
            wrap_lines=True,
        )

        # Create status bar
        status_bar = Window(
            content=FormattedTextControl(lambda: self._get_status_bar_text()),
            height=1,
            style="class:status-bar",
        )

        # Layout
        layout = Layout(
            HSplit([
                output_window,
                status_bar,
                input_area,
            ])
        )

        # Key bindings
        kb = self._setup_keybindings()

        @kb.add("enter")
        def handle_enter(event: KeyPressEvent) -> None:
            """Handle Enter - submit input."""
            if self._state == TUIState.RUNNING:
                # TODO: Add to steering queue
                return

            buffer = input_area.buffer
            text = buffer.text.strip()
            if text:
                buffer.reset()
                self._append_output(f"> {text}")
                # Start agent in background
                self._agent_task = asyncio.create_task(self._run_agent(text))

        # Create application
        self._app = Application(
            layout=layout,
            key_bindings=kb,
            style=self._setup_style(),
            full_screen=True,
            mouse_support=True,
        )

        # Run
        await self._app.run_async()
