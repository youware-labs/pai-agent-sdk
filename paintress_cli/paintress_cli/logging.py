"""Logging configuration for paintress-cli TUI.

TUI logging redirects all log output to a queue-based handler that can be
consumed by TUI components for display. This prevents log output from
interfering with TUI rendering.

Key components:
- LogEvent: Event type for log messages (extends AgentEvent)
- QueueHandler: Logging handler that emits LogEvents to a queue
- configure_tui_logging(): Setup function to redirect all loggers to TUI

Usage:
    from paintress_cli.logging import configure_tui_logging, LogEvent

    # At TUI startup
    log_queue = asyncio.Queue()
    configure_tui_logging(log_queue)

    # Log messages now appear as LogEvents in the queue
    # TUI can consume and display them
"""

from __future__ import annotations

import logging
from asyncio import Queue
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pai_agent_sdk.events import AgentEvent

if TYPE_CHECKING:
    pass

# Logger names to configure
TUI_LOGGER_NAME = "paintress_cli"
SDK_LOGGER_NAME = "pai_agent_sdk"

# Cache for initialization state
_initialized = False
_log_queue: Queue | None = None


# -----------------------------------------------------------------------------
# Log Event
# -----------------------------------------------------------------------------


@dataclass
class LogEvent(AgentEvent):
    """Log message event for TUI display.

    Emitted by QueueHandler when a log record is produced.

    Attributes:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        logger_name: Name of the logger that produced the message.
        message: Formatted log message.
        func_name: Function name where log was called.
        line_no: Line number where log was called.
    """

    level: str = "INFO"
    logger_name: str = ""
    message: str = ""
    func_name: str = ""
    line_no: int = 0


# -----------------------------------------------------------------------------
# Queue Handler
# -----------------------------------------------------------------------------


class QueueHandler(logging.Handler):
    """Logging handler that emits LogEvents to a queue.

    This handler formats log records and puts them into an asyncio Queue
    as LogEvent instances. TUI components can consume these events for
    display in a log panel.
    """

    def __init__(self, queue: Queue, level: int = logging.DEBUG) -> None:
        """Initialize the queue handler.

        Args:
            queue: Asyncio queue to emit events to.
            level: Minimum log level to handle.
        """
        super().__init__(level)
        self._queue = queue

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record as a LogEvent.

        Args:
            record: The log record to emit.
        """
        try:
            # Format the message
            msg = self.format(record)

            # Create event
            event = LogEvent(
                event_id=f"log-{record.created:.0f}-{record.lineno}",
                level=record.levelname,
                logger_name=record.name,
                message=msg,
                func_name=record.funcName,
                line_no=record.lineno,
            )

            # Put into queue (non-blocking)
            self._queue.put_nowait(event)
        except Exception:
            # Don't raise exceptions from logging
            self.handleError(record)


# -----------------------------------------------------------------------------
# Configuration Functions
# -----------------------------------------------------------------------------


def _configure_logger(name: str, queue: Queue, level: int = logging.DEBUG) -> None:
    """Configure a logger to use the queue handler.

    Args:
        name: Logger name.
        queue: Queue to emit events to.
        level: Minimum log level.
    """
    logger = logging.getLogger(name)

    # Clear existing handlers (especially stderr handlers)
    logger.handlers.clear()

    # Add queue handler
    handler = QueueHandler(queue, level)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Set level and prevent propagation
    logger.setLevel(level)
    logger.propagate = False


def configure_tui_logging(
    queue: Queue,
    level: int = logging.INFO,
) -> None:
    """Configure logging for TUI mode.

    Redirects both paintress_cli and pai_agent_sdk loggers to emit
    LogEvents to the provided queue. This prevents log output from
    appearing on stderr and interfering with TUI display.

    Args:
        queue: Asyncio queue to receive LogEvents.
        level: Minimum log level to capture (default: INFO).

    Example:
        log_queue = asyncio.Queue()
        configure_tui_logging(log_queue)

        # Later, consume events
        while True:
            event = await log_queue.get()
            display_log(event)
    """
    global _initialized, _log_queue

    if _initialized:
        return

    _log_queue = queue

    # Configure both loggers
    _configure_logger(TUI_LOGGER_NAME, queue, level)
    _configure_logger(SDK_LOGGER_NAME, queue, level)

    _initialized = True


def reset_logging() -> None:
    """Reset logging configuration.

    Useful for tests or when reconfiguring.
    """
    global _initialized, _log_queue

    for name in [TUI_LOGGER_NAME, SDK_LOGGER_NAME]:
        logger = logging.getLogger(name)
        logger.handlers.clear()

    _initialized = False
    _log_queue = None


def configure_logging(verbose: bool = False) -> None:
    """Configure basic stderr logging for CLI startup.

    This is used before TUI is initialized, for early startup messages.
    Once TUI starts, call configure_tui_logging() to switch to queue mode.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Configure both loggers with stderr handler
    for name in [TUI_LOGGER_NAME, SDK_LOGGER_NAME]:
        logger = logging.getLogger(name)
        logger.handlers.clear()

        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.setLevel(level)
        logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name.

    Args:
        name: Module name (typically __name__).

    Returns:
        Logger instance.
    """
    # Ensure logger is under TUI namespace
    if not name.startswith(TUI_LOGGER_NAME):
        name = f"{TUI_LOGGER_NAME}.{name}"

    return logging.getLogger(name)
