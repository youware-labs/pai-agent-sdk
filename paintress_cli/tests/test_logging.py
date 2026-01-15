"""Tests for logging module."""

from __future__ import annotations

import asyncio
import logging

from paintress_cli.logging import (
    LOG_FILE_NAME,
    SDK_LOGGER_NAME,
    TUI_LOGGER_NAME,
    LogEvent,
    QueueHandler,
    configure_logging,
    configure_tui_logging,
    get_logger,
    reset_logging,
)


class TestLogEvent:
    """Tests for LogEvent."""

    def test_create_event(self):
        """Test creating a log event."""
        event = LogEvent(
            event_id="log-123",
            level="INFO",
            logger_name="test",
            message="Test message",
            func_name="test_func",
            line_no=42,
        )
        assert event.level == "INFO"
        assert event.message == "Test message"
        assert event.line_no == 42

    def test_inherits_from_agent_event(self):
        """Test that LogEvent inherits from AgentEvent."""
        from pai_agent_sdk.events import AgentEvent

        event = LogEvent(event_id="test")
        assert isinstance(event, AgentEvent)


class TestQueueHandler:
    """Tests for QueueHandler."""

    def test_emit_to_queue(self):
        """Test that handler emits to queue."""
        queue: asyncio.Queue = asyncio.Queue()
        handler = QueueHandler(queue)
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        assert not queue.empty()
        event = queue.get_nowait()
        assert isinstance(event, LogEvent)
        assert event.level == "INFO"
        assert event.message == "Test message"

    def test_respects_level(self):
        """Test that handler respects log level."""
        queue: asyncio.Queue = asyncio.Queue()
        handler = QueueHandler(queue, level=logging.WARNING)

        # Verify handler level is correctly set
        assert handler.level == logging.WARNING


class TestConfigureTuiLogging:
    """Tests for configure_tui_logging."""

    def teardown_method(self):
        """Reset logging after each test."""
        reset_logging()

    def test_configures_tui_logger(self):
        """Test TUI logger is configured."""
        queue: asyncio.Queue = asyncio.Queue()
        configure_tui_logging(queue)

        logger = logging.getLogger(TUI_LOGGER_NAME)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], QueueHandler)

    def test_configures_sdk_logger(self):
        """Test SDK logger is configured."""
        queue: asyncio.Queue = asyncio.Queue()
        configure_tui_logging(queue)

        logger = logging.getLogger(SDK_LOGGER_NAME)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], QueueHandler)

    def test_verbose_adds_file_handler(self, tmp_path, monkeypatch):
        """Test verbose mode adds file handler."""
        monkeypatch.chdir(tmp_path)
        queue: asyncio.Queue = asyncio.Queue()
        configure_tui_logging(queue, verbose=True)

        logger = logging.getLogger(TUI_LOGGER_NAME)
        # Should have both QueueHandler and FileHandler
        assert len(logger.handlers) == 2
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "QueueHandler" in handler_types
        assert "FileHandler" in handler_types

        # Log file should be created
        log_file = tmp_path / LOG_FILE_NAME
        logger.info("Test verbose log")
        assert log_file.exists()

    def test_logs_go_to_queue(self):
        """Test that log messages appear in queue."""
        queue: asyncio.Queue = asyncio.Queue()
        configure_tui_logging(queue)

        logger = get_logger("test")
        logger.info("Test message")

        assert not queue.empty()
        event = queue.get_nowait()
        assert isinstance(event, LogEvent)
        assert "Test message" in event.message

    def test_sdk_logs_go_to_queue(self):
        """Test that SDK log messages appear in queue."""
        queue: asyncio.Queue = asyncio.Queue()
        configure_tui_logging(queue)

        sdk_logger = logging.getLogger(SDK_LOGGER_NAME)
        sdk_logger.info("SDK message")

        assert not queue.empty()
        event = queue.get_nowait()
        assert "SDK message" in event.message

    def test_idempotent(self):
        """Test that configure can be called multiple times."""
        queue: asyncio.Queue = asyncio.Queue()
        configure_tui_logging(queue)
        configure_tui_logging(queue)  # Should not error

        logger = logging.getLogger(TUI_LOGGER_NAME)
        assert len(logger.handlers) == 1


class TestGetLogger:
    """Tests for get_logger."""

    def teardown_method(self):
        """Reset logging after each test."""
        reset_logging()

    def test_prefixes_name(self):
        """Test that logger name is prefixed."""
        logger = get_logger("mymodule")
        assert logger.name == f"{TUI_LOGGER_NAME}.mymodule"

    def test_already_prefixed(self):
        """Test that already prefixed names are unchanged."""
        logger = get_logger(f"{TUI_LOGGER_NAME}.other")
        assert logger.name == f"{TUI_LOGGER_NAME}.other"


class TestResetLogging:
    """Tests for reset_logging."""

    def test_clears_handlers(self):
        """Test that reset clears handlers."""
        queue: asyncio.Queue = asyncio.Queue()
        configure_tui_logging(queue)

        tui_logger = logging.getLogger(TUI_LOGGER_NAME)
        sdk_logger = logging.getLogger(SDK_LOGGER_NAME)
        assert len(tui_logger.handlers) == 1
        assert len(sdk_logger.handlers) == 1

        reset_logging()

        assert len(tui_logger.handlers) == 0
        assert len(sdk_logger.handlers) == 0


class TestConfigureLogging:
    """Tests for configure_logging (CLI startup logging)."""

    def teardown_method(self):
        """Reset logging after each test."""
        reset_logging()

    def test_silent_mode(self):
        """Test non-verbose mode uses NullHandler."""
        configure_logging(verbose=False)

        logger = logging.getLogger(TUI_LOGGER_NAME)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.NullHandler)

    def test_verbose_mode_creates_file(self, tmp_path, monkeypatch):
        """Test verbose mode creates log file."""
        monkeypatch.chdir(tmp_path)
        configure_logging(verbose=True)

        logger = logging.getLogger(TUI_LOGGER_NAME)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.FileHandler)

        # Log something and verify file is created
        logger.debug("Test log message")
        log_file = tmp_path / LOG_FILE_NAME
        assert log_file.exists()

        content = log_file.read_text()
        assert "Test log message" in content
