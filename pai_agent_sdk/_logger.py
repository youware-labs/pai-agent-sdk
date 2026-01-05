"""Centralized logging configuration for pai-agent-sdk.

Usage:
    from pai_agent_sdk._logger import get_logger

    logger = get_logger(__name__)
    logger.info("message")

Configuration:
    Environment variables:
    - PAI_AGENT_LOG_LEVEL: Global log level for all SDK modules (default: WARNING)
    - PAI_AGENT_LOG_LEVEL_<MODULE>: Module-specific log level override

    Examples:
    - PAI_AGENT_LOG_LEVEL=INFO                     # All SDK logs at INFO level
    - PAI_AGENT_LOG_LEVEL_BROWSER_USE=DEBUG        # browser_use module at DEBUG
    - PAI_AGENT_LOG_LEVEL_TOOLSETS=ERROR           # toolsets module at ERROR
"""

from __future__ import annotations

import logging
import os
import sys
from typing import ClassVar

# SDK root logger name
LOGGER_NAME = "pai_agent_sdk"

# Cache for already configured loggers
_configured_loggers: set[str] = set()


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output."""

    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
        "GREEN": "\033[32m",  # Green for timestamp
        "CYAN": "\033[36m",  # Cyan for location
    }

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        colored_timestamp = f"{self.COLORS['GREEN']}{timestamp}{self.COLORS['RESET']}"

        level = record.levelname
        colored_level = f"{self.COLORS.get(level, '')}{level:<8}{self.COLORS['RESET']}"

        # Shorten logger name for readability
        name = record.name
        if name.startswith("pai_agent_sdk."):
            name = name[14:]  # Remove "pai_agent_sdk." prefix

        location = f"{self.COLORS['CYAN']}{name}:{record.funcName}:{record.lineno}{self.COLORS['RESET']}"
        colored_message = f"{self.COLORS.get(level, '')}{record.getMessage()}{self.COLORS['RESET']}"

        return f"{colored_timestamp} | {colored_level} | {location} - {colored_message}"


def _get_module_log_level(module_path: str) -> int | None:
    """Get module-specific log level from environment variable.

    Checks for module-specific overrides by walking up the module hierarchy.
    For example, for "toolsets.browser_use.tools", checks:
    - PAI_AGENT_LOG_LEVEL_TOOLSETS_BROWSER_USE_TOOLS
    - PAI_AGENT_LOG_LEVEL_TOOLSETS_BROWSER_USE
    - PAI_AGENT_LOG_LEVEL_TOOLSETS

    Args:
        module_path: Dotted module path relative to pai_agent_sdk (e.g., "toolsets.browser_use")

    Returns:
        Log level if found, None otherwise.
    """
    parts = module_path.upper().replace(".", "_").split("_")

    # Check from most specific to least specific
    for i in range(len(parts), 0, -1):
        env_var = f"PAI_AGENT_LOG_LEVEL_{'_'.join(parts[:i])}"
        level_name = os.getenv(env_var)
        if level_name:
            level = getattr(logging, level_name.upper(), None)
            if level is not None:
                return level

    return None


def _setup_sdk_logger() -> None:
    """Setup the root logger for pai-agent-sdk with handler."""
    if LOGGER_NAME in _configured_loggers:
        return

    log_level_name = os.getenv("PAI_AGENT_LOG_LEVEL", "WARNING").upper()
    log_level = getattr(logging, log_level_name, logging.WARNING)

    sdk_logger = logging.getLogger(LOGGER_NAME)
    sdk_logger.setLevel(log_level)
    sdk_logger.propagate = False

    # Only add handler if not already configured
    if not sdk_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)  # Handler accepts all; logger filters

        if sys.stderr.isatty():
            formatter = ColoredFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        handler.setFormatter(formatter)
        sdk_logger.addHandler(handler)

    _configured_loggers.add(LOGGER_NAME)


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance for the given module.

    Args:
        name: Full module name (e.g., "pai_agent_sdk.toolsets.browser_use")
              or relative name (e.g., "toolsets.browser_use").
              If None, returns the SDK root logger.

    Returns:
        A configured logger instance.

    Example:
        # In pai_agent_sdk/toolsets/browser_use/toolset.py
        from pai_agent_sdk._logger import get_logger
        logger = get_logger(__name__)  # "pai_agent_sdk.toolsets.browser_use.toolset"
    """
    # Ensure root logger is set up
    _setup_sdk_logger()

    if name is None:
        return logging.getLogger(LOGGER_NAME)

    # Normalize name to full path
    full_name = f"{LOGGER_NAME}.{name}" if not name.startswith(LOGGER_NAME) else name

    module_logger = logging.getLogger(full_name)

    # Apply module-specific log level if configured
    if full_name not in _configured_loggers:
        relative_name = full_name[len(LOGGER_NAME) + 1 :] if full_name.startswith(f"{LOGGER_NAME}.") else full_name
        module_level = _get_module_log_level(relative_name)
        if module_level is not None:
            module_logger.setLevel(module_level)
        _configured_loggers.add(full_name)

    return module_logger


# Initialize SDK root logger on import
_setup_sdk_logger()

# Convenience: SDK root logger
logger = get_logger()

__all__ = ["LOGGER_NAME", "get_logger", "logger"]
