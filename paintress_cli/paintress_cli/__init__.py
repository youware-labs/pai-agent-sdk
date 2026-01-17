"""Paintress CLI - TUI for AI agents."""

from __future__ import annotations

import importlib.metadata
import logging


def _configure_logging() -> None:
    """Configure logging to suppress noisy third-party logs.

    Suppresses:
    - asyncio ERROR logs during async generator cleanup (pydantic-ai/mcp issue)
    - MCP INFO logs (ListToolsRequest, etc.)
    """
    # Configure root logger first to prevent MCP's basicConfig from taking effect
    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s: %(message)s",
        )

    # Suppress asyncio async generator cleanup errors
    # These occur during shutdown and are harmless but noisy
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    # Suppress mcp INFO logs
    logging.getLogger("mcp").setLevel(logging.WARNING)


_configure_logging()

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
