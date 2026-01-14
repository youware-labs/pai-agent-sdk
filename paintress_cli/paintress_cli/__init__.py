"""Paintress CLI - TUI reference implementation for pai-agent-sdk."""

import importlib.metadata

__all__ = ["__version__"]

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
