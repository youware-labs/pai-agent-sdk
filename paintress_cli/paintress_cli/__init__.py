"""Paintress CLI - TUI reference implementation for pai-agent-sdk."""

import importlib.metadata

__all__ = ["__version__", "cli"]

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development mode


def cli() -> None:
    """Run the paintress CLI and exit."""
    # TODO: Implement TUI
    print(f"paintress-cli v{__version__}")
    print("TUI implementation coming soon...")
