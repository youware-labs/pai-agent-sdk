"""pai-agent-sdk: Production-ready SDK for building AI agents with Pydantic AI."""

import importlib.metadata

__all__ = ["__version__"]

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development mode
