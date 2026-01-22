"""pai-agent-sdk: Production-ready SDK for building AI agents with Pydantic AI."""

import importlib.metadata

from pai_agent_sdk.mcp import MCPServerSpec, ProcessToolCallback, create_mcp_approval_hook
from pai_agent_sdk.usage import ExtraUsageRecord, InternalUsage

__all__ = [
    "ExtraUsageRecord",
    "InternalUsage",
    "MCPServerSpec",
    "ProcessToolCallback",
    "__version__",
    "create_mcp_approval_hook",
]

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development mode
