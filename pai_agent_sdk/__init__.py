"""pai-agent-sdk: Production-ready SDK for building AI agents with Pydantic AI."""

import importlib.metadata

from pai_agent_sdk.mcp import MCPServerSpec, ProcessToolCallback, create_mcp_approval_hook
from pai_agent_sdk.media import S3MediaConfig, S3MediaUploader, create_s3_media_hook
from pai_agent_sdk.usage import ExtraUsageRecord, InternalUsage

__all__ = [
    "ExtraUsageRecord",
    "InternalUsage",
    "MCPServerSpec",
    "ProcessToolCallback",
    "S3MediaConfig",
    "S3MediaUploader",
    "__version__",
    "create_mcp_approval_hook",
    "create_s3_media_hook",
]

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development mode
