"""Multimodal content processing tools.

Tools for processing images and videos when native model support is unavailable.
"""

from pai_agent_sdk.toolsets.core.base import BaseTool
from pai_agent_sdk.toolsets.core.multimodal.image import ReadImageTool
from pai_agent_sdk.toolsets.core.multimodal.video import ReadVideoTool

tools: list[type[BaseTool]] = [
    ReadImageTool,
    ReadVideoTool,
]

__all__ = [
    "ReadImageTool",
    "ReadVideoTool",
    "tools",
]
