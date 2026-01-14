"""Paintress CLI - TUI for AI agents."""

from __future__ import annotations

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from paintress_cli.environment import TUIEnvironment
from paintress_cli.processes import (
    PROCESS_MANAGER_KEY,
    ManagedProcess,
    ProcessInfo,
    ProcessManager,
    create_process_manager,
)
from paintress_cli.toolsets import (
    KillProcessTool,
    ListProcessesTool,
    ProcessToolBase,
    ReadProcessOutputTool,
    SpawnProcessTool,
    process_tools,
)

__all__ = [
    "PROCESS_MANAGER_KEY",
    "KillProcessTool",
    "ListProcessesTool",
    "ManagedProcess",
    "ProcessInfo",
    "ProcessManager",
    "ProcessToolBase",
    "ReadProcessOutputTool",
    "SpawnProcessTool",
    "TUIEnvironment",
    "__version__",
    "create_process_manager",
    "process_tools",
]
