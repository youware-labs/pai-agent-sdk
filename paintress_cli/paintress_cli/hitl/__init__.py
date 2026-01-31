"""Human-in-the-Loop module for paintress-cli.

This module provides HITL (Human-in-the-Loop) approval workflow components:
- ApprovalManager: Manages the async approval flow
- ApprovalUI: Renders approval panels and results
- ApprovalResult: Data class for approval decisions
- ApprovalState: Data class for approval workflow state

Example:
    from paintress_cli.hitl import ApprovalManager, ApprovalUI

    manager = ApprovalManager()
    ui = ApprovalUI(renderer)

    # In the agent loop
    if isinstance(result.output, DeferredToolRequests):
        results = await manager.collect_approvals(
            result.output,
            on_display=lambda tc, idx, total: print(ui.render_approval_panel(tc, idx, total)),
        )
"""

from __future__ import annotations

from paintress_cli.hitl.approval import ApprovalManager, ApprovalResult, ApprovalState
from paintress_cli.hitl.ui import ApprovalUI

__all__ = [
    "ApprovalManager",
    "ApprovalResult",
    "ApprovalState",
    "ApprovalUI",
]
