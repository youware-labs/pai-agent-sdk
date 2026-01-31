"""Human-in-the-Loop approval management.

Provides ApprovalManager for handling tool approval workflows.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic_ai import DeferredToolRequests, DeferredToolResults, ToolDenied

if TYPE_CHECKING:
    from pydantic_ai.messages import ToolCallPart


@dataclass
class ApprovalResult:
    """Result of a tool approval decision."""

    approved: bool
    reason: str | None = None


@dataclass
class ApprovalState:
    """Current state of the approval workflow."""

    pending_approvals: list[ToolCallPart] = field(default_factory=list)
    current_index: int = 0
    is_pending: bool = False


class ApprovalManager:
    """Manages Human-in-the-Loop tool approval workflow.

    Handles the async approval flow:
    1. Receives DeferredToolRequests with tools needing approval
    2. Presents each tool for user decision
    3. Collects approval/rejection decisions
    4. Returns DeferredToolResults for agent continuation
    """

    def __init__(self) -> None:
        """Initialize ApprovalManager."""
        self._state = ApprovalState()
        self._event: asyncio.Event | None = None
        self._result: ApprovalResult | None = None

    @property
    def is_pending(self) -> bool:
        """Check if waiting for user approval."""
        return self._state.is_pending

    @property
    def pending_approvals(self) -> list[ToolCallPart]:
        """Get list of tools pending approval."""
        return self._state.pending_approvals

    @property
    def current_index(self) -> int:
        """Get index of current tool being reviewed."""
        return self._state.current_index

    @property
    def current_tool(self) -> ToolCallPart | None:
        """Get the tool currently being reviewed."""
        if not self._state.pending_approvals:
            return None
        if self._state.current_index >= len(self._state.pending_approvals):
            return None
        return self._state.pending_approvals[self._state.current_index]

    @property
    def total_count(self) -> int:
        """Get total number of tools pending approval."""
        return len(self._state.pending_approvals)

    def start_approval_flow(self, deferred: DeferredToolRequests) -> bool:
        """Start a new approval flow.

        Args:
            deferred: DeferredToolRequests containing tools needing approval.

        Returns:
            True if there are approvals to process, False otherwise.
        """
        if not deferred.approvals:
            return False

        self._state = ApprovalState(
            pending_approvals=list(deferred.approvals),
            current_index=0,
            is_pending=True,
        )
        return True

    async def wait_for_decision(self) -> ApprovalResult:
        """Wait for user to make an approval decision.

        Returns:
            ApprovalResult with the decision.
        """
        self._event = asyncio.Event()
        self._result = None

        await self._event.wait()

        result = self._result if self._result else ApprovalResult(approved=False, reason="Cancelled")
        self._event = None
        return result

    def approve(self) -> None:
        """Approve the current tool."""
        self._result = ApprovalResult(approved=True)
        if self._event:
            self._event.set()

    def reject(self, reason: str | None = None) -> None:
        """Reject the current tool.

        Args:
            reason: Optional reason for rejection.
        """
        self._result = ApprovalResult(approved=False, reason=reason)
        if self._event:
            self._event.set()

    def advance(self) -> bool:
        """Advance to the next tool.

        Returns:
            True if there are more tools, False if done.
        """
        self._state.current_index += 1
        return self._state.current_index < len(self._state.pending_approvals)

    def reset(self) -> None:
        """Reset approval state.

        Should be called after approval flow completes or on cancellation.
        """
        # Signal any waiting coroutine
        if self._event and not self._event.is_set():
            self._result = ApprovalResult(approved=False, reason="Cancelled")
            self._event.set()

        self._state = ApprovalState()
        self._event = None
        self._result = None

    async def collect_approvals(
        self,
        deferred: DeferredToolRequests,
        on_display: Callable[[ToolCallPart, int, int], None] | None = None,
        on_result: Callable[[ToolCallPart, ApprovalResult], None] | None = None,
    ) -> DeferredToolResults:
        """Collect all approval decisions for deferred tool requests.

        This is the main entry point for the approval workflow.

        Args:
            deferred: DeferredToolRequests with tools needing approval.
            on_display: Callback to display approval UI for a tool.
            on_result: Callback when a decision is made.

        Returns:
            DeferredToolResults with all approval decisions.
        """
        results = DeferredToolResults()

        if not self.start_approval_flow(deferred):
            return results

        try:
            for idx, tool_call in enumerate(deferred.approvals):
                self._state.current_index = idx

                # Display the approval UI
                if on_display:
                    on_display(tool_call, idx + 1, len(deferred.approvals))

                # Wait for decision
                decision = await self.wait_for_decision()

                # Record result
                if decision.approved:
                    results.approvals[tool_call.tool_call_id] = True
                else:
                    reason = decision.reason or "User rejected"
                    results.approvals[tool_call.tool_call_id] = ToolDenied(reason)

                # Notify callback
                if on_result:
                    on_result(tool_call, decision)

        finally:
            self._state.is_pending = False

        return results
