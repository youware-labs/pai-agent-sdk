"""Fixtures for toolset tests."""

from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext


@pytest.fixture
def mock_ctx() -> AgentContext:
    """Create a mock AgentContext for testing."""
    return AgentContext(run_id="test-run-123")


@pytest.fixture
def mock_run_context(mock_ctx: AgentContext) -> RunContext[AgentContext]:
    """Create a mock RunContext."""
    ctx = MagicMock(spec=RunContext)
    ctx.deps = mock_ctx
    return ctx
