"""Fixtures for subagent tests."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment


@pytest.fixture
def agent_context(tmp_path: Path) -> AgentContext:
    """Create a simple AgentContext for synchronous tests.

    This fixture creates a minimal AgentContext without async context manager,
    suitable for testing synchronous tool availability checks.
    """
    env = LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    )
    return AgentContext(env=env)


@pytest.fixture
def mock_run_ctx(agent_context: AgentContext) -> MagicMock:
    """Create a mock RunContext for testing.

    This fixture provides a MagicMock spec'd to RunContext with deps set to agent_context.
    """
    mock_ctx = MagicMock(spec=RunContext)
    mock_ctx.deps = agent_context
    return mock_ctx
