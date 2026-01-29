"""Tests for model_wrapper and wrapper_metadata functionality."""

from inspect import isawaitable
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import Field
from pydantic_ai.models import Model

from pai_agent_sdk.context import AgentContext, ModelWrapper
from pai_agent_sdk.environment.local import LocalEnvironment


@pytest.fixture
async def env() -> LocalEnvironment:
    """Create a test environment."""
    async with LocalEnvironment() as environment:
        yield environment


def test_model_wrapper_type_alias() -> None:
    """ModelWrapper should be a Callable type alias."""
    # Verify ModelWrapper is the expected type
    assert ModelWrapper is not None

    # A simple function should be assignable to ModelWrapper
    def my_wrapper(model: Model, agent_name: str, context: dict[str, Any]) -> Model:
        return model

    wrapper: ModelWrapper = my_wrapper
    assert callable(wrapper)


async def test_context_without_wrapper(env: LocalEnvironment) -> None:
    """Context without wrapper should have model_wrapper as None."""
    async with AgentContext(env=env) as ctx:
        assert ctx.model_wrapper is None
        assert ctx.wrapper_metadata == {}


async def test_context_with_sync_wrapper(env: LocalEnvironment) -> None:
    """Context with sync wrapper should work correctly."""
    wrapped_model = MagicMock(spec=Model)
    call_args: dict[str, Any] = {}

    def my_wrapper(model: Model, agent_name: str, context: dict[str, Any]) -> Model:
        call_args["model"] = model
        call_args["agent_name"] = agent_name
        call_args["context"] = context
        return wrapped_model

    async with AgentContext(env=env, model_wrapper=my_wrapper) as ctx:
        original_model = MagicMock(spec=Model)
        wrapper_metadata = ctx.get_wrapper_metadata()

        result = ctx.model_wrapper(original_model, "test-agent", wrapper_metadata)

        assert result is wrapped_model
        assert call_args["model"] is original_model
        assert call_args["agent_name"] == "test-agent"
        assert call_args["context"]["run_id"] == ctx.run_id


async def test_context_with_async_wrapper(env: LocalEnvironment) -> None:
    """Context with async wrapper should return awaitable."""
    wrapped_model = MagicMock(spec=Model)

    async def async_wrapper(model: Model, agent_name: str, context: dict[str, Any]) -> Model:
        return wrapped_model

    async with AgentContext(env=env, model_wrapper=async_wrapper) as ctx:
        original_model = MagicMock(spec=Model)
        wrapper_metadata = ctx.get_wrapper_metadata()

        result = ctx.model_wrapper(original_model, "test-agent", wrapper_metadata)

        # Result should be awaitable
        assert isawaitable(result)
        final_result = await result
        assert final_result is wrapped_model


async def test_model_wrapper_not_serialized(env: LocalEnvironment) -> None:
    """model_wrapper and wrapper_metadata should be excluded from serialization."""

    def my_wrapper(model: Model, agent_name: str, context: dict[str, Any]) -> Model:
        return model

    async with AgentContext(
        env=env,
        model_wrapper=my_wrapper,
        wrapper_metadata={"trace_id": "abc123"},
    ) as ctx:
        state = ctx.export_state()
        state_dict = state.model_dump()

        # model_wrapper and wrapper_metadata should not appear in exported state
        assert "model_wrapper" not in state_dict
        assert "wrapper_metadata" not in state_dict


async def test_model_wrapper_inherited_in_subagent_context(env: LocalEnvironment) -> None:
    """model_wrapper and wrapper_metadata should be inherited when creating subagent context."""

    def my_wrapper(model: Model, agent_name: str, context: dict[str, Any]) -> Model:
        return model

    async with AgentContext(
        env=env,
        model_wrapper=my_wrapper,
        wrapper_metadata={"trace_id": "abc123", "user_id": "user_456"},
    ) as ctx:
        async with ctx.create_subagent_context("test-subagent") as sub_ctx:
            # Subagent context should inherit model_wrapper
            assert sub_ctx.model_wrapper is my_wrapper
            # Subagent context should inherit wrapper_metadata
            assert sub_ctx.wrapper_metadata == {"trace_id": "abc123", "user_id": "user_456"}


# =============================================================================
# Tests for get_wrapper_metadata
# =============================================================================


async def test_get_wrapper_metadata_default(env: LocalEnvironment) -> None:
    """get_wrapper_metadata should return built-in fields by default."""
    async with AgentContext(env=env) as ctx:
        wrapper_metadata = ctx.get_wrapper_metadata()

        assert wrapper_metadata == {
            "run_id": ctx.run_id,
            "agent_id": "main",
            "parent_run_id": None,
        }


async def test_get_wrapper_metadata_with_custom_fields(env: LocalEnvironment) -> None:
    """get_wrapper_metadata should merge wrapper_metadata field with built-in fields."""
    async with AgentContext(
        env=env,
        wrapper_metadata={"trace_id": "abc123", "tags": ["production"]},
    ) as ctx:
        wrapper_metadata = ctx.get_wrapper_metadata()

        assert wrapper_metadata == {
            "run_id": ctx.run_id,
            "agent_id": "main",
            "parent_run_id": None,
            "trace_id": "abc123",
            "tags": ["production"],
        }


async def test_get_wrapper_metadata_user_fields_override(env: LocalEnvironment) -> None:
    """User-defined fields in wrapper_metadata should override built-in fields."""
    custom_run_id = "custom-run-id-123"
    async with AgentContext(
        env=env,
        wrapper_metadata={"run_id": custom_run_id},
    ) as ctx:
        wrapper_metadata = ctx.get_wrapper_metadata()

        # User-defined run_id should take precedence
        assert wrapper_metadata["run_id"] == custom_run_id


async def test_wrapper_metadata_runtime_modification(env: LocalEnvironment) -> None:
    """wrapper_metadata should be modifiable at runtime."""
    async with AgentContext(env=env) as ctx:
        # Initially empty
        assert ctx.wrapper_metadata == {}

        # Modify at runtime
        ctx.wrapper_metadata["request_id"] = "req-123"
        ctx.wrapper_metadata["session_id"] = "sess-456"

        wrapper_metadata = ctx.get_wrapper_metadata()
        assert wrapper_metadata == {
            "run_id": ctx.run_id,
            "agent_id": "main",
            "parent_run_id": None,
            "request_id": "req-123",
            "session_id": "sess-456",
        }


async def test_get_wrapper_metadata_override_in_subclass(env: LocalEnvironment) -> None:
    """Subclasses can override get_wrapper_metadata for dynamic context."""

    class MyContext(AgentContext):
        custom_field: str = Field(default="custom_value")

        def get_wrapper_metadata(self) -> dict[str, Any]:
            return {
                **super().get_wrapper_metadata(),
                "custom_field": self.custom_field,
                "dynamic_value": "computed",
            }

    async with MyContext(env=env, custom_field="my_value") as ctx:
        wrapper_metadata = ctx.get_wrapper_metadata()

        assert wrapper_metadata == {
            "run_id": ctx.run_id,
            "agent_id": "main",
            "parent_run_id": None,
            "custom_field": "my_value",
            "dynamic_value": "computed",
        }


async def test_subagent_wrapper_metadata_has_different_run_id(env: LocalEnvironment) -> None:
    """Subagent's get_wrapper_metadata should return its own run_id and agent_id."""
    async with AgentContext(
        env=env,
        wrapper_metadata={"trace_id": "shared-trace"},
    ) as ctx:
        async with ctx.create_subagent_context("test-subagent") as sub_ctx:
            parent_context = ctx.get_wrapper_metadata()
            sub_context = sub_ctx.get_wrapper_metadata()

            # Both should have shared wrapper_metadata fields
            assert parent_context["trace_id"] == "shared-trace"
            assert sub_context["trace_id"] == "shared-trace"

            # Parent has main agent_id and no parent_run_id
            assert parent_context["agent_id"] == "main"
            assert parent_context["parent_run_id"] is None

            # Subagent has its own agent_id and parent_run_id set
            assert sub_context["agent_id"] == sub_ctx.agent_id
            assert sub_context["agent_id"] != "main"
            assert sub_context["parent_run_id"] == ctx.run_id

            # run_id should be different
            assert parent_context["run_id"] == ctx.run_id
            assert sub_context["run_id"] == sub_ctx.run_id
            assert parent_context["run_id"] != sub_context["run_id"]
