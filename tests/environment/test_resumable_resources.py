"""Tests for resumable resources functionality."""

from typing import Any

import pytest

from pai_agent_sdk.environment import (
    BaseResource,
    InstructableResource,
    LocalEnvironment,
    Resource,
    ResourceEntry,
    ResourceRegistry,
    ResourceRegistryState,
    ResumableResource,
)

# --- Test fixtures and helpers ---


class SimpleResource:
    """A simple resource that only has close()."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class ResumableMockResource:
    """A resumable resource for testing."""

    def __init__(self, initial_data: str = "") -> None:
        self.data = initial_data
        self.closed = False
        self._restored_state: dict[str, Any] | None = None

    async def export_state(self) -> dict[str, Any]:
        return {"data": self.data}

    async def restore_state(self, state: dict[str, Any]) -> None:
        self.data = state.get("data", "")
        self._restored_state = state

    def close(self) -> None:
        self.closed = True


class MockBaseResource(BaseResource):
    """A BaseResource subclass for testing."""

    def __init__(self, value: str = "") -> None:
        self.value = value
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    async def export_state(self) -> dict[str, Any]:
        return {"value": self.value}

    async def restore_state(self, state: dict[str, Any]) -> None:
        self.value = state.get("value", "")


class MinimalBaseResource(BaseResource):
    """A minimal BaseResource subclass with default export/restore."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class ResourceWithInstructions(BaseResource):
    """A BaseResource subclass with context instructions."""

    def __init__(self, instructions: str) -> None:
        self._instructions = instructions
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    async def get_context_instructions(self) -> str | None:
        return self._instructions


# --- ResumableResource Protocol Tests ---


def test_resumable_resource_protocol_detection() -> None:
    """Should detect ResumableResource implementation via isinstance."""
    resumable = ResumableMockResource()
    simple = SimpleResource()

    assert isinstance(resumable, Resource)
    assert isinstance(resumable, ResumableResource)
    assert isinstance(simple, Resource)
    assert not isinstance(simple, ResumableResource)


# --- ResourceEntry and ResourceRegistryState Tests ---


def test_resource_entry_model() -> None:
    """Should create ResourceEntry with state."""
    entry = ResourceEntry(state={"key": "value", "count": 42})
    assert entry.state == {"key": "value", "count": 42}


def test_resource_registry_state_model() -> None:
    """Should create ResourceRegistryState with entries."""
    state = ResourceRegistryState(
        entries={
            "browser": ResourceEntry(state={"cookies": []}),
            "cache": ResourceEntry(state={"entries": {}}),
        }
    )
    assert len(state.entries) == 2
    assert "browser" in state.entries
    assert state.entries["browser"].state == {"cookies": []}


def test_resource_registry_state_serialization() -> None:
    """Should serialize and deserialize ResourceRegistryState."""
    original = ResourceRegistryState(
        entries={
            "resource1": ResourceEntry(state={"data": "test"}),
        }
    )

    # Serialize to JSON
    json_str = original.model_dump_json()
    assert isinstance(json_str, str)

    # Deserialize back
    restored = ResourceRegistryState.model_validate_json(json_str)
    assert restored.entries["resource1"].state == {"data": "test"}


# --- ResourceRegistry Factory Tests ---


async def test_registry_register_factory() -> None:
    """Should register and use factory."""
    registry = ResourceRegistry()

    async def create_resource() -> SimpleResource:
        return SimpleResource()

    registry.register_factory("simple", create_resource)
    assert "simple" not in registry


async def test_registry_get_or_create_new() -> None:
    """Should create resource via factory when not exists."""
    registry = ResourceRegistry()
    created_resources: list[SimpleResource] = []

    async def create_resource() -> SimpleResource:
        r = SimpleResource()
        created_resources.append(r)
        return r

    registry.register_factory("simple", create_resource)
    resource = await registry.get_or_create("simple")

    assert len(created_resources) == 1
    assert resource is created_resources[0]
    assert "simple" in registry


async def test_registry_get_or_create_existing() -> None:
    """Should return existing resource without calling factory."""
    registry = ResourceRegistry()
    call_count = 0

    async def create_resource() -> SimpleResource:
        nonlocal call_count
        call_count += 1
        return SimpleResource()

    registry.register_factory("simple", create_resource)

    # First call creates
    r1 = await registry.get_or_create("simple")
    # Second call returns existing
    r2 = await registry.get_or_create("simple")

    assert call_count == 1
    assert r1 is r2


async def test_registry_get_or_create_no_factory() -> None:
    """Should raise KeyError when no factory registered."""
    registry = ResourceRegistry()

    with pytest.raises(KeyError, match="No resource or factory registered"):
        await registry.get_or_create("missing")


async def test_registry_get_or_create_typed() -> None:
    """Should return typed resource."""
    registry = ResourceRegistry()

    async def create_resource() -> SimpleResource:
        return SimpleResource()

    registry.register_factory("simple", create_resource)
    resource = await registry.get_or_create_typed("simple", SimpleResource)

    assert isinstance(resource, SimpleResource)


async def test_registry_get_or_create_typed_wrong_type() -> None:
    """Should raise TypeError for wrong resource type."""
    registry = ResourceRegistry()

    async def create_resource() -> SimpleResource:
        return SimpleResource()

    registry.register_factory("simple", create_resource)
    await registry.get_or_create("simple")

    with pytest.raises(TypeError, match="expected ResumableMockResource"):
        await registry.get_or_create_typed("simple", ResumableMockResource)


# --- ResourceRegistry Export/Restore Tests ---


async def test_registry_export_state_resumable() -> None:
    """Should export state for resumable resources only."""
    registry = ResourceRegistry()

    async def create_simple() -> SimpleResource:
        return SimpleResource()

    async def create_resumable() -> ResumableMockResource:
        r = ResumableMockResource()
        r.data = "test_data"
        return r

    registry.register_factory("simple", create_simple)
    registry.register_factory("resumable", create_resumable)

    await registry.get_or_create("simple")
    await registry.get_or_create("resumable")

    state = await registry.export_state()

    # Only resumable resource should be in state
    assert "simple" not in state.entries
    assert "resumable" in state.entries
    assert state.entries["resumable"].state == {"data": "test_data"}


async def test_registry_restore_all() -> None:
    """Should restore all resources from pending state."""
    # Create initial state
    pending_state = ResourceRegistryState(
        entries={
            "browser": ResourceEntry(state={"data": "restored_data"}),
        }
    )

    async def create_browser() -> ResumableMockResource:
        return ResumableMockResource(initial_data="fresh")

    registry = ResourceRegistry(
        state=pending_state,
        factories={"browser": create_browser},
    )

    # Restore
    count = await registry.restore_all()

    assert count == 1
    assert "browser" in registry

    browser = registry.get_typed("browser", ResumableMockResource)
    assert browser is not None
    assert browser.data == "restored_data"
    assert browser._restored_state == {"data": "restored_data"}


async def test_registry_restore_all_idempotent() -> None:
    """Should be idempotent - second call does nothing."""
    pending_state = ResourceRegistryState(
        entries={
            "resource": ResourceEntry(state={"data": "test"}),
        }
    )

    call_count = 0

    async def create_resource() -> ResumableMockResource:
        nonlocal call_count
        call_count += 1
        return ResumableMockResource()

    registry = ResourceRegistry(
        state=pending_state,
        factories={"resource": create_resource},
    )

    # First restore
    count1 = await registry.restore_all()
    # Second restore
    count2 = await registry.restore_all()

    assert count1 == 1
    assert count2 == 0
    assert call_count == 1


async def test_registry_restore_all_no_factory() -> None:
    """Should raise KeyError when factory missing for pending resource."""
    pending_state = ResourceRegistryState(
        entries={
            "browser": ResourceEntry(state={"data": "test"}),
        }
    )

    registry = ResourceRegistry(state=pending_state)  # No factories

    with pytest.raises(KeyError, match="No factory registered for pending resource"):
        await registry.restore_all()


async def test_registry_restore_one() -> None:
    """Should restore single resource lazily."""
    pending_state = ResourceRegistryState(
        entries={
            "a": ResourceEntry(state={"data": "data_a"}),
            "b": ResourceEntry(state={"data": "data_b"}),
        }
    )

    async def create_resource() -> ResumableMockResource:
        return ResumableMockResource()

    registry = ResourceRegistry(
        state=pending_state,
        factories={"a": create_resource, "b": create_resource},
    )

    # Restore only "a"
    result = await registry.restore_one("a")

    assert result is True
    assert "a" in registry
    assert "b" not in registry

    resource_a = registry.get_typed("a", ResumableMockResource)
    assert resource_a is not None
    assert resource_a.data == "data_a"


async def test_registry_restore_one_not_in_pending() -> None:
    """Should return False when key not in pending state."""
    registry = ResourceRegistry()
    result = await registry.restore_one("missing")
    assert result is False


async def test_registry_close_all_clears_factories() -> None:
    """Should clear factories when closing all resources."""
    registry = ResourceRegistry()

    async def create_resource() -> SimpleResource:
        return SimpleResource()

    registry.register_factory("simple", create_resource)
    await registry.get_or_create("simple")

    await registry.close_all()

    assert len(registry._factories) == 0
    assert len(registry) == 0


# --- Environment Integration Tests ---


async def test_environment_constructor_with_state() -> None:
    """Should accept resource_state and resource_factories in constructor."""
    state = ResourceRegistryState(entries={"cache": ResourceEntry(state={"data": "cached"})})

    async def create_cache() -> ResumableMockResource:
        return ResumableMockResource()

    async with LocalEnvironment(
        resource_state=state,
        resource_factories={"cache": create_cache},
    ) as env:
        # Resource should be restored on enter
        cache = env.resources.get_typed("cache", ResumableMockResource)
        assert cache is not None
        assert cache.data == "cached"


async def test_environment_chaining_api() -> None:
    """Should support chaining API for factories and state."""
    state = ResourceRegistryState(entries={"session": ResourceEntry(state={"data": "user_123"})})

    async def create_session() -> ResumableMockResource:
        return ResumableMockResource()

    env = LocalEnvironment().with_resource_factory("session", create_session).with_resource_state(state)

    async with env:
        session = env.resources.get_typed("session", ResumableMockResource)
        assert session is not None
        assert session.data == "user_123"


async def test_environment_export_resource_state() -> None:
    """Should export resource state via environment method."""

    async def create_session() -> ResumableMockResource:
        r = ResumableMockResource()
        r.data = "session_data"
        return r

    async with LocalEnvironment().with_resource_factory("session", create_session) as env:
        await env.resources.get_or_create("session")
        state = await env.export_resource_state()

        assert "session" in state.entries
        assert state.entries["session"].state == {"data": "session_data"}


async def test_environment_full_roundtrip() -> None:
    """Should support full export -> JSON -> restore cycle."""

    async def create_browser() -> ResumableMockResource:
        return ResumableMockResource()

    # First session: create and use resource
    async with LocalEnvironment().with_resource_factory("browser", create_browser) as env1:
        browser = await env1.resources.get_or_create_typed("browser", ResumableMockResource)
        browser.data = "session_cookies_data"

        # Export state
        state1 = await env1.export_resource_state()
        json_data = state1.model_dump_json()

    # Second session: restore from JSON
    state2 = ResourceRegistryState.model_validate_json(json_data)

    async with LocalEnvironment(
        resource_state=state2,
        resource_factories={"browser": create_browser},
    ) as env2:
        # Resource should be restored automatically
        browser2 = env2.resources.get_typed("browser", ResumableMockResource)
        assert browser2 is not None
        assert browser2.data == "session_cookies_data"


async def test_environment_backward_compatible() -> None:
    """Should preserve existing set/get API."""
    async with LocalEnvironment() as env:
        # Old API should still work
        resource = SimpleResource()
        env.resources.set("legacy", resource)

        retrieved = env.resources.get_typed("legacy", SimpleResource)
        assert retrieved is resource


# --- BaseResource Tests ---


def test_base_resource_implements_protocols() -> None:
    """BaseResource subclasses should implement both Resource and ResumableResource."""
    resource = MockBaseResource()
    assert isinstance(resource, Resource)
    assert isinstance(resource, ResumableResource)


async def test_base_resource_close() -> None:
    """BaseResource.close() should be async."""
    resource = MockBaseResource()
    assert not resource.closed
    await resource.close()
    assert resource.closed


async def test_base_resource_export_state() -> None:
    """BaseResource subclass should export state."""
    resource = MockBaseResource(value="test_value")
    state = await resource.export_state()
    assert state == {"value": "test_value"}


async def test_base_resource_restore_state() -> None:
    """BaseResource subclass should restore state."""
    resource = MockBaseResource()
    await resource.restore_state({"value": "restored_value"})
    assert resource.value == "restored_value"


async def test_base_resource_default_export() -> None:
    """MinimalBaseResource should use default empty export."""
    resource = MinimalBaseResource()
    state = await resource.export_state()
    assert state == {}


async def test_base_resource_default_restore() -> None:
    """MinimalBaseResource should use default no-op restore."""
    resource = MinimalBaseResource()
    await resource.restore_state({"arbitrary": "data"})  # Should not raise


async def test_base_resource_with_registry() -> None:
    """BaseResource subclass should work with ResourceRegistry."""
    registry = ResourceRegistry()

    async def create_mock() -> MockBaseResource:
        return MockBaseResource(value="initial")

    registry.register_factory("mock", create_mock)
    resource = await registry.get_or_create_typed("mock", MockBaseResource)
    resource.value = "modified"

    state = await registry.export_state()
    assert "mock" in state.entries
    assert state.entries["mock"].state == {"value": "modified"}
    resource.value = "modified"

    state = await registry.export_state()
    assert "mock" in state.entries
    assert state.entries["mock"].state == {"value": "modified"}


# --- get_context_instructions Tests ---


def test_instructable_resource_protocol_detection() -> None:
    """Should detect InstructableResource implementation via isinstance."""
    instructable = ResourceWithInstructions("test")
    simple = SimpleResource()
    minimal = MinimalBaseResource()

    assert isinstance(instructable, InstructableResource)
    assert not isinstance(simple, InstructableResource)
    # MinimalBaseResource has get_context_instructions from BaseResource
    assert isinstance(minimal, InstructableResource)


async def test_base_resource_default_context_instructions() -> None:
    """BaseResource default get_context_instructions returns None."""
    resource = MinimalBaseResource()
    result = await resource.get_context_instructions()
    assert result is None


async def test_base_resource_custom_context_instructions() -> None:
    """BaseResource subclass can provide custom instructions."""
    resource = ResourceWithInstructions("Use browser for web tasks.")
    result = await resource.get_context_instructions()
    assert result == "Use browser for web tasks."


async def test_registry_get_context_instructions_empty() -> None:
    """ResourceRegistry returns None when no resources have instructions."""
    registry = ResourceRegistry()
    registry.set("simple", SimpleResource())
    result = await registry.get_context_instructions()
    assert result is None


async def test_registry_get_context_instructions_with_resources() -> None:
    """ResourceRegistry collects instructions from all resources."""
    registry = ResourceRegistry()

    async def create_r1() -> ResourceWithInstructions:
        return ResourceWithInstructions("Instructions for R1")

    async def create_r2() -> ResourceWithInstructions:
        return ResourceWithInstructions("Instructions for R2")

    registry.register_factory("r1", create_r1)
    registry.register_factory("r2", create_r2)
    await registry.get_or_create("r1")
    await registry.get_or_create("r2")

    result = await registry.get_context_instructions()
    assert result is not None
    assert "Instructions for R1" in result
    assert "Instructions for R2" in result
    assert "r1" in result
    assert "r2" in result


async def test_environment_context_instructions_includes_resources() -> None:
    """Environment.get_context_instructions includes resource instructions."""

    async def create_browser() -> ResourceWithInstructions:
        return ResourceWithInstructions("Browser session is active.")

    async with LocalEnvironment().with_resource_factory("browser", create_browser) as env:
        await env.resources.get_or_create("browser")
        result = await env.get_context_instructions()
        assert "Browser session is active." in result
