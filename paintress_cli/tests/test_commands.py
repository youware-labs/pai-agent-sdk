"""Tests for paintress_cli.app.commands module."""

from __future__ import annotations

from typing import Any

import pytest
from paintress_cli.app import (
    BUILTIN_COMMANDS,
    Command,
    CommandRegistry,
    create_default_registry,
)

# =============================================================================
# Mock Context
# =============================================================================


class MockCommandContext:
    """Mock command context for testing."""

    def __init__(self) -> None:
        self.outputs: list[str] = []
        self.system_outputs: list[str] = []
        self.config: dict[str, Any] = {}

    def output(self, text: str) -> None:
        self.outputs.append(text)

    def output_system(self, text: str) -> None:
        self.system_outputs.append(text)

    def get_config(self) -> dict[str, Any]:
        return self.config


# =============================================================================
# Command Tests
# =============================================================================


def test_command_creation():
    """Test Command dataclass creation."""

    def handler(ctx, args):
        pass

    cmd = Command(
        name="test",
        handler=handler,
        description="Test command",
        aliases=["t", "tst"],
    )

    assert cmd.name == "test"
    assert cmd.description == "Test command"
    assert cmd.aliases == ["t", "tst"]


def test_command_default_values():
    """Test Command default values."""

    def handler(ctx, args):
        pass

    cmd = Command(name="test", handler=handler)

    assert cmd.description == ""
    assert cmd.aliases == []


# =============================================================================
# CommandRegistry Tests
# =============================================================================


def test_registry_init():
    """Test CommandRegistry initialization."""
    registry = CommandRegistry()
    assert registry.list_commands() == []


def test_registry_register():
    """Test registering a command."""
    registry = CommandRegistry()

    def handler(ctx, args):
        pass

    registry.register("test", handler, "Test command")

    assert registry.has("test")
    cmd = registry.get("test")
    assert cmd is not None
    assert cmd.name == "test"
    assert cmd.description == "Test command"


def test_registry_register_with_aliases():
    """Test registering a command with aliases."""
    registry = CommandRegistry()

    def handler(ctx, args):
        pass

    registry.register("help", handler, "Show help", aliases=["h", "?"])

    assert registry.has("help")
    assert registry.has("h")
    assert registry.has("?")

    # All should return the same command
    assert registry.get("help") == registry.get("h")
    assert registry.get("help") == registry.get("?")


def test_registry_get_not_found():
    """Test getting non-existent command."""
    registry = CommandRegistry()

    assert registry.get("nonexistent") is None
    assert not registry.has("nonexistent")


def test_registry_list_commands():
    """Test listing all commands."""
    registry = CommandRegistry()

    def handler1(ctx, args):
        pass

    def handler2(ctx, args):
        pass

    registry.register("cmd1", handler1)
    registry.register("cmd2", handler2)

    commands = registry.list_commands()
    assert len(commands) == 2
    names = {c.name for c in commands}
    assert names == {"cmd1", "cmd2"}


@pytest.mark.asyncio
async def test_registry_execute_sync():
    """Test executing a sync command."""
    registry = CommandRegistry()
    ctx = MockCommandContext()
    executed = []

    def handler(context, args):
        executed.append(args)
        context.output(f"Got: {args}")

    registry.register("echo", handler)

    result = await registry.execute("/echo hello world", ctx)

    assert result is True
    assert executed == ["hello world"]
    assert ctx.outputs == ["Got: hello world"]


@pytest.mark.asyncio
async def test_registry_execute_async():
    """Test executing an async command."""
    registry = CommandRegistry()
    ctx = MockCommandContext()
    executed = []

    async def handler(context, args):
        executed.append(args)
        context.output(f"Async: {args}")

    registry.register("async_cmd", handler)

    result = await registry.execute("/async_cmd test", ctx)

    assert result is True
    assert executed == ["test"]
    assert ctx.outputs == ["Async: test"]


@pytest.mark.asyncio
async def test_registry_execute_not_found():
    """Test executing non-existent command."""
    registry = CommandRegistry()
    ctx = MockCommandContext()

    result = await registry.execute("/nonexistent", ctx)

    assert result is False


@pytest.mark.asyncio
async def test_registry_execute_no_args():
    """Test executing command without args."""
    registry = CommandRegistry()
    ctx = MockCommandContext()
    received_args = []

    def handler(context, args):
        received_args.append(args)

    registry.register("noargs", handler)

    result = await registry.execute("/noargs", ctx)

    assert result is True
    assert received_args == [""]


@pytest.mark.asyncio
async def test_registry_execute_case_insensitive():
    """Test command name is case insensitive."""
    registry = CommandRegistry()
    ctx = MockCommandContext()
    executed = []

    def handler(context, args):
        executed.append(True)

    registry.register("test", handler)

    await registry.execute("/TEST", ctx)
    await registry.execute("/Test", ctx)
    await registry.execute("/test", ctx)

    assert len(executed) == 3


@pytest.mark.asyncio
async def test_registry_execute_via_alias():
    """Test executing via alias."""
    registry = CommandRegistry()
    ctx = MockCommandContext()
    executed = []

    def handler(context, args):
        executed.append(args)

    registry.register("help", handler, aliases=["h"])

    await registry.execute("/h info", ctx)

    assert executed == ["info"]


# =============================================================================
# BUILTIN_COMMANDS Tests
# =============================================================================


def test_builtin_commands():
    """Test BUILTIN_COMMANDS contains expected commands."""
    expected = {"help", "clear", "cost", "dump", "load", "exit", "act", "plan"}
    assert expected.issubset(BUILTIN_COMMANDS)


def test_builtin_commands_is_frozen():
    """Test BUILTIN_COMMANDS is immutable."""
    assert isinstance(BUILTIN_COMMANDS, frozenset)


# =============================================================================
# create_default_registry Tests
# =============================================================================


def test_create_default_registry():
    """Test create_default_registry returns empty registry."""
    registry = create_default_registry()
    assert isinstance(registry, CommandRegistry)
    # Empty by default (handlers registered by TUIApp)
    assert len(registry.list_commands()) == 0
