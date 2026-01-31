"""Command handling for TUI slash commands.

Provides a registry-based command system for slash commands.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    pass


class CommandContext(Protocol):
    """Protocol for command execution context."""

    def output(self, text: str) -> None:
        """Output text to display."""
        ...

    def output_system(self, text: str) -> None:
        """Output system message."""
        ...

    def get_config(self) -> Any:
        """Get configuration."""
        ...


@dataclass
class Command:
    """A registered slash command."""

    name: str
    handler: Callable[[CommandContext, str], Awaitable[None] | None]
    description: str = ""
    aliases: list[str] = field(default_factory=list)


class CommandRegistry:
    """Registry for slash commands.

    Supports:
    - Built-in commands (cannot be overridden)
    - Custom commands from config
    - Command aliases
    """

    def __init__(self) -> None:
        """Initialize command registry."""
        self._commands: dict[str, Command] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        name: str,
        handler: Callable[[CommandContext, str], Awaitable[None] | None],
        description: str = "",
        aliases: list[str] | None = None,
    ) -> None:
        """Register a command.

        Args:
            name: Command name (without leading /).
            handler: Async or sync function taking (ctx, args).
            description: Help text for the command.
            aliases: Alternative names for the command.
        """
        cmd = Command(
            name=name,
            handler=handler,
            description=description,
            aliases=aliases or [],
        )
        self._commands[name] = cmd

        # Register aliases
        for alias in cmd.aliases:
            self._aliases[alias] = name

    def get(self, name: str) -> Command | None:
        """Get a command by name or alias.

        Args:
            name: Command name (without leading /).

        Returns:
            Command if found, None otherwise.
        """
        # Check direct name
        if name in self._commands:
            return self._commands[name]

        # Check aliases
        if name in self._aliases:
            return self._commands.get(self._aliases[name])

        return None

    def has(self, name: str) -> bool:
        """Check if command exists."""
        return self.get(name) is not None

    def list_commands(self) -> list[Command]:
        """Get all registered commands."""
        return list(self._commands.values())

    async def execute(self, command_str: str, ctx: CommandContext) -> bool:
        """Execute a command.

        Args:
            command_str: Full command string (e.g., "/help" or "/dump folder").
            ctx: Command context.

        Returns:
            True if command was found and executed, False otherwise.
        """
        # Parse command
        parts = command_str.lstrip("/").split(maxsplit=1)
        name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Find command
        cmd = self.get(name)
        if not cmd:
            return False

        # Execute
        result = cmd.handler(ctx, args)
        if result is not None:
            await result

        return True


# Built-in command names (reserved)
BUILTIN_COMMANDS = frozenset({
    "help",
    "clear",
    "cost",
    "dump",
    "load",
    "exit",
    "act",
    "plan",
})


def create_default_registry() -> CommandRegistry:
    """Create a registry with placeholder for built-in commands.

    Note: Actual handlers are registered by TUIApp since they need
    access to app state.
    """
    return CommandRegistry()
