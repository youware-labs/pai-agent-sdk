# Environment Management

This document describes the Environment architecture in pai-agent-sdk, including resource management, lifecycle hooks, and usage patterns.

## Overview

The Environment system provides:

- **FileOperator**: Abstraction for file system operations
- **Shell**: Abstraction for command execution
- **ResourceRegistry**: Type-safe runtime resource management
- **Lifecycle hooks**: `_setup()` / `_teardown()` pattern for subclasses

## Architecture

```
Environment (ABC) - Long-lived, owns resources
  ├── _resources: ResourceRegistry     # Runtime resources (browser, db, etc.)
  ├── _file_operator: FileOperator     # File system operations
  ├── _shell: Shell                    # Command execution
  ├── _setup() -> None                 # Subclass hook: initialization
  ├── _teardown() -> None              # Subclass hook: cleanup
  ├── __aenter__() -> Self             # Base class: calls _setup()
  └── __aexit__()                      # Base class: _teardown() + resources.close_all()

AgentContext - Short-lived, references resources
  ├── file_operator: FileOperator      # Reference from Environment
  ├── shell: Shell                     # Reference from Environment
  ├── resources: ResourceRegistry      # Reference from Environment
  └── (session state: run_id, model_cfg, etc.)
```

## Basic Usage

### Using AsyncExitStack (Recommended)

```python
from contextlib import AsyncExitStack
from pai_agent_sdk.environment import LocalEnvironment
from pai_agent_sdk.context import AgentContext

async with AsyncExitStack() as stack:
    env = await stack.enter_async_context(
        LocalEnvironment(allowed_paths=[path], tmp_base_dir=path)
    )
    ctx = await stack.enter_async_context(
        AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            resources=env.resources,
        )
    )
    # Use env and ctx here
```

### Multiple Sessions Sharing Environment

```python
async with LocalEnvironment(tmp_base_dir=Path("/tmp")) as env:
    # First session
    async with AgentContext(
        file_operator=env.file_operator,
        shell=env.shell,
        resources=env.resources,
    ) as ctx1:
        await ctx1.file_operator.read_file("test.txt")

    # Second session (reuses same environment)
    async with AgentContext(
        file_operator=env.file_operator,
        shell=env.shell,
        resources=env.resources,
    ) as ctx2:
        ...
# Resources cleaned up when environment exits
```

## Resource Management

### ResourceRegistry

The `ResourceRegistry` provides type-safe resource management with protocol validation:

```python
from pai_agent_sdk.environment import Resource, ResourceRegistry

# Resource Protocol - must implement close()
class Browser:
    async def close(self) -> None:
        await self._driver.quit()

# Register resources
registry = ResourceRegistry()
browser = await Browser.create()
registry.set("browser", browser)  # Validates Resource protocol

# Access resources
browser = registry.get("browser")                    # Returns Resource | None
browser = registry.get_typed("browser", Browser)     # Type-safe access

# Cleanup (called automatically by Environment)
await registry.close_all()
```

### Using Resources in Tools

```python
async def screenshot_tool(ctx: RunContext[AgentContext], url: str) -> str:
    browser = ctx.deps.resources.get_typed("browser", Browser)
    if browser is None:
        return "Browser not available"
    return await browser.screenshot(url)
```

## Creating Custom Environments

### The \_setup/\_teardown Pattern

Subclasses implement hooks instead of `__aenter__`/`__aexit__`:

```python
from pai_agent_sdk.environment import Environment, FileOperator, Shell

class MyEnvironment(Environment):
    def __init__(self, config: MyConfig):
        super().__init__()  # Initialize ResourceRegistry
        self._config = config

    async def _setup(self) -> None:
        """Initialize resources. Called by __aenter__."""
        # Required: set up file_operator and shell
        self._file_operator = LocalFileOperator(...)
        self._shell = LocalShell(...)

        # Optional: register custom resources
        db = await Database.connect(self._config.db_url)
        self._resources.set("db", db)

    async def _teardown(self) -> None:
        """Clean up resources. Called by __aexit__.

        Note: resources.close_all() is called automatically after this.
        """
        # Clean up environment-specific resources
        if self._tmp_dir:
            self._tmp_dir.cleanup()

        self._file_operator = None
        self._shell = None
```

### Why Not Override __aenter__/__aexit__?

1. **Safe inheritance**: No need to remember `await super().__aenter__()`
2. **Consistent cleanup**: Base class ensures `resources.close_all()` is always called
3. **Clear contract**: `_setup` for init, `_teardown` for env-specific cleanup

## Available Implementations

### LocalEnvironment

Local filesystem and shell execution:

```python
from pai_agent_sdk.environment import LocalEnvironment

async with LocalEnvironment(
    allowed_paths=[Path("/workspace")],
    default_path=Path("/workspace"),
    shell_timeout=30.0,
    tmp_base_dir=Path("/tmp"),
    enable_tmp_dir=True,
) as env:
    ...
```

### DockerEnvironment

Docker container execution with local file mounting:

```python
from pai_agent_sdk.environment import DockerEnvironment

async with DockerEnvironment(
    mount_dir=Path("/home/user/project"),
    container_workdir="/workspace",
    image="python:3.11",              # Create new container
    # container_id="abc123",          # Or use existing container
    cleanup_on_exit=True,
) as env:
    # Files at mount_dir appear at container_workdir
    await env.file_operator.write_file("test.py", "print('hello')")
    # Shell runs inside container
    code, stdout, stderr = await env.shell.execute("python test.py")
```

## Best Practices

1. **Use AsyncExitStack** for dependent context managers
2. **Call `super().__init__()`** in custom Environment subclasses
3. **Register resources** that need cleanup via `_resources.set()`
4. **Use `get_typed()`** for type-safe resource access in tools
5. **Don't override `__aenter__`/`__aexit__`** - use `_setup`/`_teardown` instead
