# Resumable Resources

Export and restore resource states across process restarts via async factories.

> **Note**: Base protocols and classes (`Resource`, `ResumableResource`, `BaseResource`, `ResourceRegistry`, etc.) are defined in the [agent-environment](https://github.com/youware-labs/agent-environment) protocol package.

## Overview

- **Resource**: Protocol requiring `close()` method
- **ResumableResource**: Protocol adding `export_state()`/`restore_state()`
- **InstructableResource**: Protocol adding `get_context_instructions()`
- **ToolableResource**: Protocol adding `get_toolsets()` for resource-provided tools
- **BaseResource**: Abstract base class implementing all protocols with defaults
- **ResourceFactory**: Async callable that creates resources
- **ResourceRegistryState**: Serializable state container

```mermaid
flowchart LR
    subgraph first["First Run"]
        direction TB
        A1[Environment.__aenter__] --> A2[_setup]
        A2 --> A3[register_factory]
        A3 --> A4[get_or_create]
        A4 --> A5[Use resource]
        A5 --> A6[export_resource_state]
        A6 --> A7[Save JSON]
    end

    subgraph second["Subsequent Run"]
        direction TB
        B1[Load JSON] --> B2[Environment with state+factories]
        B2 --> B3[__aenter__]
        B3 --> B4[_setup]
        B4 --> B5[restore_all]
        B5 --> B6[Resource ready]
    end

    first --> second
```

## Using BaseResource (Recommended)

`BaseResource` is a convenience abstract class with async `close()` and default no-op export/restore:

```python
from agent_environment import BaseResource

class BrowserSession(BaseResource):
    def __init__(self, browser: Browser):
        self._browser = browser

    async def close(self) -> None:
        await self._browser.close()

    async def export_state(self) -> dict[str, Any]:
        return {"cookies": await self._browser.get_cookies()}

    async def restore_state(self, state: dict[str, Any]) -> None:
        await self._browser.set_cookies(state.get("cookies", []))

    async def get_context_instructions(self) -> str | None:
        return "Browser session is active. Use browser tools for web navigation."

    def get_toolsets(self) -> list[Any]:
        """Provide browser-related tools."""
        return [BrowserToolset(self._browser)]
```

For resources that don't need state persistence, just implement `close()`:

```python
class DatabasePool(BaseResource):
    async def close(self) -> None:
        await self._pool.close()
    # export_state/restore_state use defaults (empty dict / no-op)
```

## Resource Lifecycle: setup()

`BaseResource.setup()` is called by `ResourceRegistry` after factory creation, before `restore_state()`. Use it for async initialization:

```python
class ProcessManager(BaseResource):
    def __init__(self):
        self._processes: dict[str, Process] = {}

    async def setup(self) -> None:
        """Called after factory creation, before restore_state()."""
        # Perform async initialization here
        await self._initialize_process_pool()

    async def close(self) -> None:
        await self._cleanup_all_processes()
```

Lifecycle order:

1. Factory creates resource instance (`__init__`)
2. `setup()` called for async initialization
3. `restore_state()` called if restoring from saved state
4. Resource is ready for use
5. `close()` called on cleanup

## Resource-Provided Toolsets

Resources can provide toolsets via `get_toolsets()`. This enables encapsulation where a resource owns both its state and the tools that operate on it:

```python
class ProcessManager(BaseResource):
    def get_toolsets(self) -> list[Any]:
        """Return process management tools."""
        from my_app.toolsets import ProcessToolset
        return [ProcessToolset(self)]

    async def get_context_instructions(self) -> str | None:
        """Report running processes to agent."""
        if not self._processes:
            return None
        return f"<processes count='{len(self._processes)}'>...</processes>"
```

Collect all resource toolsets via `ResourceRegistry`:

```python
async with LocalEnvironment() as env:
    env.resources.register_factory("process_manager", create_process_manager)
    await env.resources.get_or_create("process_manager")

    # Aggregate toolsets from all resources
    resource_toolsets = env.resources.get_toolsets()

    agent = Agent(
        ...,
        toolsets=[*core_toolsets, *env.get_toolsets()],
    )
```

## Implementing ResumableResource Protocol

For classes that can't inherit from `BaseResource`:

```python
class BrowserSession:
    async def export_state(self) -> dict[str, Any]:
        return {"cookies": await self._browser.get_cookies()}

    async def restore_state(self, state: dict[str, Any]) -> None:
        await self._browser.set_cookies(state.get("cookies", []))

    def close(self) -> None:
        self._browser.close()
```

## Basic Usage

### First Run: Create and Export

Factory functions receive the `Environment` instance:

```python
from agent_environment import Environment

async def create_browser(env: Environment) -> BrowserSession:
    return BrowserSession(
        file_operator=env.file_operator,
        tmp_dir=env.tmp_dir,
    )

async with LocalEnvironment() as env:
    env.resources.register_factory("browser", create_browser)
    browser = await env.resources.get_or_create("browser")

    # Use browser...

    state = await env.export_resource_state()
    Path("state.json").write_text(state.model_dump_json())
```

### Subsequent Run: Restore

```python
state = ResourceRegistryState.model_validate_json(Path("state.json").read_text())

async with LocalEnvironment(
    resource_state=state,
    resource_factories={"browser": create_browser},
) as env:
    browser = env.resources.get("browser")  # Already restored
```

### Chaining API

```python
env = (LocalEnvironment()
    .with_resource_factory("browser", create_browser)
    .with_resource_state(state))
```

## ResourceRegistry API

```python
# Factory registration
registry.register_factory("key", async_factory)

# Lazy creation
resource = await registry.get_or_create("key")
typed = await registry.get_or_create_typed("key", MyResource)

# State management
state = await registry.export_state()
count = await registry.restore_all()
restored = await registry.restore_one("key")

# Toolset collection (aggregates from all resources)
toolsets = registry.get_toolsets()

# Existing API (preserved)
registry.set("key", resource)
registry.get("key")
registry.get_typed("key", MyResource)
```

## Key Behaviors

- **Non-resumable resources**: Silently skipped during export/restore
- **Idempotent restore**: `restore_all()` clears pending state after first call
- **Lazy restoration**: Use `restore_one()` to restore on demand
- **Automatic restore**: `Environment.__aenter__` calls `restore_all()` after `_setup()`
- **Context instructions**: Resources with `get_context_instructions()` contribute to `Environment.get_context_instructions()`
- **Toolset aggregation**: Resources with `get_toolsets()` contribute tools via `ResourceRegistry.get_toolsets()`

## See Also

- [environment.md](environment.md) - Environment management
- [context.md](context.md) - AgentContext and session state
