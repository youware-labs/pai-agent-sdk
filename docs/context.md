# AgentContext and Session Management

This document describes the AgentContext architecture in pai-agent-sdk, including session state management, resumable sessions, and extending for custom use cases.

## Overview

The AgentContext system provides:

- **Session State**: Run ID, timing, user prompts, handoff messages
- **Model Configuration**: Context window, capabilities, and model settings
- **Tool Configuration**: API keys and tool-specific settings
- **Resumable Sessions**: Export/restore state for session persistence
- **Subagent History**: Conversation history management for subagents

## Architecture

```mermaid
flowchart TB
    subgraph Environment["Environment (long-lived)"]
        FileOp[FileOperator]
        Shell[Shell]
        Resources[ResourceRegistry]
    end

    subgraph Context["AgentContext (short-lived)"]
        State[Session State]
        ModelCfg[ModelConfig]
        ToolCfg[ToolConfig]
        History[Subagent History]
    end

    subgraph Resumable["ResumableState"]
        SerializedHistory[Serialized History]
        Usages[Extra Usages]
        Prompts[User Prompts]
    end

    Environment --> Context
    Context -->|export_state| Resumable
    Resumable -->|restore| Context
```

## Basic Usage

### Using create_agent (Recommended)

```python
from pai_agent_sdk.agents import create_agent

async with create_agent("openai:gpt-4") as runtime:
    result = await runtime.agent.run("Hello", deps=runtime.ctx)
    print(result.output)
```

### Manual Context Management

```python
from contextlib import AsyncExitStack
from pai_agent_sdk.environment import LocalEnvironment
from pai_agent_sdk.context import AgentContext, ModelConfig, ToolConfig

async with AsyncExitStack() as stack:
    env = await stack.enter_async_context(LocalEnvironment())
    ctx = await stack.enter_async_context(
        AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
            resources=env.resources,
            model_cfg=ModelConfig(context_window=200000),
            tool_config=ToolConfig(tavily_api_key="..."),
        )
    )
    # Use ctx here
```

## Resumable Sessions

AgentContext supports exporting and restoring session state, enabling:

- Multi-turn conversations across restarts
- Session persistence to database or file
- State transfer between different contexts

### Exporting State

```python
# Save state to JSON file
state = ctx.export_state()
with open("session.json", "w") as f:
    f.write(state.model_dump_json(indent=2))
```

### Restoring State

```python
from pai_agent_sdk.context import ResumableState

# Load and restore with create_agent
with open("session.json") as f:
    state = ResumableState.model_validate_json(f.read())

async with create_agent("openai:gpt-4", state=state) as runtime:
    # Session is restored
    result = await runtime.agent.run("Continue our conversation", deps=runtime.ctx)
```

### Chaining with with_state

The `with_state` method supports `None` for convenient conditional restoration:

```python
# Works with both state and None
maybe_state = load_state_if_exists()  # Returns ResumableState | None
async with AgentContext(...).with_state(maybe_state) as ctx:
    ...
```

## Extending AgentContext

### Custom Context with Additional Fields

```python
from pai_agent_sdk.context import AgentContext, ResumableState

class MyContext(AgentContext):
    """Custom context with additional session state."""

    custom_field: str = ""
    session_metadata: dict[str, Any] = {}

    def export_state(self) -> "MyState":
        """Export including custom fields."""
        base = super().export_state()
        return MyState(
            **base.model_dump(),
            custom_field=self.custom_field,
            session_metadata=self.session_metadata,
        )
```

### Custom ResumableState

Extend `ResumableState` and override `restore()` to handle custom fields:

```python
class MyState(ResumableState):
    """Custom state with additional fields."""

    custom_field: str = ""
    session_metadata: dict[str, Any] = {}

    def restore(self, ctx: "MyContext") -> None:
        """Restore base state plus custom fields."""
        super().restore(ctx)
        ctx.custom_field = self.custom_field
        ctx.session_metadata = dict(self.session_metadata)
```

### Complete Example

```python
from typing import Any
from pai_agent_sdk.context import AgentContext, ResumableState
from pai_agent_sdk.agents import create_agent

class MyState(ResumableState):
    user_preferences: dict[str, Any] = {}
    conversation_topic: str = ""

    def restore(self, ctx: "MyContext") -> None:
        super().restore(ctx)
        ctx.user_preferences = dict(self.user_preferences)
        ctx.conversation_topic = self.conversation_topic


class MyContext(AgentContext):
    user_preferences: dict[str, Any] = {}
    conversation_topic: str = ""

    def export_state(self) -> MyState:
        base = super().export_state()
        return MyState(
            **base.model_dump(),
            user_preferences=self.user_preferences,
            conversation_topic=self.conversation_topic,
        )


# Usage
async with create_agent(
    "openai:gpt-4",
    context_type=MyContext,
    state=loaded_state,  # MyState instance
) as runtime:
    runtime.ctx.conversation_topic = "Python programming"
    result = await runtime.agent.run("Hello", deps=runtime.ctx)

    # Save for later
    state = runtime.ctx.export_state()
```

## Configuration Classes

### ModelConfig

Configure model-related settings:

```python
from pai_agent_sdk.context import ModelConfig

model_cfg = ModelConfig(
    context_window=200000,      # Max tokens for context
    has_image_capability=True,  # Enable image processing
    has_video_capability=False, # Disable video processing
)
```

### ToolConfig

Configure tool-specific settings and API keys:

```python
from pai_agent_sdk.context import ToolConfig

tool_config = ToolConfig(
    tavily_api_key="tvly-xxx",      # For web search
    firecrawl_api_key="fc-xxx",     # For web scraping
    # Add other tool-specific settings
)
```

## ResumableState Fields

The base `ResumableState` includes:

| Field                     | Type                     | Description                                    |
| ------------------------- | ------------------------ | ---------------------------------------------- |
| `subagent_history`        | `dict[str, list[dict]]`  | Serialized conversation history per subagent   |
| `extra_usages`            | `list[ExtraUsageRecord]` | Token usage records from tools/filters         |
| `user_prompts`            | `list[str]`              | Collected user prompts                         |
| `handoff_message`         | `str \| None`            | Context handoff message                        |
| `deferred_tool_metadata`  | `dict[str, dict]`        | Metadata for deferred tool calls               |
| `need_user_approve_tools` | `list[str]`              | Tool names requiring user approval (HITL flow) |

## Best Practices

1. **Use create_agent for new projects**: It handles lifecycle management automatically
2. **Always use AsyncExitStack**: When manually managing multiple contexts
3. **Extend ResumableState.restore()**: When adding custom fields to ensure proper restoration
4. **Export state before shutdown**: To enable session resumption
5. **Validate state before restore**: Use Pydantic's validation when loading from external sources

## Related Documentation

- [Environment Management](environment.md) - FileOperator, Shell, and ResourceRegistry
- [Toolset Architecture](toolset.md) - Creating and using tools
- [Logging Configuration](logging.md) - Configure SDK logging
