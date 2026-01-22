# AgentContext and Session Management

Session state management, resumable sessions, and extending for custom use cases.

## Overview

- **Session State**: Run ID, timing, user prompts, handoff messages
- **Model Configuration**: Context window, capabilities, and model settings
- **Tool Configuration**: API keys and tool-specific settings
- **Resumable Sessions**: Export/restore state for session persistence

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
    end

    subgraph Resumable["ResumableState"]
        SerializedHistory[Serialized History]
        Usages[Extra Usages]
    end

    Environment --> Context
    Context -->|export_state| Resumable
    Resumable -->|restore| Context
```

## Basic Usage

### Recommended: create_agent + stream_agent

```python
from pai_agent_sdk.agents import create_agent, stream_agent

runtime = create_agent("openai:gpt-4")
async with stream_agent(runtime, "Hello") as streamer:
    async for event in streamer:
        print(event)
```

### System Prompt Templates

`create_agent` supports Jinja2 templating for system prompts:

```python
# Template string with variables
runtime = create_agent(
    "openai:gpt-4",
    system_prompt="You are a {{ role }}. {{ instructions | default('') }}",
    system_prompt_template_vars={"role": "code reviewer"},
)

# Default template file (prompts/main.md) with variables
runtime = create_agent(
    "openai:gpt-4",
    system_prompt_template_vars={"project_name": "my-project"},
)
```

Templates are always rendered with Jinja2, supporting conditionals and default values even when `template_vars` is empty.

### Manual Context Management

```python
from pai_agent_sdk.environment import LocalEnvironment
from pai_agent_sdk.context import AgentContext, ModelConfig, ToolConfig

async with LocalEnvironment() as env:
    async with AgentContext(
        env=env,
        model_cfg=ModelConfig(context_window=200000),
        tool_config=ToolConfig(tavily_api_key="..."),
    ) as ctx:
        await ctx.file_operator.read_file("test.txt")
```

## Resumable Sessions

Export and restore session state for multi-turn conversations across restarts.

```python
# Export
state = ctx.export_state()
with open("session.json", "w") as f:
    f.write(state.model_dump_json())

# Restore
from pai_agent_sdk.context import ResumableState
state = ResumableState.model_validate_json(Path("session.json").read_text())
runtime = create_agent("openai:gpt-4", state=state)
```

The `with_state` method accepts `None` for conditional restoration:

```python
async with AgentContext(...).with_state(maybe_state) as ctx:
    ...
```

## Configuration Classes

### ModelConfig

```python
ModelConfig(
    context_window=200000,
    has_image_capability=True,
    has_video_capability=False,
)
```

### ToolConfig

```python
ToolConfig(
    tavily_api_key="tvly-xxx",
    firecrawl_api_key="fc-xxx",
)
```

## Extending ModelConfig and ToolConfig

Both `ModelConfig` and `ToolConfig` support extension for custom settings.

### Option 1: Inheritance (Recommended)

For full type safety, inherit from the config class and override the field in `AgentContext`:

```python
from pydantic import Field
from pai_agent_sdk.context import AgentContext, ToolConfig, ModelConfig

class MyToolConfig(ToolConfig):
    """Custom tool configuration with additional API keys."""
    my_service_api_key: str | None = None
    my_custom_setting: int = 100

class MyModelConfig(ModelConfig):
    """Custom model configuration."""
    custom_threshold: float = 0.8

class MyContext(AgentContext):
    tool_config: MyToolConfig = Field(default_factory=MyToolConfig)
    model_cfg: MyModelConfig = Field(default_factory=MyModelConfig)

# Usage with create_agent
runtime = create_agent(
    "openai:gpt-4o",
    context_type=MyContext,
    tool_config=MyToolConfig(my_service_api_key="xxx"),
    model_cfg=MyModelConfig(custom_threshold=0.9),
)
```

### Option 2: Extra Attributes (Quick Prototyping)

Both classes have `extra="allow"`, enabling arbitrary attributes without subclassing:

```python
# Extra attributes are accepted but not type-checked
config = ToolConfig(
    tavily_api_key="tvly-xxx",
    my_custom_key="value",  # Extra attribute
)

# Access via attribute or model_extra
config.my_custom_key  # Works at runtime
config.model_extra["my_custom_key"]  # Also works
```

> **Note**: Option 1 is recommended for production code as it provides IDE autocomplete and type checking. Option 2 is useful for quick experiments or dynamic configuration.

## ResumableState Fields

| Field                     | Type                     | Description                                  |
| ------------------------- | ------------------------ | -------------------------------------------- |
| `subagent_history`        | `dict[str, list[dict]]`  | Serialized conversation history per subagent |
| `extra_usages`            | `list[ExtraUsageRecord]` | Token usage records from tools/filters       |
| `user_prompts`            | `list[str]`              | Collected user prompts                       |
| `handoff_message`         | `str \| None`            | Context handoff message                      |
| `need_user_approve_tools` | `list[str]`              | Tool names requiring user approval           |

### ExtraUsageRecord Fields

| Field      | Type       | Description                                                           |
| ---------- | ---------- | --------------------------------------------------------------------- |
| `uuid`     | `str`      | Unique identifier (tool_call_id or generated UUID)                    |
| `agent`    | `str`      | Agent name (e.g., 'compact', 'image_understanding', 'search')         |
| `model_id` | `str`      | Model identifier (e.g., 'openai:gpt-4o', 'anthropic:claude-sonnet-4') |
| `usage`    | `RunUsage` | Token usage from this call                                            |

## Extending AgentContext

Extend `AgentContext` and `ResumableState` for custom fields:

```python
class MyContext(AgentContext):
    custom_field: str = ""

    def export_state(self) -> "MyState":
        base = super().export_state()
        return MyState(**base.model_dump(), custom_field=self.custom_field)

class MyState(ResumableState):
    custom_field: str = ""

    def restore(self, ctx: "MyContext") -> None:
        super().restore(ctx)
        ctx.custom_field = self.custom_field
```

> Full examples: `pai_agent_sdk/context.py`

## ToolIdWrapper

Normalizes tool call IDs across different model providers (OpenAI `call_`, Anthropic `toolu_`, etc.) for consistent session resumption and HITL flows.

Used automatically by the SDK streaming infrastructure.

## See Also

- [environment.md](environment.md) - FileOperator, Shell, and ResourceRegistry
- [toolset.md](toolset.md) - Creating and using tools
- [agent-environment](https://github.com/youware-labs/agent-environment) - Base protocol definitions
