## Project Overview

**pai-agent-sdk** is an application framework for building AI agents with [Pydantic AI](https://ai.pydantic.dev/). It provides environment abstractions, session management, and hierarchical agent patterns.

- **Language**: Python 3.11+
- **Package Manager**: uv
- **Build System**: hatchling

## Key Features

- **Environment-based Architecture**: Inject file operations, shell access, and resources via `Environment` for clean separation of concerns (LocalEnvironment, DockerEnvironment)
- **Resumable Sessions**: Export/restore `AgentContext` state for multi-turn conversations across restarts
- **Hierarchical Agents**: Subagent system with task delegation, tool inheritance, and markdown-based configuration
- **Skills System**: Markdown-based instruction files with hot reload and progressive loading
- **Human-in-the-Loop**: Built-in approval workflows for sensitive tool operations
- **Toolset Architecture**: Extensible tool system with pre/post hooks for logging, validation, and error handling
- **Resumable Resources**: Export and restore resource states (like browser sessions) across process restarts
- **Browser Automation**: Docker-based headless Chrome sandbox for safe browser automation
- **Streaming Support**: Real-time streaming of agent responses and tool executions

## Project Structure

```
pai_agent_sdk/
├── agents/                # Agent implementations
│   ├── main.py            # create_agent, stream_agent entry points
│   ├── compact.py         # Compact agent variant
│   ├── image_understanding.py  # Image understanding agent
│   ├── video_understanding.py  # Video understanding agent
│   └── models/            # Model configuration and inference
│
├── context.py             # AgentContext, ModelConfig, ToolConfig, ResumableState
│
├── environment/           # Environment management
│   ├── base.py            # Environment ABC, FileOperator, Shell, ResourceRegistry, BaseResource
│   ├── local.py           # LocalEnvironment for local filesystem
│   └── docker.py          # DockerEnvironment for container-based execution
│
├── toolsets/              # Tool implementations
│   ├── core/              # Core toolsets collection
│   │   ├── base.py        # BaseTool, Toolset, GlobalHooks (base classes)
│   │   ├── content/       # Content loading tools
│   │   ├── context/       # Context management tools (handoff)
│   │   ├── document/      # Document processing tools
│   │   ├── enhance/       # Enhancement tools (todo, thinking)
│   │   ├── filesystem/    # File system operation tools
│   │   ├── multimodal/    # Multimodal tools (read_image, read_video)
│   │   ├── shell/         # Shell command execution tools
│   │   ├── subagent/      # Subagent delegation tools
│   │   └── web/           # Web interaction tools
│   └── browser_use/       # Browser automation toolset (independent)
│
├── subagents/             # Subagent system
│   ├── config.py          # SubagentConfig parsing
│   ├── factory.py         # Subagent tool factory functions
│   └── presets/           # Built-in subagent presets
│       ├── debugger.md    # Debugging specialist
│       ├── explorer.md    # Codebase exploration specialist
│       ├── searcher.md    # Search specialist
│       └── code-reviewer.md # Code review specialist
│
├── filters/               # Message history processors
│   ├── handoff.py         # Handoff message processing
│   ├── image.py           # Image filtering
│   ├── system_prompt.py   # System prompt filtering
│   └── tool_args.py       # Tool argument fixing
│
├── sandbox/               # Sandbox environments
│   └── browser/           # Browser sandbox
│
├── skills/                # Skill definitions
│   └── checkpointing/     # Checkpointing skill
│
├── stream/                # Stream processing
├── presets.py             # Preset configurations (model settings, etc.)
├── utils.py               # Utility functions
└── _logger.py             # Centralized logging

tests/                     # Test suite (pytest)
├── environment/           # Environment tests
├── filters/               # Filter tests
├── sandbox/               # Sandbox tests
├── subagents/             # Subagent tests
└── toolsets/              # Toolset tests
```

## Development Workflow

After modifying any code:

1. `make lint` - Quick formatting and auto-fix (ruff + pre-commit)
2. `make check` - Full validation (lock file, pre-commit, Pyright type checking, deptry)
3. `make test` - Run test suite with coverage

## Key Commands

| Command         | Description                                         |
| --------------- | --------------------------------------------------- |
| `make install`  | Create venv with uv and install pre-commit hooks    |
| `make lint`     | Run pre-commit linters (ruff format/lint)           |
| `make check`    | Full validation: lint + pyright + deptry            |
| `make test`     | Run pytest with coverage (inline snapshot disabled) |
| `make test-fix` | Run pytest with inline snapshot update enabled      |
| `make build`    | Build wheel file                                    |

## Code Style

- **Formatter**: ruff (line-length: 120)
- **Type Checking**: pyright (standard mode)
- **Target Python**: 3.11
- **Import Style**: All imports must be at module level (top of file). Do not use function-level imports except within `TYPE_CHECKING` blocks for avoiding circular dependencies.

## Testing

- Framework: pytest with pytest-asyncio
- Coverage: pytest-cov
- Test location: `tests/`
- **Test Style**: Use standalone functions (not classes)

## Dependencies

Core dependencies:

- pydantic-ai-slim (AI agent framework)
- pydantic / pydantic-settings (data validation and configuration)
- httpx, anyio (async HTTP and concurrency)
- cdp-use (browser automation)
- pillow (image processing)
- jinja2 (template rendering for tool instructions)

Optional dependencies:

- `docker` - Docker sandbox support
- `web` - Web tools (tavily-python, firecrawl-py, markitdown)
- `document` - Document processing (pymupdf, markitdown)

## Environment Configuration

API keys and settings are loaded from environment variables or `.env` file via `pydantic-settings`. See `.env.example` for all available variables.

**Important**: When adding or modifying environment variables, always update `.env.example` as the single source of truth.

## Architecture Reference

### AgentContext and Sessions

See [docs/context.md](docs/context.md) for details:

- Session state management (run_id, timing, user prompts)
- Resumable sessions with `export_state()` and `with_state()`
- Extending `AgentContext` and `ResumableState` for custom fields
- Using `create_agent` with `state` parameter for session restoration

### Toolset Architecture

See [docs/toolset.md](docs/toolset.md) for details:

- Creating custom tools with `BaseTool`
- Hook system (pre/post hooks, global hooks)
- Error handling in post-hooks (exceptions as results)
- Extending `Toolset` via `_call_tool_func` for timeout/retry/custom logic

### Subagent System

See [docs/subagent.md](docs/subagent.md) for details:

- Hierarchical agent architecture and task delegation
- Markdown configuration format (YAML frontmatter + system prompt)
- Tool inheritance and availability rules
- Built-in presets (debugger, explorer, searcher, code-reviewer)

### Skills System

See [docs/skills.md](docs/skills.md) for details:

- Markdown-based instruction files with YAML frontmatter
- Progressive loading (frontmatter only until activation)
- Hot reload with frontmatter change detection
- Pre-scan hook for external skill synchronization

### Environment Management

See [docs/environment.md](docs/environment.md) for details:

- Environment ABC: FileOperator, Shell, ResourceRegistry
- `_setup`/`_teardown` pattern for custom environments
- `LocalEnvironment` and `DockerEnvironment` usage
- `AsyncExitStack` for managing dependent context managers

### Resumable Resources

See [docs/resumable-resources.md](docs/resumable-resources.md) for details:

- `Resource` protocol (requires `close()`) and `ResumableResource` protocol (adds `export_state`/`restore_state`)
- `BaseResource` abstract base class with async `close()` and default export/restore
- Factory-based resource creation and lazy initialization
- State export/restore for session persistence across restarts

### Model Configuration

See [docs/model.md](docs/model.md) for details:

- Native pydantic-ai model strings (direct provider connection)
- Gateway mode (route requests through unified gateway)
- Sticky routing and extra headers

### Logging

See [docs/logging.md](docs/logging.md) for details:

- Global log level: `PAI_AGENT_LOG_LEVEL`
- Module-specific log levels: `PAI_AGENT_LOG_LEVEL_<MODULE_PATH>`
- Use `get_logger(__name__)` to obtain logger

## Examples

| Example                                     | Description                                                               |
| ------------------------------------------- | ------------------------------------------------------------------------- |
| [general.py](examples/general.py)           | Complete production pattern with streaming, HITL, and session persistence |
| [deepresearch.py](examples/deepresearch.py) | Autonomous research agent with web search and content extraction          |
| [browser_use.py](examples/browser_use.py)   | Browser automation with Docker-based headless Chrome sandbox              |

## Quick Start

```python
from pai_agent_sdk.agents import create_agent

async with create_agent("openai:gpt-4o") as runtime:
    result = await runtime.agent.run("Hello", deps=runtime.ctx)
    print(result.output)
```
