## Project Overview

**pai-agent-sdk** is a Python library providing toolsets and context management for building agents with Pydantic AI.

- **Language**: Python 3.11+
- **Package Manager**: uv
- **Build System**: hatchling

## Project Structure

```
pai_agent_sdk/
  agents/         # Agent implementations (main, search, reasoning, compact, etc.)
  filters/        # Filter utilities
  sandbox/        # Sandbox environments (browser, shell)
  skills/         # Skill definitions
  stream/         # Stream processing
  toolsets/       # Tool implementations
    __init__.py   # Re-exports from core.base and browser_use
    browser_use/  # BrowserUse toolset (independent)
    core/         # Core toolsets collection
      base.py     # Base classes for toolsets (BaseTool, Toolset, etc.)
      content/    # Content loading tools (load)
      context/    # Context management tools (handoff)
      document/   # Document processing tools
      enhance/    # Enhancement tools (todo, thinking)
      filesystem/ # File system operation tools
      multimodal/ # Multimodal processing tools (read_image, read_video)
      shell/      # Shell command execution tools
      subagent/   # Sub-agent delegation tools
      web/        # Web interaction tools
  context.py      # Context management (AgentContext, ModelConfig, ToolConfig)
  utils.py        # Utility functions
tests/            # Test suite (pytest)
  toolsets/
    browser_use/  # BrowserUse tests
    core/         # Core toolsets tests
      content/    # content module tests
      enhance/    # enhance module tests
      test_base.py # base module tests
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

Core dependencies include:

- pydantic-ai-slim (AI agent framework)
- pydantic / pydantic-settings
- httpx, anyio
- cdp-use (browser automation)
- pillow (image processing)
- jinja2 (template rendering for tool instructions)

Optional:

- docker (for sandbox features)
- web: tavily-python, firecrawl-py, markitdown (for web tools)

## Environment Configuration

API keys and settings are loaded from environment variables or `.env` file via `pydantic-settings`. See [.env.example](.env.example) for all available variables.

**Important**: When adding or modifying environment variables or default configurations, always update `.env.example` to keep it as the single source of truth.

## Architecture Notes

- Toolsets extend `base.py` for consistent tool registration
- Context management is centralized in `context.py`
- Agents are modular and located in `agents/` directory
- Logging is centralized in `_logger.py` - see [docs/logging.md](docs/logging.md) for configuration

## AgentContext and Sessions

See [docs/context.md](docs/context.md) for detailed AgentContext documentation, including:

- Session state management (run_id, timing, user prompts)
- Resumable sessions with `export_state()` and `with_state()`
- Extending `AgentContext` and `ResumableState` for custom fields
- Using `create_agent` with `state` parameter for session restoration

## Toolset Architecture

See [docs/toolset.md](docs/toolset.md) for detailed Toolset documentation, including:

- Creating custom tools with `BaseTool`
- Hook system (pre/post hooks, global hooks)
- Error handling in post-hooks (exceptions as results)
- Extending `Toolset` via `_call_tool_func` for timeout/retry/custom logic

## Async Context Manager Patterns

See [docs/environment.md](docs/environment.md) for detailed Environment architecture documentation, including:

- Using `AsyncExitStack` for dependent context managers
- Resource management with `ResourceRegistry`
- The `_setup`/`_teardown` pattern for custom environments
- `LocalEnvironment` and `DockerEnvironment` usage
