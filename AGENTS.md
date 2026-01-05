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
  toolsets/       # Tool implementations (browser_use, filesystem, shell, web, etc.)
  context.py      # Context management
  utils.py        # Utility functions
tests/            # Test suite (pytest)
```

## Development Workflow

After modifying any code:

1. `make lint` - Quick formatting and auto-fix (ruff + pre-commit)
2. `make check` - Full validation (lock file, pre-commit, Pyright type checking, deptry)
3. `make test` - Run test suite with coverage

## Key Commands

| Command        | Description                                      |
| -------------- | ------------------------------------------------ |
| `make install` | Create venv with uv and install pre-commit hooks |
| `make lint`    | Run pre-commit linters (ruff format/lint)        |
| `make check`   | Full validation: lint + pyright + deptry         |
| `make test`    | Run pytest with coverage                         |
| `make build`   | Build wheel file                                 |

## Code Style

- **Formatter**: ruff (line-length: 120)
- **Type Checking**: pyright (standard mode)
- **Target Python**: 3.11

## Testing

- Framework: pytest with pytest-asyncio
- Coverage: pytest-cov
- Test location: `tests/`

## Dependencies

Core dependencies include:

- pydantic-ai-slim (AI agent framework)
- pydantic / pydantic-settings
- httpx, anyio
- cdp-use (browser automation)
- pillow (image processing)

Optional:

- docker (for sandbox features)

## Architecture Notes

- Toolsets extend `base.py` for consistent tool registration
- Context management is centralized in `context.py`
- Agents are modular and located in `agents/` directory
- Logging is centralized in `_logger.py` - see [docs/logging.md](docs/logging.md) for configuration
