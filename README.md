# pai-agent-sdk

[![Release](https://img.shields.io/github/v/release/wh1isper/pai-agent-sdk)](https://img.shields.io/github/v/release/wh1isper/pai-agent-sdk)
[![Build status](https://img.shields.io/github/actions/workflow/status/wh1isper/pai-agent-sdk/main.yml?branch=main)](https://github.com/wh1isper/pai-agent-sdk/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/wh1isper/pai-agent-sdk/branch/main/graph/badge.svg)](https://codecov.io/gh/wh1isper/pai-agent-sdk)
[![Commit activity](https://img.shields.io/github/commit-activity/m/wh1isper/pai-agent-sdk)](https://img.shields.io/github/commit-activity/m/wh1isper/pai-agent-sdk)
[![License](https://img.shields.io/github/license/wh1isper/pai-agent-sdk)](https://img.shields.io/github/license/wh1isper/pai-agent-sdk)

Toolsets and context management for building agents with Pydantic AI.

## Installation

```bash
# Using pip
pip install pai-agent-sdk

# Using uv
uv add pai-agent-sdk

# With Docker sandbox support
pip install pai-agent-sdk[docker]
```

## Configuration

Copy `.env.example` to `.env` and configure your API keys. See [.env.example](.env.example) for all available environment variables.

## Documentation

- [AgentContext & Sessions](docs/context.md) - Session state, resumable sessions, extending context
- [Toolset Architecture](docs/toolset.md) - Create tools, use hooks, handle errors, extend Toolset
- [Custom Environments](docs/environment.md) - Extend context management with custom environments
- [Logging Configuration](docs/logging.md) - Configure SDK logging levels

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.
