# PAI Agent SDK

> PAI is for Paintress or Pydantic AI

[![Release](https://img.shields.io/github/v/release/youware-labs/pai-agent-sdk)](https://img.shields.io/github/v/release/youware-labs/pai-agent-sdk)
[![Build status](https://img.shields.io/github/actions/workflow/status/youware-labs/pai-agent-sdk/main.yml?branch=main)](https://github.com/youware-labs/pai-agent-sdk/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/youware-labs/pai-agent-sdk/branch/main/graph/badge.svg)](https://codecov.io/gh/youware-labs/pai-agent-sdk)
[![Commit activity](https://img.shields.io/github/commit-activity/m/youware-labs/pai-agent-sdk)](https://img.shields.io/github/commit-activity/m/youware-labs/pai-agent-sdk)
[![License](https://img.shields.io/github/license/youware-labs/pai-agent-sdk)](https://img.shields.io/github/license/youware-labs/pai-agent-sdk)

Application framework for building AI agents with [Pydantic AI](https://ai.pydantic.dev/). Used to build the next-generation agent system of [YouWare](https://www.youware.com).

## Key Features

- **Environment-based Architecture**: Inject file operations, shell access, and resources via `Environment` for clean separation of concerns (supports LocalEnvironment and DockerEnvironment)
- **Resumable Sessions**: Export and restore `AgentContext` state for multi-turn conversations across restarts
- **Hierarchical Agents**: Subagent system with task delegation, tool inheritance, and markdown-based configuration
- **Skills System**: Markdown-based instruction files with hot reload and progressive loading
- **Human-in-the-Loop**: Built-in approval workflows for sensitive tool operations
- **Toolset Architecture**: Extensible tool system with pre/post hooks for logging, validation, and error handling
- **Resumable Resources**: Export and restore resource states (like browser sessions) across process restarts
- **Browser Automation**: Docker-based headless Chrome sandbox for safe browser automation
- **Streaming Support**: Real-time streaming of agent responses and tool executions

## Installation

```bash
# Recommended: install with all optional dependencies
pip install pai-agent-sdk[all]
uv add pai-agent-sdk[all]

# Or install individual extras as needed
pip install pai-agent-sdk[docker]    # Docker sandbox support
pip install pai-agent-sdk[web]       # Web tools (tavily, firecrawl, markitdown)
pip install pai-agent-sdk[document]  # Document processing (pymupdf, markitdown)
```

## Project Structure

This repository contains:

- **pai_agent_sdk/** - Core SDK with environment abstraction, toolsets, and session management
- **paintress_cli/** - Reference CLI implementation with TUI for interactive agent sessions
- **examples/** - Code examples demonstrating SDK features
- **docs/** - Documentation for SDK architecture and APIs

## Quick Start

### Using the SDK

```python
from pai_agent_sdk.agents import create_agent, stream_agent

# create_agent returns AgentRuntime (not a context manager)
runtime = create_agent("openai:gpt-4o")

# stream_agent manages runtime lifecycle automatically
async with stream_agent(runtime, "Hello") as streamer:
    async for event in streamer:
        print(event)
```

### Using Paintress CLI

For a ready-to-use terminal interface, try [paintress-cli](paintress_cli/README.md) - a TUI reference implementation built on top of pai-agent-sdk:

```bash
# Run directly with uvx (no installation needed)
uvx paintress-cli

# Or install globally
uv tool install paintress-cli
pip install paintress-cli
```

Features:

- Rich terminal UI with syntax highlighting and streaming output
- Built-in tool approval workflows (human-in-the-loop)
- Session management with conversation history
- Browser automation support via Docker sandbox
- MCP (Model Context Protocol) server integration

## Examples

Check out the [examples/](examples/) directory:

| Example                                     | Description                                                             |
| ------------------------------------------- | ----------------------------------------------------------------------- |
| [general.py](examples/general.py)           | Complete pattern with streaming, HITL approval, and session persistence |
| [deepresearch.py](examples/deepresearch.py) | Autonomous research agent with web search and content extraction        |
| [browser_use.py](examples/browser_use.py)   | Browser automation with Docker-based headless Chrome sandbox            |

## For Agent Users

If you're using an AI agent (e.g., Claude, Cursor) that supports skills:

- **Clone this repo**: The [SKILL.md](SKILL.md) file in the repository root provides comprehensive guidance for agents
- **Download release package**: Get the latest `SKILL.zip` from the [Releases](https://github.com/youware-labs/pai-agent-sdk/releases) page (automatically built during each release)

## Configuration

Copy `examples/.env.example` to `examples/.env` and configure your API keys.

## Documentation

- [AgentContext & Sessions](docs/context.md) - Session state, resumable sessions, extending context
- [Toolset Architecture](docs/toolset.md) - Create tools, use hooks, handle errors, extend Toolset
- [Subagent System](docs/subagent.md) - Hierarchical agents, builtin presets, markdown configuration
- [Skills System](docs/skills.md) - Markdown-based skills, hot reload, pre-scan hooks
- [Custom Environments](docs/environment.md) - Environment lifecycle, resource management
- [Resumable Resources](docs/resumable-resources.md) - Export and restore resource states across restarts
- [Model Configuration](docs/model.md) - Provider setup, gateway mode
- [Logging Configuration](docs/logging.md) - Configure SDK logging levels

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.
