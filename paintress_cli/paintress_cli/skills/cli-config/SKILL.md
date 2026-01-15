---
name: cli-config
description: Guide for configuring Paintress CLI. Use this skill when users want to configure models, tools, subagents, custom commands, or other CLI settings. Covers both global and project-level configuration.
---

# Paintress CLI Configuration

Configuration is loaded from multiple locations with project-level priority (no merging between levels).

## Configuration Locations

| Level | Location | Priority |
|-------|----------|----------|
| Global | `~/.config/youware-labs/paintress-cli/` | Default |
| Project | `.paintress/` | Overrides global |

## Configuration Files

### config.toml (Global)

Main configuration file for model, display, browser, and subagents.

```toml
[general]
# Model configuration (required)
# Format: "provider:model_name"
model = "anthropic:claude-sonnet-4-5"

# Model settings preset or custom dict
# Presets: anthropic_default, openai_default, google_default
model_settings = "anthropic_default"

# Model config for context management
# Presets: claude_200k, claude_1m, gpt5_270k, gemini_1m
model_cfg = "claude_200k"

# Maximum requests per session
max_requests = 1000

[env]
# Environment variable overrides for API keys
# ANTHROPIC_API_KEY = "sk-ant-..."

[display]
code_theme = "dark"           # "dark" or "light"
max_tool_result_lines = 5
max_arg_length = 100
show_token_usage = true
show_elapsed_time = true

[browser]
# cdp_url = "auto"            # null, "auto", or explicit URL
browser_image = "zenika/alpine-chrome:latest"
browser_timeout = 30

[subagents]
disabled = []                 # Subagents to disable by name
# [subagents.overrides.explorer]
# model = "openai:gpt-4o"
```

### tools.toml (Project)

Project-level tool permissions in `.paintress/tools.toml`:

```toml
[tools]
# Tools requiring user approval before execution
need_approval = ["shell", "replace"]
```

Common patterns:
- `[]` - No approval needed (trust all tools)
- `["shell"]` - Approve shell commands only
- `["shell", "replace", "edit"]` - Approve all code modifications

### mcp.json

MCP server configurations:

```json
{
  "servers": {
    "my-server": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@some/mcp-server"],
      "env": {}
    }
  }
}
```

## Custom Slash Commands

Define custom commands in `config.toml`:

```toml
[commands.deploy]
description = "Deploy to production"
mode = "act"                  # Optional: "act" or "plan"
prompt = """
Please help me deploy to production...
"""
```

Built-in commands: `/init`, `/commit`, `/review`

## Subagent Configuration

Create markdown files in `~/.config/youware-labs/paintress-cli/subagents/`:

```markdown
---
name: my-subagent
description: Brief description shown when selecting tools
instruction: |
  When to use this subagent and what to provide
tools:
  - grep
  - view
optional_tools:
  - shell
model: inherit
---

You are a specialist in [domain].

## Process
1. Step one
2. Step two
```

### Configuration Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique identifier, becomes the tool name |
| `description` | Yes | Shown to model when selecting tools |
| `instruction` | No | Injected into parent's system prompt |
| `tools` | No | Required tools - ALL must be available |
| `optional_tools` | No | Optional tools - included if available |
| `model` | No | `"inherit"` or model name |

### Tool Availability Rules

- **Required tools** (`tools`): Subagent disabled if ANY unavailable
- **Optional tools** (`optional_tools`): Included only if available
- **No tools specified**: Inherits all parent tools

### Builtin Presets

| Preset | Purpose | Required Tools |
|--------|---------|----------------|
| `debugger` | Root cause analysis | glob, grep, view, ls |
| `explorer` | Codebase navigation | glob, grep, view, ls |
| `code-reviewer` | Code quality review | glob, grep, view, ls |
| `searcher` | Web research | search |

## Skills Directory

Skills are loaded from (highest priority last):

1. Built-in: `paintress_cli/skills/` (package bundled)
2. Global: `~/.config/youware-labs/paintress-cli/skills/`
3. Project: `.paintress/skills/`

## Environment Variables

TUI settings can be overridden via `PAINTRESS_*` environment variables:

- `PAINTRESS_CODE_THEME`
- `PAINTRESS_SHOW_TOKEN_USAGE`
- `PAINTRESS_CDP_URL`
- `PAINTRESS_SESSION_DIR`

## Quick Setup

Run `paintress setup` to initialize global configuration with defaults.
