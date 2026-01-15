---
name: subagent-config
description: Guide for creating and configuring custom subagents. Use this skill when users want to create new subagents, modify existing ones, or understand the subagent system. Subagents are specialized AI assistants that the main agent can delegate tasks to, each with focused capabilities and tools.
---

# Subagent Configuration

Subagents are specialized AI assistants that extend the main agent's capabilities through task delegation.

## Quick Start

Create a markdown file in `~/.config/youware-labs/paintress-cli/subagents/`:

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

## Output Format
[Define expected output structure]
```

## Configuration Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique identifier, becomes the tool name |
| `description` | Yes | Shown to model when selecting tools |
| `instruction` | No | Injected into parent's system prompt |
| `tools` | No | Required tools - ALL must be available |
| `optional_tools` | No | Optional tools - included if available |
| `model` | No | `"inherit"` or model name |
| `model_settings` | No | `"inherit"` or preset name |
| `model_cfg` | No | `"inherit"` or preset name |

## Tool Availability Rules

- **Required tools** (`tools`): Subagent disabled if ANY unavailable
- **Optional tools** (`optional_tools`): Included only if available
- **No tools specified**: Inherits all parent tools

## Writing Effective Subagents

### 1. Clear Focus

Each subagent should have a single, well-defined purpose.

### 2. Helpful Instruction

```yaml
instruction: |
  Use this subagent when:
  - [Trigger condition 1]
  - [Trigger condition 2]

  Provide:
  - [Required input 1]

  Returns:
  - [Output description]
```

### 3. Minimal Required Tools

Only require essential tools. Use `optional_tools` for nice-to-have capabilities.

### 4. Structured System Prompt

Include in the markdown body:
- Role definition and expertise
- Step-by-step process
- Output format specification

## Builtin Presets

| Preset | Purpose | Required Tools |
|--------|---------|----------------|
| `debugger` | Root cause analysis | glob, grep, view, ls |
| `explorer` | Codebase navigation | glob, grep, view, ls |
| `code-reviewer` | Code quality review | glob, grep, view, ls |
| `searcher` | Web research | search |

See `references/presets.md` for full configurations.

## Common Tool Patterns

**Research**: `tools: [search]`, `optional_tools: [scrape, fetch]`

**Code Analysis**: `tools: [glob, grep, view, ls]`, `optional_tools: [shell]`

**Code Modification**: `tools: [glob, grep, view, ls]`, `optional_tools: [edit, multi_edit, replace]`

## File Location

Custom subagents: `~/.config/youware-labs/paintress-cli/subagents/*.md`
