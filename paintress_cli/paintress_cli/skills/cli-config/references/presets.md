# Builtin Subagent Presets

## debugger

```yaml
name: debugger
description: Debugging specialist for errors, test failures, and unexpected behavior.
instruction: |
  Use when encountering error messages, stack traces, test failures, or unexpected behavior.
  Provide: error message, stack trace, steps to reproduce, expected vs actual behavior.
  Returns: root cause analysis, fix recommendations, verification steps.
tools: [glob, grep, view, ls]
optional_tools: [shell, edit, multi_edit, replace]
model: inherit
```

**System Prompt Focus:**
- Information gathering and hypothesis formation
- Evidence-based root cause identification
- Output: Root Cause, Evidence, Recommended Fix, Verification

## explorer

```yaml
name: explorer
description: Codebase exploration specialist for navigation and understanding.
instruction: |
  Use when understanding codebase structure, finding implementations, or locating usages.
  Provide: what to find, starting points, context.
  Returns: file paths, code snippets, pattern summaries.
tools: [glob, grep, view, ls]
optional_tools: [edit, multi_edit, replace]
model: inherit
```

**System Prompt Focus:**
- Exploration strategies (definitions, structure, usage tracing)
- Output: Search Summary, Key Findings, Structure Overview

## code-reviewer

```yaml
name: code-reviewer
description: Code review specialist for quality, security, and maintainability.
instruction: |
  Use after implementing features, before commits, or when refactoring.
  Provide: file paths, code context, specific concerns.
  Returns: issues by severity, fix recommendations, security notes.
tools: [glob, grep, view, ls]
optional_tools: [search, scrape, fetch]
model: inherit
```

**System Prompt Focus:**
- Review checklist (Correctness, Security, Quality, Performance)
- Output: Critical Issues, Warnings, Suggestions, Positive Notes

## searcher

```yaml
name: searcher
description: Web research specialist for documentation and solutions.
instruction: |
  Use for API docs, error solutions, best practices, or current information.
  Provide: specific question, context, constraints.
  Returns: relevant information, code examples, sources.
tools: [search]
optional_tools: [scrape, fetch, edit, multi_edit, replace]
model: inherit
```

**System Prompt Focus:**
- Search strategies (technical, current info, problem solving)
- Output: Research Summary, Key Findings, Additional Resources
