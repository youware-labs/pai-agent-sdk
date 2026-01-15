"""Subagent configuration parsing utilities.

This module handles parsing markdown files with YAML frontmatter into SubagentConfig objects.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class SubagentConfig(BaseModel):
    """Parsed subagent configuration from markdown file.

    This model represents the configuration extracted from a markdown file
    with YAML frontmatter.
    """

    name: str
    """Unique name for the subagent, used as tool name."""

    description: str
    """Description shown to the model when selecting tools."""

    instruction: str | None = None
    """Optional instruction injected into system prompt."""

    system_prompt: str
    """The markdown body content, used as the subagent's system prompt."""

    tools: list[str] | None = None
    """Required tools from parent toolset. ALL must be available for subagent to be enabled."""

    optional_tools: list[str] | None = None
    """Optional tools from parent toolset. Included if available, not required for availability."""

    model: str | None = None
    """Model to use: 'inherit' (default), or model name like 'anthropic:claude-sonnet-4'."""

    model_settings: str | dict[str, Any] | None = None
    """ModelSettings: 'inherit', preset name (e.g., 'anthropic_medium'), or dict config."""

    model_cfg: str | dict[str, Any] | None = None
    """ModelConfig: 'inherit' (default), preset name (e.g., 'claude_200k'), or dict config."""


def parse_subagent_markdown(content: str) -> SubagentConfig:
    """Parse subagent configuration from markdown with YAML frontmatter.

    The markdown file should have YAML frontmatter delimited by '---' at the
    start and end, followed by the system prompt content.

    Args:
        content: Markdown content with YAML frontmatter.

    Returns:
        SubagentConfig with parsed configuration.

    Raises:
        ValueError: If the content doesn't have valid YAML frontmatter.

    Example::

        config = parse_subagent_markdown('''
        ---
        name: debugger
        description: Debug code issues
        tools:
          - grep
          - view
        model_settings: anthropic_high
        ---

        You are a debugging expert...
        ''')
    """
    # Match YAML frontmatter pattern
    pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
    match = re.match(pattern, content.strip(), re.DOTALL)

    if not match:
        msg = "Invalid markdown format: expected YAML frontmatter delimited by '---'"
        raise ValueError(msg)

    yaml_content = match.group(1)
    body_content = match.group(2).strip()

    # Parse YAML
    try:
        frontmatter = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        msg = f"Invalid YAML in frontmatter: {e}"
        raise ValueError(msg) from e

    if not isinstance(frontmatter, dict):
        msg = "YAML frontmatter must be a mapping"
        raise TypeError(msg)

    # Validate required fields
    if "name" not in frontmatter:
        msg = "Missing required field 'name' in frontmatter"
        raise ValueError(msg)
    if "description" not in frontmatter:
        msg = "Missing required field 'description' in frontmatter"
        raise ValueError(msg)

    # Handle tools field - can be string (comma-separated) or list
    tools = frontmatter.get("tools")
    if isinstance(tools, str):
        tools = [t.strip() for t in tools.split(",") if t.strip()]

    # Handle optional_tools field - can be string (comma-separated) or list
    optional_tools = frontmatter.get("optional_tools")
    if isinstance(optional_tools, str):
        optional_tools = [t.strip() for t in optional_tools.split(",") if t.strip()]

    return SubagentConfig(
        name=frontmatter["name"],
        description=frontmatter["description"],
        instruction=frontmatter.get("instruction"),
        system_prompt=body_content,
        tools=tools,
        optional_tools=optional_tools,
        model=frontmatter.get("model"),
        model_settings=frontmatter.get("model_settings"),
        model_cfg=frontmatter.get("model_cfg"),
    )


def load_subagent_from_file(path: Path | str) -> SubagentConfig:
    """Load subagent configuration from a markdown file.

    Args:
        path: Path to the markdown file.

    Returns:
        SubagentConfig with parsed configuration.
    """
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    return parse_subagent_markdown(content)


def load_subagents_from_dir(dir_path: Path | str) -> dict[str, SubagentConfig]:
    """Load all subagent configurations from a directory.

    Scans the directory for .md files and parses each as a subagent config.

    Args:
        dir_path: Path to the directory containing markdown files.

    Returns:
        Dict mapping subagent names to their configurations.
    """
    dir_path = Path(dir_path)
    configs: dict[str, SubagentConfig] = {}

    for md_file in dir_path.glob("*.md"):
        try:
            config = load_subagent_from_file(md_file)
            configs[config.name] = config
        except ValueError:
            # Skip invalid files
            continue

    return configs
