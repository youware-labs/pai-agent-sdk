"""Skill configuration parsing utilities.

This module handles parsing markdown files with YAML frontmatter into SkillConfig objects.
Skills use a format similar to subagents but only require name and description in the frontmatter.

Skill file format:
    ---
    name: my-skill
    description: Brief description of what this skill does and when to use it.
    ---

    # Skill Content

    The markdown body contains the full skill instructions that will be loaded
    when the skill is activated.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SkillConfig(BaseModel):
    """Parsed skill configuration from markdown file.

    This model represents the configuration extracted from a SKILL.md file
    with YAML frontmatter. Only name and description are required in the
    frontmatter; the body content is loaded separately when the skill is activated.
    """

    name: str
    """Unique name for the skill, used for identification."""

    description: str
    """Description shown to the model when selecting skills. Should explain
    what the skill does and when to use it."""

    path: Path
    """Path to the skill directory containing SKILL.md."""

    content_hash: str = ""
    """Hash of the frontmatter content for change detection."""

    extra: dict[str, Any] = Field(default_factory=dict)
    """Additional frontmatter fields for extensibility."""


def _compute_frontmatter_hash(yaml_content: str) -> str:
    """Compute hash of frontmatter content for change detection.

    Only hashes the frontmatter (name + description) since that's what
    affects the system prompt. Body content changes don't invalidate cache.

    Args:
        yaml_content: The YAML frontmatter string.

    Returns:
        SHA256 hash of the frontmatter content.
    """
    return hashlib.sha256(yaml_content.encode("utf-8")).hexdigest()


def parse_skill_markdown(content: str, skill_path: Path | None = None) -> SkillConfig:
    """Parse skill configuration from markdown with YAML frontmatter.

    The markdown file should have YAML frontmatter delimited by '---' at the
    start and end. Only name and description are required.

    Args:
        content: Markdown content with YAML frontmatter.
        skill_path: Path to the skill directory (optional).

    Returns:
        SkillConfig with parsed configuration.

    Raises:
        ValueError: If the content doesn't have valid YAML frontmatter.
        TypeError: If YAML frontmatter is not a mapping.

    Example::

        config = parse_skill_markdown('''
        ---
        name: code-reviewer
        description: Review code for quality and best practices.
        ---

        # Code Review Guidelines

        When reviewing code, check for...
        ''')
    """
    # Match YAML frontmatter pattern
    pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
    match = re.match(pattern, content.strip(), re.DOTALL)

    if not match:
        msg = "Invalid markdown format: expected YAML frontmatter delimited by '---'"
        raise ValueError(msg)

    yaml_content = match.group(1)
    # body_content = match.group(2).strip()  # Not used in frontmatter-only parsing

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

    # Extract known fields and put the rest in extra
    name = frontmatter.pop("name")
    description = frontmatter.pop("description")
    extra = frontmatter  # Remaining fields

    return SkillConfig(
        name=name,
        description=description,
        path=skill_path or Path("."),
        content_hash=_compute_frontmatter_hash(yaml_content),
        extra=extra,
    )


def load_skill_from_file(path: Path | str) -> SkillConfig:
    """Load skill configuration from a SKILL.md file.

    Args:
        path: Path to the SKILL.md file.

    Returns:
        SkillConfig with parsed configuration.
    """
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    # skill_path is the directory containing SKILL.md
    skill_path = path.parent if path.name == "SKILL.md" else path
    return parse_skill_markdown(content, skill_path)


def load_skills_from_dir(dir_path: Path | str) -> dict[str, SkillConfig]:
    """Load all skill configurations from a directory.

    Scans for skills in two formats:
    1. Direct SKILL.md files in the directory
    2. Subdirectories containing SKILL.md

    Args:
        dir_path: Path to the directory containing skills.

    Returns:
        Dict mapping skill names to their configurations.
    """
    dir_path = Path(dir_path)
    configs: dict[str, SkillConfig] = {}

    if not dir_path.exists() or not dir_path.is_dir():
        return configs

    # Check for direct SKILL.md in dir_path
    direct_skill = dir_path / "SKILL.md"
    if direct_skill.exists():
        try:
            config = load_skill_from_file(direct_skill)
            configs[config.name] = config
        except (ValueError, TypeError):
            pass  # Skip invalid files

    # Check subdirectories for SKILL.md
    for subdir in dir_path.iterdir():
        if subdir.is_dir():
            skill_file = subdir / "SKILL.md"
            if skill_file.exists():
                try:
                    config = load_skill_from_file(skill_file)
                    configs[config.name] = config
                except (ValueError, TypeError):
                    pass  # Skip invalid files

    return configs
