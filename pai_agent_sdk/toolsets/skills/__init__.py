"""Skills toolset for loading and managing skills.

Skills are markdown files with YAML frontmatter that provide specialized
instructions for specific tasks. The SkillToolset discovers skills from
FileOperator's allowed paths and injects their metadata into the system prompt.

Example skill file (skills/my-skill/SKILL.md):

    ---
    name: my-skill
    description: Brief description of what this skill does and when to use it.
    ---

    # Detailed Instructions

    When performing this task, follow these steps...

Usage::

    from pai_agent_sdk.toolsets.skills import SkillToolset

    skill_toolset = SkillToolset()

    async with create_agent(
        model="anthropic:claude-sonnet-4",
        tools=[skill_toolset],
    ) as runtime:
        # Skills from all allowed_paths/skills/ directories will be available
        ...
"""

from pai_agent_sdk.toolsets.skills.config import (
    SkillConfig,
    load_skill_from_file,
    load_skills_from_dir,
    parse_skill_markdown,
)
from pai_agent_sdk.toolsets.skills.toolset import PreScanHook, SkillToolset

__all__ = [
    "PreScanHook",
    "SkillConfig",
    "SkillToolset",
    "load_skill_from_file",
    "load_skills_from_dir",
    "parse_skill_markdown",
]
