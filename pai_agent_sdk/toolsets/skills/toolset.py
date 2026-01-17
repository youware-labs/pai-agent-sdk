"""Skill toolset for loading and managing skills.

This module provides the SkillToolset which:
1. Scans FileOperator's allowed_paths for skills/ directories
2. Injects skill metadata (name + description) into system prompt via get_instructions
3. Supports hot-reload by detecting frontmatter changes between requests

Architecture Notes:
    - FileOperator's allowed_paths and Shell's environment are assumed to be aligned,
      so skills can read files and execute commands in the same context.
    - Hot-reload only triggers at request boundaries (in get_instructions), not during
      agent execution, to preserve context cache stability within a request.
    - Only frontmatter (name + description) is loaded into system prompt. Full skill
      content is loaded on-demand when the skill is activated.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.base import BaseToolset
from pai_agent_sdk.toolsets.skills.config import SkillConfig, load_skills_from_dir

if TYPE_CHECKING:
    from agent_environment import FileOperator

logger = get_logger(__name__)

# Default subdirectory name to scan for skills
SKILLS_DIR_NAME = "skills"


class SkillToolset(BaseToolset[AgentContext]):
    """Toolset that manages skills from FileOperator's allowed paths.

    This toolset scans for skills in the `skills/` subdirectory of each
    allowed path in FileOperator. Skills are markdown files with YAML
    frontmatter containing name and description.

    Features:
        - Auto-discovery: Scans all allowed_paths for skills/ directories
        - Hot-reload: Detects frontmatter changes at request boundaries
        - Lazy loading: Only frontmatter is loaded; body content loaded on activation

    Example:
        FileOperator with allowed_paths:
            - /home/user/project (default_path)
            - /home/user/.config/myapp

        Will scan for skills in:
            - /home/user/project/skills/
            - /home/user/.config/myapp/skills/

    Usage::

        from pai_agent_sdk.toolsets.skills import SkillToolset

        skill_toolset = SkillToolset()

        async with create_agent(
            model="anthropic:claude-sonnet-4",
            tools=[skill_toolset],
        ) as runtime:
            ...
    """

    def __init__(self, skills_dir_name: str = SKILLS_DIR_NAME, *, toolset_id: str | None = None) -> None:
        """Initialize SkillToolset.

        Args:
            skills_dir_name: Name of the subdirectory to scan for skills.
                Defaults to "skills".
            toolset_id: Optional unique ID for this toolset instance.
        """
        self._skills_dir_name = skills_dir_name
        self._skills_cache: dict[str, SkillConfig] = {}
        self._last_scan_paths: frozenset[Path] = frozenset()
        self._toolset_id = toolset_id

    @property
    def id(self) -> str | None:
        """Return the toolset ID."""
        return self._toolset_id

    async def get_tools(self, ctx: RunContext[AgentContext]) -> dict[str, Any]:
        """Return empty dict - SkillToolset provides instructions only, no tools."""
        return {}

    def _get_skills_directories(self, file_operator: FileOperator) -> list[Path]:
        """Get all skills directories from FileOperator's allowed paths.

        Args:
            file_operator: The FileOperator instance.

        Returns:
            List of paths to skills directories that exist.
        """
        skills_dirs: list[Path] = []

        # Access the internal _allowed_paths attribute
        allowed_paths: list[Path] = file_operator._allowed_paths

        for allowed_path in allowed_paths:
            skills_dir = allowed_path / self._skills_dir_name
            if skills_dir.exists() and skills_dir.is_dir():
                skills_dirs.append(skills_dir)
                logger.debug(f"Found skills directory: {skills_dir}")

        return skills_dirs

    def _scan_skills(self, file_operator: FileOperator) -> dict[str, SkillConfig]:
        """Scan all skills directories and load skill configurations.

        This method implements hot-reload by comparing frontmatter hashes.
        If a skill's frontmatter hasn't changed, the cached config is reused.

        Args:
            file_operator: The FileOperator instance.

        Returns:
            Dict mapping skill names to their configurations.
        """
        skills_dirs = self._get_skills_directories(file_operator)
        current_paths = frozenset(skills_dirs)

        # Check if paths changed (directories added/removed)
        paths_changed = current_paths != self._last_scan_paths
        if paths_changed:
            logger.debug("Skills directories changed, rescanning all")
            self._last_scan_paths = current_paths

        new_skills: dict[str, SkillConfig] = {}

        for skills_dir in skills_dirs:
            dir_skills = load_skills_from_dir(skills_dir)

            for name, config in dir_skills.items():
                # Check if skill already cached with same hash (hot-reload check)
                cached = self._skills_cache.get(name)
                if cached and cached.content_hash == config.content_hash:
                    # Frontmatter unchanged, reuse cached config
                    new_skills[name] = cached
                    logger.debug(f"Skill '{name}' unchanged, using cache")
                else:
                    # New or changed skill
                    new_skills[name] = config
                    if cached:
                        logger.info(f"Skill '{name}' frontmatter changed, reloading")
                    else:
                        logger.info(f"Discovered new skill: '{name}'")

        # Update cache
        self._skills_cache = new_skills

        return new_skills

    def _format_skills_instruction(self, skills: dict[str, SkillConfig]) -> str | None:
        """Format skills metadata for system prompt injection.

        Args:
            skills: Dict of skill name to SkillConfig.

        Returns:
            Formatted instruction string, or None if no skills.
        """
        if not skills:
            return None

        lines = ["<available-skills>"]

        for name, config in sorted(skills.items()):
            lines.append(f'<skill name="{name}">')
            lines.append(f"  <description>{config.description}</description>")
            lines.append(f"  <path>{config.path}</path>")
            lines.append("</skill>")

        lines.append("</available-skills>")
        lines.append("")
        lines.append("<skill-usage-instructions>")
        lines.append("When a user request matches a skill's description:")
        lines.append("1. Read the skill's SKILL.md file to get detailed instructions")
        lines.append("2. Follow the skill's guidelines to complete the task")
        lines.append("3. Use available file and shell tools to execute the skill's instructions")
        lines.append("</skill-usage-instructions>")

        return "\n".join(lines)

    def get_instructions(self, ctx: RunContext[AgentContext]) -> str | None:
        """Get skill instructions to inject into system prompt.

        This method is called at the start of each request, providing the
        opportunity for hot-reload detection. If skill frontmatter has changed
        since the last request, the updated metadata will be injected.

        Args:
            ctx: The run context containing AgentContext with file_operator.

        Returns:
            Formatted skill metadata string, or None if no skills or no file_operator.
        """
        file_operator = ctx.deps.file_operator
        if file_operator is None:
            logger.debug("SkillToolset: No file_operator available, skipping skill scan")
            return None

        # Scan for skills (with hot-reload detection)
        skills = self._scan_skills(file_operator)

        if not skills:
            logger.debug("SkillToolset: No skills found in any allowed path")
            return None

        logger.debug(f"SkillToolset: Found {len(skills)} skill(s): {list(skills.keys())}")

        return self._format_skills_instruction(skills)

    # -------------------------------------------------------------------------
    # AbstractToolset implementation (no tools, just instructions)
    # -------------------------------------------------------------------------

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentContext],
        tool: Any,
    ) -> Any:
        """Not implemented - SkillToolset provides no tools."""
        msg = f"SkillToolset does not provide tools, received call for '{name}'"
        raise NotImplementedError(msg)
