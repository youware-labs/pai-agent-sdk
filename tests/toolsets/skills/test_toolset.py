"""Tests for SkillToolset."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.toolsets.skills import SkillToolset


@pytest.fixture
async def env_with_skills(tmp_path: Path):
    """Create environment with skills directories."""
    # Create main project directory with skills
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    project_skills = project_dir / "skills"
    project_skills.mkdir()

    # Create a skill in project
    skill1_dir = project_skills / "project-skill"
    skill1_dir.mkdir()
    (skill1_dir / "SKILL.md").write_text("""---
name: project-skill
description: A project-specific skill.
---

# Project Skill

Do something specific to this project.
""")

    # Create config directory with skills
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_skills = config_dir / "skills"
    config_skills.mkdir()

    # Create a skill in config
    skill2_dir = config_skills / "global-skill"
    skill2_dir.mkdir()
    (skill2_dir / "SKILL.md").write_text("""---
name: global-skill
description: A global skill available everywhere.
---

# Global Skill

Available across all projects.
""")

    async with LocalEnvironment(
        default_path=project_dir,
        allowed_paths=[project_dir, config_dir],
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            yield ctx


@pytest.fixture
def mock_run_ctx_with_skills(env_with_skills: AgentContext) -> MagicMock:
    """Create mock RunContext with skills environment."""
    mock_ctx = MagicMock(spec=RunContext)
    mock_ctx.deps = env_with_skills
    return mock_ctx


# =============================================================================
# SkillToolset tests
# =============================================================================


async def test_skill_toolset_get_instructions(mock_run_ctx_with_skills: MagicMock):
    """Test that SkillToolset loads and formats skill instructions."""
    toolset = SkillToolset()
    instructions = await toolset.get_instructions(mock_run_ctx_with_skills)

    assert instructions is not None
    assert "<available-skills>" in instructions
    assert "project-skill" in instructions
    assert "global-skill" in instructions
    assert "A project-specific skill" in instructions
    assert "A global skill available everywhere" in instructions
    assert "<skill-usage-instructions>" in instructions


async def test_skill_toolset_no_file_operator():
    """Test that SkillToolset returns None when no file_operator."""
    mock_ctx = MagicMock(spec=RunContext)
    mock_ctx.deps = MagicMock(spec=AgentContext)
    mock_ctx.deps.file_operator = None

    toolset = SkillToolset()
    instructions = await toolset.get_instructions(mock_ctx)

    assert instructions is None


async def test_skill_toolset_no_skills(tmp_path: Path):
    """Test that SkillToolset returns None when no skills found."""
    # Create environment without any skills directories
    async with LocalEnvironment(
        default_path=tmp_path,
        allowed_paths=[tmp_path],
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            mock_ctx = MagicMock(spec=RunContext)
            mock_ctx.deps = ctx

            toolset = SkillToolset()
            instructions = await toolset.get_instructions(mock_ctx)

            assert instructions is None


async def test_skill_toolset_hot_reload(tmp_path: Path):
    """Test that SkillToolset detects changes in skill frontmatter."""
    # Create environment with skills
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    skill_dir = skills_dir / "changing-skill"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"

    # Initial skill content
    skill_file.write_text("""---
name: changing-skill
description: Version 1 description.
---

Content v1.
""")

    async with LocalEnvironment(
        default_path=tmp_path,
        allowed_paths=[tmp_path],
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            mock_ctx = MagicMock(spec=RunContext)
            mock_ctx.deps = ctx

            toolset = SkillToolset()

            # First call - loads skill
            instructions1 = await toolset.get_instructions(mock_ctx)
            assert instructions1 is not None
            assert "Version 1 description" in instructions1

            # Modify skill frontmatter
            skill_file.write_text("""---
name: changing-skill
description: Version 2 description.
---

Content v2.
""")

            # Second call - should detect change and reload
            instructions2 = await toolset.get_instructions(mock_ctx)
            assert instructions2 is not None
            assert "Version 2 description" in instructions2
            assert "Version 1 description" not in instructions2


async def test_skill_toolset_cache_unchanged(tmp_path: Path):
    """Test that SkillToolset uses cache for unchanged skills."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    skill_dir = skills_dir / "stable-skill"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"

    skill_file.write_text("""---
name: stable-skill
description: Stable description.
---

Content.
""")

    async with LocalEnvironment(
        default_path=tmp_path,
        allowed_paths=[tmp_path],
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            mock_ctx = MagicMock(spec=RunContext)
            mock_ctx.deps = ctx

            toolset = SkillToolset()

            # First call
            _ = await toolset.get_instructions(mock_ctx)
            cached_skill = toolset._skills_cache.get("stable-skill")
            assert cached_skill is not None

            # Second call - should reuse cache (same object)
            _ = await toolset.get_instructions(mock_ctx)
            cached_skill2 = toolset._skills_cache.get("stable-skill")

            assert cached_skill is cached_skill2  # Same object reference


async def test_skill_toolset_custom_dir_name(tmp_path: Path):
    """Test SkillToolset with custom skills directory name."""
    custom_skills_dir = tmp_path / "custom-skills"
    custom_skills_dir.mkdir()

    skill_dir = custom_skills_dir / "custom-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("""---
name: custom-skill
description: Found in custom directory.
---

Content.
""")

    async with LocalEnvironment(
        default_path=tmp_path,
        allowed_paths=[tmp_path],
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            mock_ctx = MagicMock(spec=RunContext)
            mock_ctx.deps = ctx

            # Default dir name - should not find skill
            default_toolset = SkillToolset()
            instructions_default = await default_toolset.get_instructions(mock_ctx)
            assert instructions_default is None

            # Custom dir name - should find skill
            custom_toolset = SkillToolset(skills_dir_name="custom-skills")
            instructions_custom = await custom_toolset.get_instructions(mock_ctx)
            assert instructions_custom is not None
            assert "custom-skill" in instructions_custom


def test_skill_toolset_tool_defs():
    """Test that SkillToolset provides no tools."""
    toolset = SkillToolset()

    mock_ctx = MagicMock(spec=RunContext)
    mock_ctx.deps = MagicMock(spec=AgentContext)
    mock_ctx.deps.file_operator = None

    # get_tools is async, but we can check the toolset has id property
    assert toolset.id is None


async def test_skill_toolset_call_tool_raises():
    """Test that calling a tool raises NotImplementedError."""
    toolset = SkillToolset()

    mock_ctx = MagicMock(spec=RunContext)

    with pytest.raises(NotImplementedError, match="does not provide tools"):
        await toolset.call_tool("any_tool", {}, mock_ctx, None)


async def test_skill_toolset_pre_scan_hook_sync(tmp_path: Path):
    """Test that SkillToolset calls sync pre_scan_hook with (toolset, ctx)."""
    hook_called = []

    def sync_hook(toolset: SkillToolset, ctx: RunContext[AgentContext]):
        hook_called.append((toolset, ctx))

    async with LocalEnvironment(
        default_path=tmp_path,
        allowed_paths=[tmp_path],
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            mock_ctx = MagicMock(spec=RunContext)
            mock_ctx.deps = ctx

            toolset = SkillToolset(pre_scan_hook=sync_hook)
            await toolset.get_instructions(mock_ctx)

            assert len(hook_called) == 1
            assert hook_called[0][0] is toolset
            assert hook_called[0][1] is mock_ctx


async def test_skill_toolset_pre_scan_hook_async(tmp_path: Path):
    """Test that SkillToolset calls async pre_scan_hook with (toolset, ctx)."""
    hook_called = []

    async def async_hook(toolset: SkillToolset, ctx: RunContext[AgentContext]):
        hook_called.append((toolset, ctx))

    async with LocalEnvironment(
        default_path=tmp_path,
        allowed_paths=[tmp_path],
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            mock_ctx = MagicMock(spec=RunContext)
            mock_ctx.deps = ctx

            toolset = SkillToolset(pre_scan_hook=async_hook)
            await toolset.get_instructions(mock_ctx)

            assert len(hook_called) == 1
            assert hook_called[0][0] is toolset
            assert hook_called[0][1] is mock_ctx


async def test_skill_toolset_pre_scan_hook_accesses_config(tmp_path: Path):
    """Test that pre_scan_hook can access toolset config."""
    captured_dir_name = []

    def hook(toolset: SkillToolset, ctx: RunContext[AgentContext]):
        captured_dir_name.append(toolset.skills_dir_name)

    async with LocalEnvironment(
        default_path=tmp_path,
        allowed_paths=[tmp_path],
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            mock_ctx = MagicMock(spec=RunContext)
            mock_ctx.deps = ctx

            toolset = SkillToolset(skills_dir_name="custom-dir", pre_scan_hook=hook)
            await toolset.get_instructions(mock_ctx)

            assert captured_dir_name == ["custom-dir"]
