"""Tests for skill configuration parsing."""

from pathlib import Path

import pytest

from pai_agent_sdk.toolsets.skills.config import (
    load_skill_from_file,
    load_skills_from_dir,
    parse_skill_markdown,
)

# =============================================================================
# parse_skill_markdown tests
# =============================================================================


def test_parse_skill_markdown_basic():
    """Test parsing a basic skill markdown file."""
    content = """---
name: test-skill
description: A test skill for testing.
---

# Test Skill Content

This is the body content.
"""
    config = parse_skill_markdown(content)

    assert config.name == "test-skill"
    assert config.description == "A test skill for testing."
    assert config.content_hash != ""
    assert config.extra == {}


def test_parse_skill_markdown_with_extra_fields():
    """Test parsing skill with extra frontmatter fields."""
    content = """---
name: advanced-skill
description: An advanced skill.
custom_field: custom_value
another_field: 123
---

Body content here.
"""
    config = parse_skill_markdown(content)

    assert config.name == "advanced-skill"
    assert config.description == "An advanced skill."
    assert config.extra == {"custom_field": "custom_value", "another_field": 123}


def test_parse_skill_markdown_missing_name():
    """Test that missing name raises ValueError."""
    content = """---
description: A skill without a name.
---

Body content.
"""
    with pytest.raises(ValueError, match="Missing required field 'name'"):
        parse_skill_markdown(content)


def test_parse_skill_markdown_missing_description():
    """Test that missing description raises ValueError."""
    content = """---
name: incomplete-skill
---

Body content.
"""
    with pytest.raises(ValueError, match="Missing required field 'description'"):
        parse_skill_markdown(content)


def test_parse_skill_markdown_no_frontmatter():
    """Test that missing frontmatter raises ValueError."""
    content = """# Just Markdown

No YAML frontmatter here.
"""
    with pytest.raises(ValueError, match="expected YAML frontmatter"):
        parse_skill_markdown(content)


def test_parse_skill_markdown_invalid_yaml():
    """Test that invalid YAML raises ValueError."""
    content = """---
name: bad-skill
description: [invalid yaml
---

Body content.
"""
    with pytest.raises(ValueError, match="Invalid YAML"):
        parse_skill_markdown(content)


def test_parse_skill_markdown_non_mapping_frontmatter():
    """Test that non-mapping frontmatter raises TypeError."""
    content = """---
- list
- items
---

Body content.
"""
    with pytest.raises(TypeError, match="must be a mapping"):
        parse_skill_markdown(content)


def test_parse_skill_markdown_with_path():
    """Test parsing with explicit path."""
    content = """---
name: path-skill
description: A skill with path.
---

Body.
"""
    skill_path = Path("/custom/path")
    config = parse_skill_markdown(content, skill_path)

    assert config.path == skill_path


def test_content_hash_changes_with_frontmatter():
    """Test that content hash changes when frontmatter changes."""
    content1 = """---
name: skill
description: Version 1
---

Body.
"""
    content2 = """---
name: skill
description: Version 2
---

Body.
"""
    config1 = parse_skill_markdown(content1)
    config2 = parse_skill_markdown(content2)

    assert config1.content_hash != config2.content_hash


def test_content_hash_same_for_same_frontmatter():
    """Test that content hash is same for identical frontmatter."""
    content1 = """---
name: skill
description: Same description
---

Body 1.
"""
    content2 = """---
name: skill
description: Same description
---

Different body content.
"""
    config1 = parse_skill_markdown(content1)
    config2 = parse_skill_markdown(content2)

    # Hash is based on frontmatter only, so should be the same
    assert config1.content_hash == config2.content_hash


# =============================================================================
# load_skill_from_file tests
# =============================================================================


def test_load_skill_from_file(tmp_path: Path):
    """Test loading skill from SKILL.md file."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text("""---
name: file-skill
description: Loaded from file.
---

Content.
""")

    config = load_skill_from_file(skill_file)

    assert config.name == "file-skill"
    assert config.description == "Loaded from file."
    assert config.path == skill_dir


# =============================================================================
# load_skills_from_dir tests
# =============================================================================


def test_load_skills_from_dir_empty(tmp_path: Path):
    """Test loading from empty directory."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    configs = load_skills_from_dir(skills_dir)

    assert configs == {}


def test_load_skills_from_dir_nonexistent(tmp_path: Path):
    """Test loading from nonexistent directory."""
    configs = load_skills_from_dir(tmp_path / "nonexistent")

    assert configs == {}


def test_load_skills_from_dir_with_subdirs(tmp_path: Path):
    """Test loading skills from subdirectories."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Create skill in subdirectory
    skill1_dir = skills_dir / "skill-one"
    skill1_dir.mkdir()
    (skill1_dir / "SKILL.md").write_text("""---
name: skill-one
description: First skill.
---

Content 1.
""")

    # Create another skill
    skill2_dir = skills_dir / "skill-two"
    skill2_dir.mkdir()
    (skill2_dir / "SKILL.md").write_text("""---
name: skill-two
description: Second skill.
---

Content 2.
""")

    configs = load_skills_from_dir(skills_dir)

    assert len(configs) == 2
    assert "skill-one" in configs
    assert "skill-two" in configs
    assert configs["skill-one"].description == "First skill."
    assert configs["skill-two"].description == "Second skill."


def test_load_skills_from_dir_skips_invalid(tmp_path: Path):
    """Test that invalid skill files are skipped."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Valid skill
    valid_dir = skills_dir / "valid"
    valid_dir.mkdir()
    (valid_dir / "SKILL.md").write_text("""---
name: valid-skill
description: Valid.
---

Content.
""")

    # Invalid skill (missing description)
    invalid_dir = skills_dir / "invalid"
    invalid_dir.mkdir()
    (invalid_dir / "SKILL.md").write_text("""---
name: invalid-skill
---

No description.
""")

    configs = load_skills_from_dir(skills_dir)

    assert len(configs) == 1
    assert "valid-skill" in configs
    assert "invalid-skill" not in configs


def test_load_skills_from_dir_direct_skill_md(tmp_path: Path):
    """Test loading skill from direct SKILL.md in the directory."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Direct SKILL.md in skills_dir
    (skills_dir / "SKILL.md").write_text("""---
name: direct-skill
description: Direct skill in root.
---

Content.
""")

    configs = load_skills_from_dir(skills_dir)

    assert "direct-skill" in configs
