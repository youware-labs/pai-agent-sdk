"""Tests for subagents.config module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pai_agent_sdk.presets import INHERIT
from pai_agent_sdk.subagents.config import (
    SubagentConfig,
    load_subagent_from_file,
    load_subagents_from_dir,
    parse_subagent_markdown,
)


def test_parse_subagent_markdown_basic() -> None:
    """Test basic parsing of subagent markdown."""
    content = """---
name: test-agent
description: A test agent
---

You are a test agent.
"""
    config = parse_subagent_markdown(content)

    assert config.name == "test-agent"
    assert config.description == "A test agent"
    assert config.system_prompt == "You are a test agent."
    assert config.tools is None
    assert config.model is None
    assert config.model_settings is None
    assert config.instruction is None


def test_parse_subagent_markdown_full() -> None:
    """Test parsing with all fields."""
    content = """---
name: debugger
description: Debug code issues
instruction: Use this tool for debugging
tools:
  - grep
  - view
  - edit
model: anthropic:claude-sonnet-4
model_settings: anthropic_high
---

You are a debugging expert.

Help users debug their code.
"""
    config = parse_subagent_markdown(content)

    assert config.name == "debugger"
    assert config.description == "Debug code issues"
    assert config.instruction == "Use this tool for debugging"
    assert config.tools == ["grep", "view", "edit"]
    assert config.model == "anthropic:claude-sonnet-4"
    assert config.model_settings == "anthropic_high"
    assert "debugging expert" in config.system_prompt


def test_parse_subagent_markdown_tools_as_string() -> None:
    """Test parsing tools as comma-separated string."""
    content = """---
name: test
description: Test
tools: tool1, tool2, tool3
---

Prompt
"""
    config = parse_subagent_markdown(content)
    assert config.tools == ["tool1", "tool2", "tool3"]


def test_parse_subagent_markdown_inherit_values() -> None:
    """Test parsing with 'inherit' values."""
    content = """---
name: test
description: Test
model: inherit
model_settings: inherit
---

Prompt
"""
    config = parse_subagent_markdown(content)
    assert config.model == INHERIT
    assert config.model_settings == INHERIT


def test_parse_subagent_markdown_dict_model_settings() -> None:
    """Test parsing with dict model_settings."""
    content = """---
name: test
description: Test
model_settings:
  temperature: 0.5
  max_tokens: 4096
---

Prompt
"""
    config = parse_subagent_markdown(content)
    assert config.model_settings == {"temperature": 0.5, "max_tokens": 4096}


def test_parse_subagent_markdown_missing_frontmatter() -> None:
    """Test error when frontmatter is missing."""
    content = "Just some content without frontmatter"

    with pytest.raises(ValueError, match="expected YAML frontmatter"):
        parse_subagent_markdown(content)


def test_parse_subagent_markdown_missing_name() -> None:
    """Test error when name is missing."""
    content = """---
description: Test
---

Prompt
"""
    with pytest.raises(ValueError, match="Missing required field 'name'"):
        parse_subagent_markdown(content)


def test_parse_subagent_markdown_missing_description() -> None:
    """Test error when description is missing."""
    content = """---
name: test
---

Prompt
"""
    with pytest.raises(ValueError, match="Missing required field 'description'"):
        parse_subagent_markdown(content)


def test_load_subagent_from_file(tmp_path: Path) -> None:
    """Test loading subagent from file."""
    md_file = tmp_path / "test.md"
    md_file.write_text("""---
name: file-agent
description: Loaded from file
---

File content
""")

    config = load_subagent_from_file(md_file)
    assert config.name == "file-agent"
    assert config.description == "Loaded from file"


def test_load_subagents_from_dir(tmp_path: Path) -> None:
    """Test loading all subagents from directory."""
    # Create multiple md files
    (tmp_path / "agent1.md").write_text("""---
name: agent1
description: First agent
---

Agent 1
""")
    (tmp_path / "agent2.md").write_text("""---
name: agent2
description: Second agent
---

Agent 2
""")
    # Create invalid file (should be skipped)
    (tmp_path / "invalid.md").write_text("No frontmatter")
    # Create non-md file (should be ignored)
    (tmp_path / "readme.txt").write_text("Readme")

    configs = load_subagents_from_dir(tmp_path)

    assert len(configs) == 2
    assert "agent1" in configs
    assert "agent2" in configs
    assert configs["agent1"].description == "First agent"
    assert configs["agent2"].description == "Second agent"


def test_subagent_config_model() -> None:
    """Test SubagentConfig model validation."""
    config = SubagentConfig(
        name="test",
        description="Test description",
        system_prompt="Test prompt",
    )
    assert config.name == "test"
    assert config.tools is None


def test_parse_subagent_markdown_with_model_cfg() -> None:
    """Test parsing subagent markdown with model_cfg."""
    content = """---
name: fast_searcher
description: Quick search with smaller context
tools:
  - search_with_tavily
model_cfg:
  context_window: 50000
  compact_threshold: 0.80
  max_images: 5
---

You are a fast search specialist.
"""
    config = parse_subagent_markdown(content)

    assert config.name == "fast_searcher"
    assert config.model_cfg == {
        "context_window": 50000,
        "compact_threshold": 0.80,
        "max_images": 5,
    }


def test_parse_subagent_markdown_with_model_cfg_preset() -> None:
    """Test parsing subagent markdown with model_cfg as preset string."""
    content = """---
name: claude_agent
description: Claude optimized agent
model_cfg: claude_200k
---

You are an agent optimized for Claude.
"""
    config = parse_subagent_markdown(content)

    assert config.name == "claude_agent"
    assert config.model_cfg == "claude_200k"


def test_parse_subagent_markdown_with_model_cfg_inherit() -> None:
    """Test parsing subagent markdown with model_cfg set to inherit."""
    content = """---
name: inheriting_agent
description: Agent that inherits model config
model_cfg: inherit
---

You inherit configuration from parent.
"""
    config = parse_subagent_markdown(content)

    assert config.name == "inheriting_agent"
    assert config.model_cfg == INHERIT
