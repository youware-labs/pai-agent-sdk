"""Skills directory management.

Skills are loaded from the following locations (in order of priority):
1. Built-in skills: paintress_cli/skills/ (shipped with package)
2. Global skills: ~/.config/youware-labs/paintress-cli/skills/
3. Project skills: .paintress/skills/

Later directories take precedence over earlier ones for skills with the same name.
"""

from pathlib import Path

from paintress_cli.config import ConfigManager

# Built-in skills directory (shipped with package)
BUILTIN_SKILLS_DIR = Path(__file__).parent

# Global skills directory (from config)
GLOBAL_SKILLS_DIR = ConfigManager.DEFAULT_CONFIG_DIR / "skills"

# Project config directory
PROJECT_CONFIG_DIR = ConfigManager.PROJECT_CONFIG_DIR


def get_skills_dirs(working_dir: Path | None = None) -> list[Path]:
    """Get list of skills directories in priority order.

    Args:
        working_dir: Working directory for project-level skills.
                    Defaults to current working directory.

    Returns:
        List of skill directories, from lowest to highest priority.
        Later entries override earlier ones.
    """
    if working_dir is None:
        working_dir = Path.cwd()

    return [
        BUILTIN_SKILLS_DIR,  # 1. Built-in (lowest priority)
        GLOBAL_SKILLS_DIR,  # 2. Global config
        working_dir / PROJECT_CONFIG_DIR / "skills",  # 3. Project (highest priority)
    ]


def list_skills(working_dir: Path | None = None) -> dict[str, Path]:
    """List all available skills.

    Args:
        working_dir: Working directory for project-level skills.

    Returns:
        Dict mapping skill name to its directory path.
        If multiple directories contain the same skill, higher priority wins.
    """
    skills: dict[str, Path] = {}

    for skills_dir in get_skills_dirs(working_dir):
        if not skills_dir.exists():
            continue

        for entry in skills_dir.iterdir():
            if entry.is_dir() and not entry.name.startswith(("_", ".")):
                skills[entry.name] = entry

    return skills


def get_skill_path(name: str, working_dir: Path | None = None) -> Path | None:
    """Get path to a specific skill.

    Args:
        name: Skill name (directory name).
        working_dir: Working directory for project-level skills.

    Returns:
        Path to the skill directory, or None if not found.
    """
    # Search in reverse order (highest priority first)
    for skills_dir in reversed(get_skills_dirs(working_dir)):
        skill_path = skills_dir / name
        if skill_path.exists() and skill_path.is_dir():
            return skill_path

    return None


__all__ = [
    "BUILTIN_SKILLS_DIR",
    "GLOBAL_SKILLS_DIR",
    "get_skill_path",
    "get_skills_dirs",
    "list_skills",
]
