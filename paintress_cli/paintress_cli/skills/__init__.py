"""Skills module for pai-agent-sdk."""

from pathlib import Path

# Base directory for skills
SKILLS_BASE_DIR = Path(__file__).parent


def get_skill_dirs() -> list[Path]:
    """Get absolute paths of all skill directories under SKILLS_BASE_DIR."""
    return [p.resolve() for p in SKILLS_BASE_DIR.iterdir() if p.is_dir() and not p.name.startswith("__")]
