"""Configuration management using pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentContextSettings(BaseSettings):
    """Configuration for AgentContext with environment variable support.

    All settings can be overridden via environment variables with the prefix PAI_AGENT_.
    For example, to set working_dir, use PAI_AGENT_WORKING_DIR=/path/to/dir.
    """

    model_config = SettingsConfigDict(
        env_prefix="PAI_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    working_dir: Path | None = None
    """Working directory for tool path validation. If None, uses current working directory."""

    tmp_base_dir: Path | None = None
    """Base directory for creating the session temporary directory. If None, uses system default."""
