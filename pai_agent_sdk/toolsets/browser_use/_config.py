"""Configuration management using pydantic-settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class BrowserUseSettings(BaseSettings):
    """Configuration for BrowserUseToolset with environment variable support.

    All settings can be overridden via environment variables with the prefix PAI_AGENT_BROWSER_USE_.
    For example, to set max_retries, use PAI_AGENT_BROWSER_USE_MAX_RETRIES=5.
    """

    model_config = SettingsConfigDict(
        env_prefix="PAI_AGENT_BROWSER_USE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    max_retries: int = 3
    """Maximum retry attempts for tool calls. Defaults to 3."""

    prefix: str | None = None
    """Tool name prefix. Defaults to None."""

    always_use_new_page: bool = False
    """Force create new page instead of reusing existing. Defaults to False."""

    auto_cleanup_page: bool = False
    """Automatically close created page targets on context exit. Defaults to False.

    Can be combined with always_use_new_page=True to create new pages and automatically clean them up.
    When False (default), created pages remain open after context exit, useful for debugging or inspection.
    """
