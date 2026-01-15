"""Configuration management for paintress-cli.

Configuration files are loaded with project-level priority (no merging):

1. **config.toml** (model + TUI settings):
   - Global: ~/.config/youware-labs/paintress-cli/config.toml
   - Project: .paintress/config.toml (overrides global entirely)
   - Contains: model, model_settings, display, steering, session, browser, subagents, env

2. **tools.toml** (tool permissions):
   - Global: ~/.config/youware-labs/paintress-cli/tools.toml
   - Project: .paintress/tools.toml (overrides global entirely)
   - Contains: need_approval list

3. **mcp.json** (MCP server configurations):
   - Global: ~/.config/youware-labs/paintress-cli/mcp.json
   - Project: .paintress/mcp.json (overrides global entirely)

4. **Environment variables** (PAINTRESS_*):
   - TUI configuration overrides only (merged on top of config.toml)
   - Does not affect model settings
"""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# =============================================================================
# Configuration Models
# =============================================================================


class GeneralConfig(BaseModel):
    """General agent configuration (global only)."""

    model: str = ""
    """Default model for main agent. Format: 'provider:model_name'. Empty means not configured."""

    model_settings: str | dict[str, Any] | None = None
    """Model settings: preset name (e.g., 'anthropic_high') or dict of actual values."""

    model_cfg: str | dict[str, Any] | None = None
    """Model config for context management: preset name (e.g., 'claude_200k', 'gemini_1m') or dict."""

    max_requests: int = 1000
    """Maximum requests per session."""

    system_prompt_file: str = ""
    """Path to custom system prompt file. Empty uses built-in default."""

    @property
    def is_configured(self) -> bool:
        """Check if model is configured."""
        return bool(self.model)


class DisplayConfig(BaseModel):
    """Display and rendering configuration."""

    code_theme: Literal["dark", "light"] = "dark"
    """Code highlighting theme."""

    max_tool_result_lines: int = 5
    """Maximum lines to show for tool results."""

    max_arg_length: int = 100
    """Maximum length for tool argument display."""

    max_output_lines: int = 1000
    """Maximum lines to keep in output buffer. Lower values improve performance."""

    show_token_usage: bool = True
    """Show token usage in status bar."""

    show_elapsed_time: bool = True
    """Show elapsed time."""


class BrowserConfig(BaseModel):
    """Browser automation configuration."""

    cdp_url: str | None = None
    """CDP URL for browser automation."""

    browser_image: str = "zenika/alpine-chrome:latest"
    """Docker image for auto-start browser."""

    browser_timeout: int = 30
    """Browser startup timeout in seconds."""


class ToolsConfig(BaseModel):
    """Tool permission configuration (project only)."""

    need_approval: list[str] = Field(default_factory=list)
    """Tools requiring user approval before execution."""


class SubagentOverride(BaseModel):
    """Override settings for a specific subagent."""

    model: str | None = None
    """Override model for this subagent."""

    model_settings: str | dict[str, Any] | None = None
    """Override model settings: preset name or dict of actual values."""


class SubagentsConfig(BaseModel):
    """Subagent configuration.

    Subagents are loaded from ~/.config/youware-labs/paintress-cli/subagents/
    which is initialized by `paintress setup`.
    """

    disabled: list[str] = Field(default_factory=list)
    """Subagents to disable (by name)."""

    overrides: dict[str, SubagentOverride] = Field(default_factory=dict)
    """Override settings for specific subagents."""


class CommandDefinition(BaseModel):
    """Definition for a custom slash command.

    Custom commands trigger a predefined prompt when invoked via /name.
    """

    prompt: str
    """The prompt text to send to the agent."""

    mode: Literal["act", "plan"] | None = None
    """Optional mode to switch to before executing (act or plan)."""

    description: str = ""
    """Description shown in /help output."""


# Default commands provided out of the box (minimal set)
# Additional commands like /commit, /review can be added in config.toml
DEFAULT_COMMANDS: dict[str, CommandDefinition] = {
    "init": CommandDefinition(
        prompt="Please initialize the project's AGENTS.md file.",
        mode="act",
        description="Initialize AGENTS.md",
    ),
}


class MCPServerConfig(BaseModel):
    """MCP server configuration."""

    transport: Literal["stdio", "streamable_http"] = "stdio"
    """Transport type: stdio or streamable_http."""

    command: str | None = None
    """Command for stdio transport."""

    args: list[str] = Field(default_factory=list)
    """Command arguments for stdio transport."""

    env: dict[str, str] = Field(default_factory=dict)
    """Environment variables for the server."""

    url: str | None = None
    """URL for streamable_http transport."""

    headers: dict[str, str] = Field(default_factory=dict)
    """Headers for streamable_http transport."""


class MCPConfig(BaseModel):
    """MCP configuration from mcp.json."""

    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    """MCP server configurations."""


class PaintressConfig(BaseModel):
    """Complete paintress-cli configuration."""

    # From global config
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    subagents: SubagentsConfig = Field(default_factory=SubagentsConfig)
    env: dict[str, str] = Field(default_factory=dict)
    """Environment variable overrides (e.g., API keys)."""
    # From project config
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    # Custom slash commands
    commands: dict[str, CommandDefinition] = Field(default_factory=dict)
    """Custom slash commands (merged with defaults)."""

    def get_commands(self) -> dict[str, CommandDefinition]:
        """Get all commands (defaults + user-defined, user overrides defaults)."""
        result = dict(DEFAULT_COMMANDS)
        result.update(self.commands)
        return result

    @property
    def is_configured(self) -> bool:
        """Check if minimum required configuration is present."""
        return self.general.is_configured


# =============================================================================
# Environment Settings (TUI only, using pydantic-settings)
# =============================================================================


class EnvSettings(BaseSettings):
    """TUI settings from environment variables.

    Only TUI-related settings, not model configuration.
    """

    model_config = SettingsConfigDict(
        env_prefix="PAINTRESS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Display
    code_theme: Literal["dark", "light"] | None = None
    show_token_usage: bool | None = None
    show_elapsed_time: bool | None = None

    # Browser
    cdp_url: str | None = None
    browser_image: str | None = None
    browser_timeout: int | None = None

    # Steering
    steering_enabled: bool | None = None
    steering_prefix: str | None = None

    # Session
    session_dir: str | None = None
    auto_save_history: bool | None = None
    auto_restore: bool | None = None


# =============================================================================
# ConfigManager
# =============================================================================


class ConfigManager:
    """Manages configuration loading from global, project, and environment sources."""

    DEFAULT_CONFIG_DIR = Path.home() / ".config" / "youware-labs" / "paintress-cli"
    PROJECT_CONFIG_DIR = ".paintress"

    def __init__(
        self,
        config_dir: Path | None = None,
        project_dir: Path | None = None,
    ) -> None:
        self._config_dir = config_dir or self.DEFAULT_CONFIG_DIR
        self._project_dir = project_dir or Path.cwd()
        self._config: PaintressConfig | None = None
        self._loaded_sources: list[str] = []

    @property
    def config(self) -> PaintressConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            self._config = self.load()
        return self._config

    @property
    def config_dir(self) -> Path:
        """Get global config directory."""
        return self._config_dir

    @property
    def project_dir(self) -> Path:
        """Get project directory."""
        return self._project_dir

    @property
    def loaded_sources(self) -> list[str]:
        """Get list of loaded configuration sources."""
        return self._loaded_sources.copy()

    def load(self) -> PaintressConfig:
        """Load configuration from all sources.

        Priority (higher wins, no merging between levels):
        1. config.toml: Project > Global
        2. tools.toml: Project > Global
        3. Environment overrides (TUI settings only, merged on top of config.toml)
        """
        self._loaded_sources = []
        merged: dict[str, Any] = {}

        # Layer 1: config.toml (project takes priority over global, no merge)
        project_config_file = self._project_dir / self.PROJECT_CONFIG_DIR / "config.toml"
        global_config_file = self._config_dir / "config.toml"

        if project_config_file.exists():
            with open(project_config_file, "rb") as f:
                config = tomllib.load(f)
            for key in config:
                if key != "tools":
                    merged[key] = config[key]
            self._loaded_sources.append(str(project_config_file))
        elif global_config_file.exists():
            with open(global_config_file, "rb") as f:
                config = tomllib.load(f)
            for key in config:
                if key != "tools":
                    merged[key] = config[key]
            self._loaded_sources.append(str(global_config_file))

        # Layer 2: Environment overrides (TUI only, merged)
        env_overrides = self._load_env_overrides()
        if env_overrides:
            merged = _deep_merge(merged, env_overrides)
            self._loaded_sources.append("environment")

        # Layer 3: tools.toml (project takes priority over global, no merge)
        project_tools_file = self._project_dir / self.PROJECT_CONFIG_DIR / "tools.toml"
        global_tools_file = self._config_dir / "tools.toml"

        if project_tools_file.exists():
            with open(project_tools_file, "rb") as f:
                tools_config = tomllib.load(f)
            if "tools" in tools_config:
                merged["tools"] = tools_config["tools"]
            self._loaded_sources.append(str(project_tools_file))
        elif global_tools_file.exists():
            with open(global_tools_file, "rb") as f:
                tools_config = tomllib.load(f)
            if "tools" in tools_config:
                merged["tools"] = tools_config["tools"]
            self._loaded_sources.append(str(global_tools_file))

        self._config = PaintressConfig.model_validate(merged)
        return self._config

    def reload(self) -> PaintressConfig:
        """Force reload configuration."""
        self._config = None
        return self.load()

    def _load_env_overrides(self) -> dict[str, Any]:
        """Load TUI settings from environment using pydantic-settings."""
        env = EnvSettings()
        overrides: dict[str, Any] = {}

        # Display
        display: dict[str, Any] = {}
        if env.code_theme is not None:
            display["code_theme"] = env.code_theme
        if env.show_token_usage is not None:
            display["show_token_usage"] = env.show_token_usage
        if env.show_elapsed_time is not None:
            display["show_elapsed_time"] = env.show_elapsed_time
        if display:
            overrides["display"] = display

        # Browser
        browser: dict[str, Any] = {}
        if env.cdp_url is not None:
            browser["cdp_url"] = env.cdp_url
        if env.browser_image is not None:
            browser["browser_image"] = env.browser_image
        if env.browser_timeout is not None:
            browser["browser_timeout"] = env.browser_timeout
        if browser:
            overrides["browser"] = browser

        # Steering
        steering: dict[str, Any] = {}
        if env.steering_enabled is not None:
            steering["enabled"] = env.steering_enabled
        if env.steering_prefix is not None:
            steering["prefix"] = env.steering_prefix
        if steering:
            overrides["steering"] = steering

        # Session
        session: dict[str, Any] = {}
        if env.session_dir is not None:
            session["session_dir"] = env.session_dir
        if env.auto_save_history is not None:
            session["auto_save_history"] = env.auto_save_history
        if env.auto_restore is not None:
            session["auto_restore"] = env.auto_restore
        if session:
            overrides["session"] = session

        return overrides

    def load_mcp_config(self) -> MCPConfig | None:
        """Load MCP configuration from mcp.json.

        Project config takes priority over global config (no merging).

        Returns:
            MCPConfig if found, None otherwise.
        """
        # Check project first
        project_mcp = self._project_dir / self.PROJECT_CONFIG_DIR / "mcp.json"
        if project_mcp.exists():
            import json

            with open(project_mcp) as f:
                data = json.load(f)
            return MCPConfig.model_validate(data)

        # Fall back to global
        global_mcp = self._config_dir / "mcp.json"
        if global_mcp.exists():
            import json

            with open(global_mcp) as f:
                data = json.load(f)
            return MCPConfig.model_validate(data)

        return None

    def get_mcp_config_file(self) -> Path | None:
        """Get path to active MCP config file (project or global)."""
        project_mcp = self._project_dir / self.PROJECT_CONFIG_DIR / "mcp.json"
        if project_mcp.exists():
            return project_mcp
        global_mcp = self._config_dir / "mcp.json"
        if global_mcp.exists():
            return global_mcp
        return None

    def ensure_config_dir(self) -> None:
        """Create global config directory structure."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        (self._config_dir / "subagents").mkdir(exist_ok=True)

    def ensure_project_config_dir(self) -> None:
        """Create project config directory."""
        project_config_dir = self._project_dir / self.PROJECT_CONFIG_DIR
        project_config_dir.mkdir(parents=True, exist_ok=True)

    def save_default_config(self, force: bool = False) -> Path | None:
        """Save default global configuration."""
        config_file = self._config_dir / "config.toml"
        if config_file.exists() and not force:
            return None

        self.ensure_config_dir()
        config_file.write_text(_load_template("config.toml"))
        return config_file

    def save_project_config(self, force: bool = False) -> Path | None:
        """Save default project configuration."""
        self.ensure_project_config_dir()
        config_file = self._project_dir / self.PROJECT_CONFIG_DIR / "tools.toml"
        if config_file.exists() and not force:
            return None

        config_file.write_text(_load_template("tools.toml"))
        return config_file

    def get_global_config_file(self) -> Path:
        """Get path to global config file."""
        return self._config_dir / "config.toml"

    def get_project_config_file(self) -> Path:
        """Get path to project tools config file."""
        return self._project_dir / self.PROJECT_CONFIG_DIR / "tools.toml"


# =============================================================================
# Internal Utilities
# =============================================================================


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_template(name: str) -> str:
    """Load a template file."""
    template_files = resources.files("paintress_cli").joinpath("templates")
    return template_files.joinpath(name).read_text(encoding="utf-8")


# =============================================================================
# Convenience Functions
# =============================================================================


def load_config(
    config_dir: Path | None = None,
    project_dir: Path | None = None,
) -> PaintressConfig:
    """Load configuration from all sources.

    Convenience function that creates a ConfigManager and loads config.

    Args:
        config_dir: Optional custom global config directory.
        project_dir: Optional custom project directory.

    Returns:
        Loaded PaintressConfig.
    """
    manager = ConfigManager(config_dir=config_dir, project_dir=project_dir)
    return manager.load()


def get_config_manager(
    config_dir: Path | None = None,
    project_dir: Path | None = None,
) -> ConfigManager:
    """Get a ConfigManager instance.

    Args:
        config_dir: Optional custom global config directory.
        project_dir: Optional custom project directory.

    Returns:
        ConfigManager instance.
    """
    return ConfigManager(config_dir=config_dir, project_dir=project_dir)
