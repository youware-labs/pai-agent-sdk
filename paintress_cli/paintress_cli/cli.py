"""CLI entry point for paintress-cli.

Minimal CLI that launches the TUI application.
Most interactions happen inside the TUI via slash commands.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
from importlib import resources

import click

from paintress_cli import __version__
from paintress_cli.config import ConfigManager, PaintressConfig
from paintress_cli.logging import configure_logging, get_logger

logger = get_logger(__name__)


# =============================================================================
# Provider Environment Variable Mapping
# =============================================================================

PROVIDER_ENV_VARS = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"),
    "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    "openai-chat": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    "openai-responses": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    "google-gla": ("GOOGLE_API_KEY", None),
    "google-vertex": ("GOOGLE_API_KEY", None),
    "gemini": ("GOOGLE_API_KEY", None),
    "groq": ("GROQ_API_KEY", "GROQ_BASE_URL"),
    "bedrock": (None, None),  # Uses AWS credentials
}

# Provider to model_settings preset mapping
PROVIDER_MODEL_SETTINGS = {
    "anthropic": "anthropic_default",
    "openai": "openai_default",
    "openai-chat": "openai_default",
    "openai-responses": "openai_responses_default",
    "google-gla": "gemini_thinking_budget_default",
    "google-vertex": "gemini_thinking_budget_default",
    "gemini": "gemini_thinking_budget_default",
    "groq": None,  # No preset
    "bedrock": None,
}

# Provider to model_cfg preset mapping (context window, capabilities)
# - gemini: vision + video_understanding
# - anthropic/openai: vision only
# - unknown: no capabilities
PROVIDER_MODEL_CFG = {
    "anthropic": "claude_200k",
    "openai": "gpt5_270k",
    "openai-chat": "gpt5_270k",
    "openai-responses": "gpt5_270k",
    "google-gla": "gemini_1m",
    "google-vertex": "gemini_1m",
    "gemini": "gemini_1m",
    "groq": None,  # Unknown - no capabilities
    "bedrock": None,
}


def parse_model_string(model_str: str) -> tuple[str | None, str, str]:
    """Parse model string into (gateway, provider, model_id).

    Format: [gateway@]provider:model_id

    Examples:
        "anthropic:claude-sonnet-4" -> (None, "anthropic", "claude-sonnet-4")
        "mygateway@openai:gpt-4o" -> ("mygateway", "openai", "gpt-4o")
    """
    gateway = None
    if "@" in model_str:
        gateway, model_str = model_str.split("@", 1)

    if ":" not in model_str:
        raise ValueError(f"Invalid model format: {model_str}. Expected 'provider:model_id'")

    provider, model_id = model_str.split(":", 1)
    return gateway, provider, model_id


def get_env_vars_for_model(gateway: str | None, provider: str) -> list[tuple[str, str, bool]]:
    """Get required environment variables for a model.

    Returns:
        List of (env_var_name, description, required) tuples.
    """
    env_vars = []

    if gateway:
        # Gateway mode: {GATEWAY}_API_KEY and {GATEWAY}_BASE_URL
        prefix = gateway.upper()
        env_vars.append((f"{prefix}_API_KEY", "Gateway API Key", True))
        env_vars.append((f"{prefix}_BASE_URL", "Gateway Base URL", True))
    else:
        # Direct provider mode
        api_key, base_url = PROVIDER_ENV_VARS.get(provider, (f"{provider.upper()}_API_KEY", None))
        if api_key:
            env_vars.append((api_key, "API Key", True))
        if base_url:
            env_vars.append((base_url, "Base URL (optional, press Enter to skip)", False))

    return env_vars


# =============================================================================
# Setup Wizard
# =============================================================================


def run_setup_wizard(config_manager: ConfigManager) -> bool:
    """Run interactive setup wizard for first-time configuration.

    Flow:
    1. Copy template config to global config dir
    2. Prompt for model string
    3. Prompt for required environment variables
    4. Update config with values
    5. Show completion message with paths

    Returns:
        True if setup completed successfully, False if user cancelled.
    """
    click.echo()
    click.echo(click.style("Welcome to Paintress CLI!", fg="cyan", bold=True))
    click.echo("Let's set up your configuration.\n")

    # Step 1: Copy templates to config dir
    config_manager.ensure_config_dir()
    config_path = config_manager.config_dir / "config.toml"
    mcp_path = config_manager.config_dir / "mcp.json"
    subagents_dir = config_manager.config_dir / "subagents"

    # Copy config.toml template
    if not config_path.exists():
        template_path = resources.files("paintress_cli.templates").joinpath("config.toml")
        with resources.as_file(template_path) as src:
            shutil.copy(src, config_path)
        click.echo(f"Created: {config_path}")

    # Copy mcp.json template (only if not exists - never overwrite user's mcp.json)
    if not mcp_path.exists():
        mcp_template = resources.files("paintress_cli.templates").joinpath("mcp.json")
        with resources.as_file(mcp_template) as src:
            shutil.copy(src, mcp_path)
        click.echo(f"Created: {mcp_path}")
    else:
        click.echo(f"Skipped: {mcp_path} (already exists)")

    # Copy builtin subagents from pai_agent_sdk (only missing files - never overwrite)
    subagents_dir.mkdir(parents=True, exist_ok=True)
    sdk_presets = resources.files("pai_agent_sdk.subagents.presets")
    copied_subagents = []
    for item in sdk_presets.iterdir():
        if item.name.endswith(".md"):
            target_path = subagents_dir / item.name
            if not target_path.exists():
                with resources.as_file(item) as src:
                    shutil.copy(src, target_path)
                copied_subagents.append(item.name)
    if copied_subagents:
        click.echo(f"Created: {subagents_dir}/ (added: {', '.join(copied_subagents)})")
    else:
        click.echo(f"Skipped: {subagents_dir}/ (all subagents already exist)")

    click.echo()

    # Step 2: Prompt for model string
    click.echo(click.style("Step 1: Enter model", bold=True))
    click.echo("  Format: [gateway@]provider:model_id")
    click.echo("  Examples:")
    click.echo("    - anthropic:claude-sonnet-4-20250514")
    click.echo("    - openai:gpt-4o")
    click.echo("    - google-gla:gemini-2.5-pro")
    click.echo("    - mygateway@anthropic:claude-sonnet-4")
    click.echo()

    while True:
        model_str = click.prompt("Model", default="anthropic:claude-sonnet-4-20250514")
        try:
            gateway, provider, _model_id = parse_model_string(model_str)
            break
        except ValueError as e:
            click.echo(click.style(f"  Error: {e}", fg="red"))

    # Step 3: Prompt for environment variables
    click.echo()
    click.echo(click.style("Step 2: Configure credentials", bold=True))

    env_vars = get_env_vars_for_model(gateway, provider)
    env_values: dict[str, str] = {}

    for env_var, description, required in env_vars:
        # Check if already set in environment
        existing = os.environ.get(env_var)
        if existing:
            masked = existing[:8] + "..." if len(existing) > 12 else "***"
            click.echo(f"  {env_var}: found in environment ({masked})")
            if click.confirm("    Use existing value?", default=True):
                continue

        # Prompt for value
        is_secret = "KEY" in env_var or "SECRET" in env_var
        value = click.prompt(
            f"  {env_var} ({description})",
            default="" if not required else None,
            hide_input=is_secret,
            show_default=False,
        )

        if value:
            env_values[env_var] = value
        elif required:
            click.echo(click.style("  This field is required.", fg="red"))
            return False

    # Step 4: Update config file
    click.echo()
    click.echo(click.style("Step 3: Saving configuration...", bold=True))

    # Auto-detect model_settings preset based on provider
    model_settings_preset = PROVIDER_MODEL_SETTINGS.get(provider)
    if model_settings_preset:
        click.echo(f"  Auto-detected model_settings: {model_settings_preset}")

    # Auto-detect model_cfg preset based on provider
    model_cfg_preset = PROVIDER_MODEL_CFG.get(provider)
    if model_cfg_preset:
        click.echo(f"  Auto-detected model_cfg: {model_cfg_preset}")

    # Read current config and update
    config_content = config_path.read_text()

    # Update model
    config_content = re.sub(
        r'^model\s*=\s*".*"',
        f'model = "{model_str}"',
        config_content,
        flags=re.MULTILINE,
    )

    # Update model_settings if we have a preset
    if model_settings_preset:
        # Check for existing uncommented model_settings line
        if re.search(r"^model_settings\s*=", config_content, re.MULTILINE):
            config_content = re.sub(
                r"^model_settings\s*=\s*.*$",
                f'model_settings = "{model_settings_preset}"',
                config_content,
                flags=re.MULTILINE,
            )
        # Check for commented model_settings line and uncomment it
        elif re.search(r"^#\s*model_settings\s*=", config_content, re.MULTILINE):
            config_content = re.sub(
                r"^#\s*model_settings\s*=\s*.*$",
                f'model_settings = "{model_settings_preset}"',
                config_content,
                flags=re.MULTILINE,
            )
        else:
            # Add after model line as last resort
            config_content = re.sub(
                r'^(model\s*=\s*"[^"]*")$',
                f'\\1\nmodel_settings = "{model_settings_preset}"',
                config_content,
                flags=re.MULTILINE,
            )

    # Update model_cfg if we have a preset
    if model_cfg_preset:
        # Check for existing uncommented model_cfg line
        if re.search(r"^model_cfg\s*=", config_content, re.MULTILINE):
            config_content = re.sub(
                r"^model_cfg\s*=\s*.*$",
                f'model_cfg = "{model_cfg_preset}"',
                config_content,
                flags=re.MULTILINE,
            )
        # Check for commented model_cfg line and uncomment it
        elif re.search(r"^#\s*model_cfg\s*=", config_content, re.MULTILINE):
            config_content = re.sub(
                r"^#\s*model_cfg\s*=\s*.*$",
                f'model_cfg = "{model_cfg_preset}"',
                config_content,
                flags=re.MULTILINE,
            )
        else:
            # Add after model_settings line (or model line if no model_settings)
            if re.search(r"^model_settings\s*=", config_content, re.MULTILINE):
                config_content = re.sub(
                    r'^(model_settings\s*=\s*"[^"]*")$',
                    f'\\1\nmodel_cfg = "{model_cfg_preset}"',
                    config_content,
                    flags=re.MULTILINE,
                )
            else:
                config_content = re.sub(
                    r'^(model\s*=\s*"[^"]*")$',
                    f'\\1\nmodel_cfg = "{model_cfg_preset}"',
                    config_content,
                    flags=re.MULTILINE,
                )

    # Update [env] section with new values
    if env_values:
        # Find or create [env] section
        if "[env]" in config_content:
            # Add values after [env]
            env_lines = "\n".join(f'{k} = "{v}"' for k, v in env_values.items())
            config_content = re.sub(
                r"\[env\]\n(#[^\n]*\n)*",
                f"[env]\n{env_lines}\n",
                config_content,
            )
        else:
            # Append [env] section
            env_lines = "\n".join(f'{k} = "{v}"' for k, v in env_values.items())
            config_content += f"\n[env]\n{env_lines}\n"

    config_path.write_text(config_content)

    # Step 5: Show completion
    click.echo()
    click.echo(click.style("Setup complete!", fg="green", bold=True))
    click.echo()
    click.echo("Configuration saved to:")
    click.echo(f"  {config_path}")
    click.echo()
    click.echo("You can also configure:")
    click.echo(f"  - Custom subagents: {config_manager.config_dir / 'subagents/'}")
    click.echo(f"  - MCP servers: {config_manager.config_dir / 'mcp.json'}")
    click.echo()
    click.echo("Run 'paintress-cli' again to start!")
    click.echo()

    return True


def load_env_from_config(config: PaintressConfig) -> None:
    """Load environment variables from config [env] section."""
    if config.env:
        for key, value in config.env.items():
            if value and key not in os.environ:
                os.environ[key] = value


# =============================================================================
# CLI Entry Point
# =============================================================================


@click.command()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.version_option(version=__version__, prog_name="paintress-cli")
def cli(verbose: bool) -> None:
    """Paintress CLI - AI-powered coding assistant.

    Inside TUI, use slash commands:
      /help     - Show available commands
      /config   - Show/edit configuration
      /mode     - Switch between act/plan modes
      /dump     - Save session
      /load     - Load session
      /clear    - Clear conversation
      /exit     - Exit application
    """
    configure_logging(verbose=verbose)
    logger.info("Starting paintress-cli v%s", __version__)

    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load()

    # Check if configuration exists
    if not config.is_configured:
        if not run_setup_wizard(config_manager):
            sys.exit(0)
        # Reload config after setup
        config = config_manager.reload()

    # Load env vars from config
    load_env_from_config(config)

    # Run the TUI
    try:
        asyncio.run(_run_tui(config, config_manager, verbose))
    except KeyboardInterrupt:
        click.echo("\nGoodbye!")
        sys.exit(130)
    except Exception as e:
        logger.exception("Fatal error")
        click.echo()
        click.echo(click.style("=" * 60, fg="red"))
        click.echo(click.style("FATAL ERROR", fg="red", bold=True))
        click.echo(click.style("=" * 60, fg="red"))
        click.echo()
        click.echo(f"Error type: {type(e).__name__}")
        click.echo(f"Message: {e}")
        click.echo()
        # Show traceback in verbose mode or for unexpected errors
        if verbose:
            import traceback

            click.echo(click.style("Traceback:", fg="yellow"))
            click.echo(traceback.format_exc())
        else:
            click.echo("Run with --verbose flag for full traceback.")
        click.echo()
        click.echo("Common issues:")
        click.echo("  - API key not set or invalid")
        click.echo("  - Network connectivity issues")
        click.echo("  - Invalid model configuration")
        click.echo()
        click.echo("Check logs at: ~/.config/youware-labs/paintress-cli/paintress.log")
        sys.exit(1)


async def _run_tui(
    config: PaintressConfig,
    config_manager: ConfigManager,
    verbose: bool,
) -> None:
    """Run the TUI application."""
    from paintress_cli.app import TUIApp

    async with TUIApp(config=config, config_manager=config_manager, verbose=verbose) as app:
        await app.run()


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
