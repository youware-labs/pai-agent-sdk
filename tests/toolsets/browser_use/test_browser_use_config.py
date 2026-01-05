"""Tests for configuration management using pydantic-settings."""

from pai_agent_sdk.toolsets.browser_use._config import BrowserUseSettings
from pai_agent_sdk.toolsets.browser_use.toolset import BrowserUseToolset


def test_browser_use_settings_defaults():
    """Test that BrowserUseSettings has correct default values."""
    settings = BrowserUseSettings()
    assert settings.max_retries == 3
    assert settings.prefix is None
    assert settings.always_use_new_page is False
    assert settings.auto_cleanup_page is False


def test_browser_use_settings_from_env(monkeypatch):
    """Test that BrowserUseSettings loads from environment variables."""
    # Set environment variables
    monkeypatch.setenv("PAI_AGENT_BROWSER_USE_MAX_RETRIES", "5")
    monkeypatch.setenv("PAI_AGENT_BROWSER_USE_PREFIX", "custom_prefix")
    monkeypatch.setenv("PAI_AGENT_BROWSER_USE_ALWAYS_USE_NEW_PAGE", "true")
    monkeypatch.setenv("PAI_AGENT_BROWSER_USE_AUTO_CLEANUP_PAGE", "false")

    settings = BrowserUseSettings()
    assert settings.max_retries == 5
    assert settings.prefix == "custom_prefix"
    assert settings.always_use_new_page is True
    assert settings.auto_cleanup_page is False


def test_browser_use_toolset_uses_settings_defaults():
    """Test that BrowserUseToolset uses fallback defaults when settings and parameters are None."""
    toolset = BrowserUseToolset(cdp_url="http://localhost:9222/json/version")
    assert toolset.max_retries == 3  # Default fallback
    assert toolset.prefix == "browser_use"  # Default to toolset.id
    assert toolset.always_use_new_page is False  # Default fallback
    assert toolset.auto_cleanup_page is False  # Default is False


def test_browser_use_toolset_uses_env_settings(monkeypatch):
    """Test that BrowserUseToolset uses environment variables when parameters are None."""
    # Set environment variables
    monkeypatch.setenv("PAI_AGENT_BROWSER_USE_MAX_RETRIES", "10")
    monkeypatch.setenv("PAI_AGENT_BROWSER_USE_PREFIX", "env_browser")
    monkeypatch.setenv("PAI_AGENT_BROWSER_USE_ALWAYS_USE_NEW_PAGE", "true")

    toolset = BrowserUseToolset(cdp_url="http://localhost:9222/json/version")
    assert toolset.max_retries == 10
    assert toolset.prefix == "env_browser"
    assert toolset.always_use_new_page is True
    assert toolset.auto_cleanup_page is False  # Default is False regardless of always_use_new_page


def test_browser_use_toolset_parameters_override_env(monkeypatch):
    """Test that explicit parameters override environment variables."""
    # Set environment variables
    monkeypatch.setenv("PAI_AGENT_BROWSER_USE_MAX_RETRIES", "10")
    monkeypatch.setenv("PAI_AGENT_BROWSER_USE_PREFIX", "env_browser")
    monkeypatch.setenv("PAI_AGENT_BROWSER_USE_ALWAYS_USE_NEW_PAGE", "true")
    monkeypatch.setenv("PAI_AGENT_BROWSER_USE_AUTO_CLEANUP_PAGE", "true")

    # Create toolset with explicit parameters
    toolset = BrowserUseToolset(
        cdp_url="http://localhost:9222/json/version",
        max_retries=20,
        prefix="param_browser",
        always_use_new_page=False,
        auto_cleanup_page=False,
    )

    # Parameters should override environment variables
    assert toolset.max_retries == 20
    assert toolset.prefix == "param_browser"
    assert toolset.always_use_new_page is False
    assert toolset.auto_cleanup_page is False


def test_browser_use_toolset_partial_parameter_override(monkeypatch):
    """Test that only provided parameters override environment variables."""
    # Set environment variables
    monkeypatch.setenv("PAI_AGENT_BROWSER_USE_MAX_RETRIES", "10")
    monkeypatch.setenv("PAI_AGENT_BROWSER_USE_PREFIX", "env_browser")
    monkeypatch.setenv("PAI_AGENT_BROWSER_USE_ALWAYS_USE_NEW_PAGE", "true")
    monkeypatch.setenv("PAI_AGENT_BROWSER_USE_AUTO_CLEANUP_PAGE", "false")

    # Create toolset with only max_retries parameter
    toolset = BrowserUseToolset(
        cdp_url="http://localhost:9222/json/version",
        max_retries=20,  # Override env
        # prefix, always_use_new_page and auto_cleanup_page will use env values
    )

    assert toolset.max_retries == 20  # Overridden by parameter
    assert toolset.prefix == "env_browser"  # From environment
    assert toolset.always_use_new_page is True  # From environment
    assert toolset.auto_cleanup_page is False  # From environment


def test_browser_use_settings_env_file_loading(tmp_path, monkeypatch):
    """Test that BrowserUseSettings can load from .env file."""
    # Create a temporary .env file
    env_file = tmp_path / ".env"
    env_file.write_text(
        "PAI_AGENT_BROWSER_USE_MAX_RETRIES=15\n"
        "PAI_AGENT_BROWSER_USE_PREFIX=file_browser\n"
        "PAI_AGENT_BROWSER_USE_ALWAYS_USE_NEW_PAGE=false\n"
        "PAI_AGENT_BROWSER_USE_AUTO_CLEANUP_PAGE=true\n"
    )

    # Change to temp directory to pick up .env file
    monkeypatch.chdir(tmp_path)

    settings = BrowserUseSettings()
    assert settings.max_retries == 15
    assert settings.prefix == "file_browser"
    assert settings.always_use_new_page is False
    assert settings.auto_cleanup_page is True


def test_auto_cleanup_page_defaults_to_false():
    """Test that auto_cleanup_page defaults to False regardless of always_use_new_page."""
    # When always_use_new_page is True, auto_cleanup_page should still default to False
    toolset = BrowserUseToolset(
        cdp_url="http://localhost:9222/json/version",
        always_use_new_page=True,
    )
    assert toolset.auto_cleanup_page is False

    # When always_use_new_page is False, auto_cleanup_page should also default to False
    toolset = BrowserUseToolset(
        cdp_url="http://localhost:9222/json/version",
        always_use_new_page=False,
    )
    assert toolset.auto_cleanup_page is False

    # Default behavior without any parameters
    toolset = BrowserUseToolset(
        cdp_url="http://localhost:9222/json/version",
    )
    assert toolset.auto_cleanup_page is False


def test_auto_cleanup_page_can_override_default():
    """Test that auto_cleanup_page parameter can override the default behavior."""
    # Explicitly disable auto_cleanup even when always_use_new_page is True
    toolset = BrowserUseToolset(
        cdp_url="http://localhost:9222/json/version",
        always_use_new_page=True,
        auto_cleanup_page=False,
    )
    assert toolset.always_use_new_page is True
    assert toolset.auto_cleanup_page is False

    # Explicitly enable auto_cleanup even when always_use_new_page is False
    toolset = BrowserUseToolset(
        cdp_url="http://localhost:9222/json/version",
        always_use_new_page=False,
        auto_cleanup_page=True,
    )
    assert toolset.always_use_new_page is False
    assert toolset.auto_cleanup_page is True
