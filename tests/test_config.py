"""Tests for pai_agent_sdk._config module."""

from pathlib import Path

from pai_agent_sdk._config import AgentContextSettings
from pai_agent_sdk.context import AgentContext


def test_agent_context_settings_defaults() -> None:
    """Should have correct default values."""
    settings = AgentContextSettings()
    assert settings.working_dir is None
    assert settings.tmp_base_dir is None


def test_agent_context_settings_from_env(monkeypatch, tmp_path: Path) -> None:
    """Should load from environment variables."""
    working = tmp_path / "work"
    working.mkdir()
    tmp_base = tmp_path / "tmp"
    tmp_base.mkdir()

    monkeypatch.setenv("PAI_AGENT_WORKING_DIR", str(working))
    monkeypatch.setenv("PAI_AGENT_TMP_BASE_DIR", str(tmp_base))

    settings = AgentContextSettings()
    assert settings.working_dir == working
    assert settings.tmp_base_dir == tmp_base


def test_agent_context_uses_settings(monkeypatch, tmp_path: Path) -> None:
    """AgentContext should use settings from environment."""
    working = tmp_path / "work"
    working.mkdir()
    tmp_base = tmp_path / "tmp"
    tmp_base.mkdir()

    monkeypatch.setenv("PAI_AGENT_WORKING_DIR", str(working))
    monkeypatch.setenv("PAI_AGENT_TMP_BASE_DIR", str(tmp_base))

    ctx = AgentContext()
    assert ctx.working_dir == working
    assert ctx.tmp_base_dir == tmp_base


def test_agent_context_parameters_override_env(monkeypatch, tmp_path: Path) -> None:
    """Explicit parameters should override environment variables."""
    env_working = tmp_path / "env_work"
    env_working.mkdir()
    param_working = tmp_path / "param_work"
    param_working.mkdir()

    monkeypatch.setenv("PAI_AGENT_WORKING_DIR", str(env_working))

    ctx = AgentContext(working_dir=param_working)
    assert ctx.working_dir == param_working
