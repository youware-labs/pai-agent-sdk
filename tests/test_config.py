"""Tests for pai_agent_sdk._config module."""

from pathlib import Path

from pai_agent_sdk._config import AgentSettings
from pai_agent_sdk.environment.local import LocalEnvironment


def test_agent_context_settings_defaults() -> None:
    """Should have correct default values."""
    settings = AgentSettings()
    assert settings.working_dir is None
    assert settings.tmp_base_dir is None
    assert settings.image_understanding_model is None
    assert settings.video_understanding_model is None


def test_agent_context_settings_from_env(monkeypatch, tmp_path: Path) -> None:
    """Should load from environment variables."""
    working = tmp_path / "work"
    working.mkdir()
    tmp_base = tmp_path / "tmp"
    tmp_base.mkdir()

    monkeypatch.setenv("PAI_AGENT_WORKING_DIR", str(working))
    monkeypatch.setenv("PAI_AGENT_TMP_BASE_DIR", str(tmp_base))
    monkeypatch.setenv("PAI_AGENT_IMAGE_UNDERSTANDING_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("PAI_AGENT_VIDEO_UNDERSTANDING_MODEL", "google-gla:gemini-2.0-flash")

    settings = AgentSettings()
    assert settings.working_dir == working
    assert settings.tmp_base_dir == tmp_base
    assert settings.image_understanding_model == "openai:gpt-4o"
    assert settings.video_understanding_model == "google-gla:gemini-2.0-flash"


async def test_local_environment_uses_tmp_base_dir(tmp_path: Path) -> None:
    """LocalEnvironment should use tmp_base_dir for creating tmp directory."""
    tmp_base = tmp_path / "tmp"
    tmp_base.mkdir()

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_base,
    ) as env:
        assert env.tmp_dir is not None
        assert env.tmp_dir.parent == tmp_base
