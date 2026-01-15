"""Fixtures for paintress_cli tests."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest
from paintress_cli.config import ConfigManager


@pytest.fixture
def temp_home(tmp_path: Path) -> Path:
    """Create a temporary home directory."""
    home = tmp_path / "home"
    home.mkdir()
    return home


@pytest.fixture
def temp_config_dir(temp_home: Path) -> Path:
    """Create a temporary config directory under fake home."""
    config_dir = temp_home / ".config" / "youware-labs" / "paintress-cli"
    config_dir.mkdir(parents=True)
    return config_dir


@pytest.fixture
def temp_project_dir(tmp_path: Path) -> Path:
    """Create a temporary project directory."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    return project_dir


@pytest.fixture
def config_manager(temp_config_dir: Path, temp_project_dir: Path) -> ConfigManager:
    """Create a ConfigManager with temp directories."""
    return ConfigManager(config_dir=temp_config_dir, project_dir=temp_project_dir)


@pytest.fixture(autouse=True)
def mock_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock OPENAI_API_KEY for all tests that might need it."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-testing")


@pytest.fixture
def clean_env() -> Generator[None, None, None]:
    """Clean PAINTRESS_* environment variables before and after test."""
    saved_vars: dict[str, str] = {}
    for key in list(os.environ.keys()):
        if key.startswith("PAINTRESS_"):
            saved_vars[key] = os.environ.pop(key)

    yield

    for key in list(os.environ.keys()):
        if key.startswith("PAINTRESS_"):
            del os.environ[key]
    for key, value in saved_vars.items():
        os.environ[key] = value
