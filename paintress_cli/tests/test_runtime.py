"""Tests for paintress_cli.runtime module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from paintress_cli.browser import BrowserManager
from paintress_cli.config import (
    BrowserConfig,
    GeneralConfig,
    MCPConfig,
    MCPServerConfig,
    PaintressConfig,
    ToolsConfig,
)
from paintress_cli.runtime import create_tui_runtime

# =============================================================================
# create_tui_runtime Tests
# =============================================================================


def test_create_tui_runtime_minimal(tmp_path: Path) -> None:
    """Test creating runtime with minimal config."""
    config = PaintressConfig(
        general=GeneralConfig(model="openai:gpt-4"),
    )

    runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
    )

    assert runtime is not None
    assert runtime.env is not None
    assert runtime.ctx is not None
    assert runtime.agent is not None


def test_create_tui_runtime_with_model_settings(tmp_path: Path) -> None:
    """Test creating runtime with model settings preset."""
    # Use openai which is more commonly mocked in tests
    config = PaintressConfig(
        general=GeneralConfig(
            model="openai:gpt-4",
            model_settings="openai_high",
        ),
    )

    runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
    )

    assert runtime is not None


def test_create_tui_runtime_with_mcp_servers(tmp_path: Path) -> None:
    """Test creating runtime with MCP servers."""
    config = PaintressConfig(
        general=GeneralConfig(model="openai:gpt-4"),
    )
    mcp_config = MCPConfig(
        servers={
            "test": MCPServerConfig(
                transport="stdio",
                command="echo",
                args=["test"],
            ),
        }
    )

    runtime = create_tui_runtime(
        config=config,
        mcp_config=mcp_config,
        working_dir=tmp_path,
    )

    assert runtime is not None


def test_create_tui_runtime_with_browser_manager(tmp_path: Path) -> None:
    """Test creating runtime with browser manager."""
    config = PaintressConfig(
        general=GeneralConfig(model="openai:gpt-4"),
    )

    # Create a mock browser manager
    browser_config = BrowserConfig(cdp_url="ws://localhost:9222")
    browser_manager = BrowserManager(browser_config)
    # Manually set cdp_url to simulate started state
    browser_manager._cdp_url = "ws://localhost:9222"

    # Mock get_browser_toolset to avoid importing actual toolset
    with patch.object(browser_manager, "get_browser_toolset", return_value=None):
        runtime = create_tui_runtime(
            config=config,
            browser_manager=browser_manager,
            working_dir=tmp_path,
        )

    assert runtime is not None


def test_create_tui_runtime_with_need_approval(tmp_path: Path) -> None:
    """Test creating runtime with tools needing approval."""
    config = PaintressConfig(
        general=GeneralConfig(model="openai:gpt-4"),
        tools=ToolsConfig(need_approval=["shell_sandbox", "file_write"]),
    )

    runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
    )

    assert runtime is not None


def test_create_tui_runtime_uses_cwd_by_default() -> None:
    """Test that runtime uses cwd when working_dir not specified."""
    config = PaintressConfig(
        general=GeneralConfig(model="openai:gpt-4"),
    )

    runtime = create_tui_runtime(config=config)

    assert runtime is not None


def test_create_tui_runtime_with_model_cfg_preset(tmp_path: Path) -> None:
    """Test creating runtime with model_cfg preset."""
    from pai_agent_sdk.context import ModelCapability

    config = PaintressConfig(
        general=GeneralConfig(
            model="openai:gpt-4",
            model_cfg="claude_200k",
        ),
    )

    runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
    )

    assert runtime is not None
    # Check model_cfg was applied
    assert runtime.ctx.model_cfg.context_window == 200_000
    assert runtime.ctx.model_cfg.max_images == 20
    assert ModelCapability.vision in runtime.ctx.model_cfg.capabilities


def test_create_tui_runtime_with_model_cfg_gemini(tmp_path: Path) -> None:
    """Test creating runtime with gemini model_cfg preset (has video support)."""
    from pai_agent_sdk.context import ModelCapability

    # Use openai model to avoid API key requirement, but test gemini preset
    config = PaintressConfig(
        general=GeneralConfig(
            model="openai:gpt-4",
            model_cfg="gemini_1m",
        ),
    )

    runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
    )

    assert runtime is not None
    # Check gemini preset has vision + video capabilities
    assert runtime.ctx.model_cfg.context_window == 1_000_000
    assert ModelCapability.vision in runtime.ctx.model_cfg.capabilities
    assert ModelCapability.video_understanding in runtime.ctx.model_cfg.capabilities


def test_create_tui_runtime_with_model_cfg_dict(tmp_path: Path) -> None:
    """Test creating runtime with custom model_cfg dict."""
    from pai_agent_sdk.context import ModelCapability

    config = PaintressConfig(
        general=GeneralConfig(
            model="openai:gpt-4",
            model_cfg={
                "context_window": 100_000,
                "max_images": 10,
                "capabilities": ["vision"],
            },
        ),
    )

    runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
    )

    assert runtime is not None
    assert runtime.ctx.model_cfg.context_window == 100_000
    assert runtime.ctx.model_cfg.max_images == 10
    assert ModelCapability.vision in runtime.ctx.model_cfg.capabilities


def test_create_tui_runtime_with_no_model_cfg(tmp_path: Path) -> None:
    """Test creating runtime without model_cfg uses defaults."""
    config = PaintressConfig(
        general=GeneralConfig(model="openai:gpt-4"),
    )

    runtime = create_tui_runtime(
        config=config,
        working_dir=tmp_path,
    )

    assert runtime is not None
    # Default ModelConfig values
    assert runtime.ctx.model_cfg.context_window is None
    assert runtime.ctx.model_cfg.max_images == 20
    assert len(runtime.ctx.model_cfg.capabilities) == 0
