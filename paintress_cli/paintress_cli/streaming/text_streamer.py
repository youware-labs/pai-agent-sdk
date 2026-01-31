"""Text streaming utilities.

Provides TextStreamer for managing incremental text accumulation and rendering.
"""

from __future__ import annotations

from collections.abc import Callable

from paintress_cli.rendering import RichRenderer


class TextStreamer:
    """Manages streaming text accumulation and rendering.

    Handles incremental text updates during agent response streaming,
    tracking line position for in-place updates in the output buffer.
    """

    def __init__(
        self,
        renderer: RichRenderer,
        code_theme: str = "monokai",
        get_width: Callable[[], int] | None = None,
    ) -> None:
        """Initialize TextStreamer.

        Args:
            renderer: RichRenderer for markdown rendering.
            code_theme: Code highlighting theme.
            get_width: Callback to get current terminal width.
        """
        self._renderer = renderer
        self._code_theme = code_theme
        self._get_width = get_width or (lambda: 120)

        self._text = ""
        self._line_index: int | None = None
        self._is_active = False

    @property
    def text(self) -> str:
        """Get accumulated text."""
        return self._text

    @property
    def line_index(self) -> int | None:
        """Get output line index for this streaming block."""
        return self._line_index

    @property
    def is_active(self) -> bool:
        """Check if streaming is active."""
        return self._is_active

    def start(self, initial_content: str = "", line_index: int | None = None) -> str:
        """Start a new streaming text block.

        Args:
            initial_content: Initial text content.
            line_index: Line index in output buffer.

        Returns:
            Initial rendered content.
        """
        self._text = initial_content
        self._line_index = line_index
        self._is_active = True
        return initial_content

    def update(self, delta: str) -> str:
        """Update streaming text with delta.

        Args:
            delta: New text to append.

        Returns:
            Complete re-rendered markdown.
        """
        self._text += delta
        return self._render()

    def finalize(self) -> str:
        """Finalize streaming and return final render.

        Returns:
            Final rendered content.
        """
        result = self._render() if self._text else ""
        self._text = ""
        self._line_index = None
        self._is_active = False
        return result

    def _render(self) -> str:
        """Render current text as markdown."""
        if not self._text:
            return ""
        return self._renderer.render_markdown(
            self._text,
            code_theme=self._code_theme,
            width=self._get_width(),
        ).rstrip("\n")


class ThinkingStreamer:
    """Manages streaming thinking content (extended thinking from model).

    Similar to TextStreamer but renders with blockquote style.
    """

    def __init__(
        self,
        renderer: RichRenderer,
        get_width: Callable[[], int] | None = None,
    ) -> None:
        """Initialize ThinkingStreamer.

        Args:
            renderer: RichRenderer for rendering.
            get_width: Callback to get current terminal width.
        """
        self._renderer = renderer
        self._get_width = get_width or (lambda: 120)

        self._thinking = ""
        self._line_index: int | None = None
        self._is_active = False

    @property
    def thinking(self) -> str:
        """Get accumulated thinking content."""
        return self._thinking

    @property
    def line_index(self) -> int | None:
        """Get output line index for this streaming block."""
        return self._line_index

    @property
    def is_active(self) -> bool:
        """Check if streaming is active."""
        return self._is_active

    def start(self, initial_content: str = "", line_index: int | None = None) -> str:
        """Start a new streaming thinking block.

        Args:
            initial_content: Initial thinking content.
            line_index: Line index in output buffer.

        Returns:
            Initial rendered content.
        """
        self._thinking = initial_content
        self._line_index = line_index
        self._is_active = True
        return self._render()

    def update(self, delta: str) -> str:
        """Update streaming thinking with delta.

        Args:
            delta: New thinking text to append.

        Returns:
            Complete re-rendered thinking block.
        """
        self._thinking += delta
        return self._render()

    def finalize(self) -> str:
        """Finalize streaming and return final render.

        Returns:
            Final rendered content.
        """
        result = self._render() if self._thinking else ""
        self._thinking = ""
        self._line_index = None
        self._is_active = False
        return result

    def _render(self) -> str:
        """Render thinking content as styled blockquote."""
        if not self._thinking:
            return ""

        from rich.text import Text

        lines = self._thinking.split("\n")
        text = Text()
        for i, line in enumerate(lines):
            if i > 0:
                text.append("\n")
            text.append("> ", style="dim magenta")
            text.append(line, style="dim italic")

        return self._renderer.render(text, width=self._get_width()).rstrip("\n")
