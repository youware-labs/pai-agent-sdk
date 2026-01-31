"""Rich rendering utilities.

Provides RichRenderer for converting Rich renderables to ANSI strings.
"""

from __future__ import annotations

from io import StringIO
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text


class RichRenderer:
    """Convert Rich renderables to ANSI strings for output buffer.

    This renderer creates a new Console per render call to ensure clean output.
    For high-frequency rendering, consider using CachedRichRenderer.
    """

    def __init__(self, width: int | None = None) -> None:
        self._width = width or 120

    def render(self, renderable: Any, width: int | None = None) -> str:
        """Render Rich object to ANSI string.

        Args:
            renderable: Rich renderable object
            width: Optional width override.
        """
        render_width = width or self._width
        string_io = StringIO()
        console = Console(
            file=string_io,
            force_terminal=True,
            width=render_width,
            no_color=False,
        )
        console.print(renderable)
        return string_io.getvalue()

    def render_markdown(self, text: str, code_theme: str = "monokai", width: int | None = None) -> str:
        """Render markdown text to ANSI string."""
        return self.render(Markdown(text, code_theme=code_theme), width=width)

    def render_text(self, text: str, style: str | None = None) -> str:
        """Render styled text to ANSI string."""
        return self.render(Text(text, style=style or ""))

    def render_panel(
        self,
        content: str | Any,
        title: str | None = None,
        border_style: str = "blue",
    ) -> str:
        """Render a panel to ANSI string."""
        return self.render(Panel(content, title=title, border_style=border_style))


class CachedRichRenderer(RichRenderer):
    """RichRenderer with caching for improved performance.

    Caches rendered output based on content hash and width.
    Useful for high-frequency rendering scenarios.
    """

    def __init__(self, width: int | None = None, cache_size: int = 100) -> None:
        super().__init__(width)
        self._cache: dict[tuple[int, int], str] = {}
        self._cache_size = cache_size
        self._cache_order: list[tuple[int, int]] = []

    def render(self, renderable: Any, width: int | None = None) -> str:
        """Render with caching based on content hash."""
        render_width = width or self._width

        # Create cache key from content hash and width
        try:
            content_hash = hash(str(renderable))
        except Exception:
            # Fall back to non-cached render if hashing fails
            return super().render(renderable, width)

        cache_key = (content_hash, render_width)

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Render and cache
        result = super().render(renderable, width)

        # Manage cache size
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry
            oldest_key = self._cache_order.pop(0)
            self._cache.pop(oldest_key, None)

        self._cache[cache_key] = result
        self._cache_order.append(cache_key)

        return result

    def clear_cache(self) -> None:
        """Clear the render cache."""
        self._cache.clear()
        self._cache_order.clear()
