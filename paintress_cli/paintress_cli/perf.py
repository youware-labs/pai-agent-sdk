"""Performance profiling utilities for TUI hot paths.

Enable with environment variable: PAINTRESS_PERF=1
Logs timing statistics to paintress_cli.log at INFO level.

Usage:
    from paintress_cli.perf import perf_timer, perf_report

    with perf_timer("render_markdown"):
        # ... expensive operation ...

    # Get report string
    print(perf_report())
"""

from __future__ import annotations

__all__ = ["PERF_ENABLED", "perf_log_report", "perf_report", "perf_reset", "perf_timer"]

import os
import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager

from paintress_cli.logging import get_logger

logger = get_logger(__name__)

# Global enable flag (checked once at import)
PERF_ENABLED = os.environ.get("PAINTRESS_PERF", "").strip() in ("1", "true", "yes")

# Statistics storage: name -> list of durations
_stats: dict[str, list[float]] = defaultdict(list)
_max_samples = 1000  # Keep last N samples per metric


@contextmanager
def perf_timer(name: str) -> Generator[None, None, None]:
    """Time a block of code and record the duration.

    No-op when PAINTRESS_PERF is not enabled.
    """
    if not PERF_ENABLED:
        yield
        return

    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start

    samples = _stats[name]
    samples.append(elapsed)
    if len(samples) > _max_samples:
        # Keep only recent samples
        _stats[name] = samples[-_max_samples:]

    # Log slow operations (>50ms)
    if elapsed > 0.05:
        logger.info("PERF SLOW: %s took %.1fms", name, elapsed * 1000)


def perf_report() -> str:
    """Generate a summary report of all recorded timings.

    Returns:
        Formatted report string with min/avg/max/p99/count per metric.
    """
    if not _stats:
        return "No performance data recorded. Set PAINTRESS_PERF=1 to enable."

    lines = ["=== Performance Report ===", ""]
    lines.append(f"{'Metric':<35} {'Count':>6} {'Min':>8} {'Avg':>8} {'P95':>8} {'Max':>8}")
    lines.append("-" * 80)

    for name in sorted(_stats.keys()):
        samples = _stats[name]
        if not samples:
            continue

        count = len(samples)
        min_ms = min(samples) * 1000
        avg_ms = sum(samples) / count * 1000
        max_ms = max(samples) * 1000

        # P95
        sorted_samples = sorted(samples)
        p95_idx = min(int(count * 0.95), count - 1)
        p95_ms = sorted_samples[p95_idx] * 1000

        lines.append(f"{name:<35} {count:>6} {min_ms:>7.1f}ms {avg_ms:>7.1f}ms {p95_ms:>7.1f}ms {max_ms:>7.1f}ms")

    lines.append("")
    return "\n".join(lines)


def perf_reset() -> None:
    """Clear all recorded statistics."""
    _stats.clear()


def perf_log_report() -> None:
    """Log the performance report at INFO level."""
    if _stats:
        logger.info("\n%s", perf_report())
