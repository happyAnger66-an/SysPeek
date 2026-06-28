"""Benchmark registry.

Benchmarks register themselves (via :func:`register`) so the CLI can discover,
filter, and run them without hard-coding the list. Importing
``syspeek.benchmarks`` populates the registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from syspeek.benchmarks.base import Benchmark

BENCHMARKS: dict[str, "Benchmark"] = {}


def register(benchmark: "Benchmark") -> "Benchmark":
    """Register a benchmark instance by its ``name``."""
    if benchmark.name in BENCHMARKS:
        raise ValueError(f"Duplicate benchmark name: {benchmark.name}")
    BENCHMARKS[benchmark.name] = benchmark
    return benchmark


def get_benchmarks(
    names: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
) -> list["Benchmark"]:
    """Return registered benchmarks filtered by name and/or category."""
    # Ensure built-in benchmarks are imported and registered.
    import syspeek.benchmarks  # noqa: F401

    selected = list(BENCHMARKS.values())
    if names:
        wanted = set(names)
        selected = [b for b in selected if b.name in wanted]
    if categories:
        cats = set(categories)
        selected = [b for b in selected if b.category in cats]
    return selected
