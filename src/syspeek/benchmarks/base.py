"""Base class for all benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod

from syspeek.core.device import DeviceInfo
from syspeek.core.result import BenchmarkResult, RunContext


class Benchmark(ABC):
    """A self-describing, registrable benchmark.

    Subclasses set ``name`` and ``category`` and implement :meth:`run`. They may
    override :meth:`applicable` to skip on unsupported hardware.
    """

    name: str = ""
    category: str = ""  # "compute" | "memory" | "latency"
    description: str = ""

    def applicable(self, device: DeviceInfo) -> bool:  # noqa: ARG002
        """Whether this benchmark can run on ``device``. Default: always."""
        return True

    @abstractmethod
    def run(self, ctx: RunContext) -> list[BenchmarkResult]:
        """Execute the benchmark and return one or more results."""
        raise NotImplementedError
