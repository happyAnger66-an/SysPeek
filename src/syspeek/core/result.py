"""Data models for benchmark configuration and results."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from syspeek.core.device import DeviceInfo
    from syspeek.theoretical import SpecSourceMode


@dataclass
class TimingStats:
    """Aggregated timing statistics over repeated measurements (milliseconds)."""

    samples_ms: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples_ms)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms) if self.samples_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.samples_ms) if self.samples_ms else 0.0

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.samples_ms) if self.samples_ms else 0.0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms) if self.samples_ms else 0.0

    @property
    def std_ms(self) -> float:
        return statistics.pstdev(self.samples_ms) if len(self.samples_ms) > 1 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "std_ms": self.std_ms,
        }


@dataclass
class BenchmarkResult:
    """A single benchmark measurement.

    ``value`` is the headline number reported in the table (best/representative),
    expressed in ``unit`` (e.g. "TFLOPS", "GB/s", "us").
    """

    name: str
    category: str  # "compute" | "memory" | "latency"
    value: float
    unit: str
    config: dict[str, Any] = field(default_factory=dict)
    timing: Optional[TimingStats] = None
    theoretical: Optional[float] = None  # same unit as ``value``
    theoretical_source: Optional[str] = None  # "auto" | "fixed"
    theoretical_detail: Optional[str] = None  # derivation / table key
    extra: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def efficiency(self) -> Optional[float]:
        """Achieved / theoretical, as a fraction (None if no theoretical peak)."""
        if self.theoretical and self.theoretical > 0:
            return self.value / self.theoretical
        return None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("timing", None)
        d["timing"] = self.timing.to_dict() if self.timing else None
        d["efficiency"] = self.efficiency
        d["theoretical_source"] = self.theoretical_source
        d["theoretical_detail"] = self.theoretical_detail
        return d


@dataclass
class RunContext:
    """Shared context passed to every benchmark's ``run`` method."""

    device: "DeviceInfo"
    torch_device: str = "cuda:0"
    warmup: int = 10
    rep: int = 50
    flush_l2: bool = True
    # Per-benchmark size overrides; benchmarks fall back to their own defaults.
    sizes: dict[str, Any] = field(default_factory=dict)
    dtypes: Optional[list[str]] = None  # restrict compute dtypes if set
    spec_source: "SpecSourceMode | None" = None  # set by CLI; default in runner
