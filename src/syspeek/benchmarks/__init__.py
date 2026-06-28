"""Built-in benchmarks. Importing this module registers them."""

from __future__ import annotations

from syspeek.benchmarks.base import Benchmark
from syspeek.benchmarks.compute_flops import GemmFlopsBenchmark
from syspeek.benchmarks.latency import KernelLaunchLatencyBenchmark
from syspeek.benchmarks.memory_hbm import HbmBandwidthBenchmark
from syspeek.benchmarks.memory_transfer import HostDeviceBandwidthBenchmark
from syspeek.core.registry import register

# Register built-ins (idempotent guard for repeated imports).
from syspeek.core.registry import BENCHMARKS as _REG

for _bench in (
    GemmFlopsBenchmark(),
    HostDeviceBandwidthBenchmark(),
    HbmBandwidthBenchmark(),
    KernelLaunchLatencyBenchmark(),
):
    if _bench.name not in _REG:
        register(_bench)

__all__ = [
    "Benchmark",
    "GemmFlopsBenchmark",
    "HostDeviceBandwidthBenchmark",
    "HbmBandwidthBenchmark",
    "KernelLaunchLatencyBenchmark",
]
