"""Core infrastructure for SysPeek: timing, device info, results, registry."""

from __future__ import annotations

from syspeek.core.device import DeviceInfo, detect_device
from syspeek.core.registry import BENCHMARKS, get_benchmarks, register
from syspeek.core.result import BenchmarkResult, RunContext, TimingStats
from syspeek.core.timing import CudaTimer

__all__ = [
    "DeviceInfo",
    "detect_device",
    "BENCHMARKS",
    "get_benchmarks",
    "register",
    "BenchmarkResult",
    "RunContext",
    "TimingStats",
    "CudaTimer",
]
