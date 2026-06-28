"""Kernel launch latency.

Estimates the per-kernel launch + dispatch overhead by timing a very small
kernel (a tiny element-wise op) so that runtime is dominated by launch cost
rather than actual work.
"""

from __future__ import annotations

import torch

from syspeek.benchmarks.base import Benchmark
from syspeek.core.result import BenchmarkResult, RunContext
from syspeek.core.timing import CudaTimer


class KernelLaunchLatencyBenchmark(Benchmark):
    name = "kernel_launch"
    category = "latency"
    description = "Per-kernel launch/dispatch overhead via a tiny element-wise op."

    def run(self, ctx: RunContext) -> list[BenchmarkResult]:
        device = ctx.torch_device
        x = torch.zeros(1, device=device)
        timer = CudaTimer(ctx.warmup, max(ctx.rep, 200), flush_l2=False, device=device)
        stats = timer.time(lambda: x.add_(1.0))
        us = stats.median_ms * 1e3
        return [
            BenchmarkResult(
                name="kernel_launch",
                category=self.category,
                value=us,
                unit="us",
                config={"op": "tensor.add_"},
                timing=stats,
            )
        ]
