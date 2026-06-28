"""On-device memory (VRAM / LPDDR) bandwidth.

Measures effective bandwidth of GPU-local memory — GDDR on discrete GPUs,
unified LPDDR on Jetson — **not** HBM-specific (despite older internal names).

Uses a large device-to-device copy (read+write) and a write-only fill probe.
"""

from __future__ import annotations

import torch

from syspeek.benchmarks.base import Benchmark
from syspeek.core.result import BenchmarkResult, RunContext
from syspeek.core.timing import CudaTimer

_GB = 1e9


class DeviceMemBandwidthBenchmark(Benchmark):
    name = "device_mem_bw"
    category = "memory"
    description = (
        "GPU local memory bandwidth (GDDR/LPDDR): device copy (read+write) and write-only."
    )

    def run(self, ctx: RunContext) -> list[BenchmarkResult]:
        device = ctx.torch_device
        nbytes = int(
            ctx.sizes.get(
                "device_mem_bytes",
                ctx.sizes.get("hbm_bytes", 1024 * 1024 * 1024),  # legacy key
            )
        )
        nbytes = self._fit(nbytes, device)
        timer = CudaTimer(ctx.warmup, ctx.rep, flush_l2=False, device=device)

        results: list[BenchmarkResult] = []
        try:
            src = torch.empty(nbytes, dtype=torch.int8, device=device)
            dst = torch.empty(nbytes, dtype=torch.int8, device=device)
        except torch.cuda.OutOfMemoryError as e:
            return [
                BenchmarkResult(
                    name="device_mem_copy",
                    category=self.category,
                    value=0.0,
                    unit="GB/s",
                    config={"bytes": nbytes},
                    error=f"OOM: {e}",
                )
            ]

        copy_stats = timer.time(lambda: dst.copy_(src))
        copy_gbps = (2.0 * nbytes) / (copy_stats.median_ms * 1e-3) / _GB
        peak = self._theoretical(ctx)
        results.append(
            BenchmarkResult(
                name="device_mem_copy",
                category=self.category,
                value=copy_gbps,
                unit="GB/s",
                config={"bytes": nbytes, "access": "read+write"},
                timing=copy_stats,
                theoretical=peak.value,
                theoretical_source=peak.source,
                theoretical_detail=peak.detail,
            )
        )

        write_stats = timer.time(lambda: dst.fill_(0))
        write_gbps = nbytes / (write_stats.median_ms * 1e-3) / _GB
        results.append(
            BenchmarkResult(
                name="device_mem_write",
                category=self.category,
                value=write_gbps,
                unit="GB/s",
                config={"bytes": nbytes, "access": "write"},
                timing=write_stats,
                theoretical=peak.value,
                theoretical_source=peak.source,
                theoretical_detail=peak.detail,
            )
        )

        del src, dst
        torch.cuda.empty_cache()
        return results

    @staticmethod
    def _fit(nbytes: int, device: str) -> int:
        """Cap buffer size to a safe fraction of free memory (need 2 buffers)."""
        try:
            free, _ = torch.cuda.mem_get_info(device)
        except Exception:
            return nbytes
        budget = int(free * 0.35)  # two buffers + headroom
        return max(64 * 1024 * 1024, min(nbytes, budget))

    @staticmethod
    def _theoretical(ctx: RunContext):
        from syspeek.theoretical import SpecSourceMode, resolve_theoretical

        mode = ctx.spec_source or SpecSourceMode.AUTO_FALLBACK
        return resolve_theoretical(ctx.device, "mem_bandwidth_gbps", mode)
