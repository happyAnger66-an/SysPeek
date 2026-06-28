"""On-device memory bandwidth (HBM/GDDR on discrete GPUs, LPDDR on Jetson).

Uses a large device-to-device copy as the canonical effective-bandwidth probe:
each element is read once and written once, so bytes moved = 2 * buffer size.
A write-only (memset) probe is reported as additional context.
"""

from __future__ import annotations

import torch

from syspeek.benchmarks.base import Benchmark
from syspeek.core.result import BenchmarkResult, RunContext
from syspeek.core.timing import CudaTimer

_GB = 1e9


class HbmBandwidthBenchmark(Benchmark):
    name = "device_mem_bw"
    category = "memory"
    description = "On-device memory bandwidth via large copy (read+write) and write-only."

    def run(self, ctx: RunContext) -> list[BenchmarkResult]:
        device = ctx.torch_device
        nbytes = int(ctx.sizes.get("hbm_bytes", 1024 * 1024 * 1024))
        nbytes = self._fit(nbytes, device)
        # Disable L2 flush: buffers far exceed L2, and zeroing the flush buffer
        # would itself consume the bandwidth we are trying to measure.
        timer = CudaTimer(ctx.warmup, ctx.rep, flush_l2=False, device=device)

        results: list[BenchmarkResult] = []
        try:
            src = torch.empty(nbytes, dtype=torch.int8, device=device)
            dst = torch.empty(nbytes, dtype=torch.int8, device=device)
        except torch.cuda.OutOfMemoryError as e:
            return [
                BenchmarkResult(
                    name="hbm_copy",
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
                name="hbm_copy",
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
                name="hbm_write",
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
