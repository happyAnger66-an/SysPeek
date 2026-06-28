"""Host<->Device transfer bandwidth (PCIe on discrete GPUs, shared-mem on Jetson).

Uses **multi-stream parallel copies** by default to reduce CPU submission bottlenecks
(pageable memory) and overlap DMA requests. Pinned memory on a discrete GPU is
usually already link-limited with a single buffer; multi-stream mainly helps
pageable paths.

Theoretical peaks (PCIe or shared-memory upper bound) are shown for **pinned**
transfers; pageable runs the same link ceiling for efficiency reference but CPU
staging typically prevents reaching it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import torch

from syspeek.benchmarks.base import Benchmark
from syspeek.core.result import BenchmarkResult, RunContext
from syspeek.core.timing import CudaTimer

_GB = 1e9


@dataclass
class _TransferPlan:
    """Pre-built buffers and copy functions for one pinned/pageable mode."""

    nbytes: int
    pinned: bool
    num_streams: int
    mode: str
    h2d_fn: Callable[[], None]
    d2h_fn: Callable[[], None]


def _build_plan(
    nbytes: int,
    device: str,
    pinned: bool,
    num_streams: int,
    mode: str,
) -> _TransferPlan:
    num_streams = max(1, num_streams)
    if mode == "single" or num_streams == 1:
        return _build_single(nbytes, device, pinned)

    chunk = nbytes // num_streams
    total = chunk * num_streams
    hosts = [
        torch.empty(chunk, dtype=torch.int8, pin_memory=pinned)
        for _ in range(num_streams)
    ]
    devs = [torch.empty(chunk, dtype=torch.int8, device=device) for _ in range(num_streams)]
    streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]

    def _h2d_multi_stream() -> None:
        for i in range(num_streams):
            with torch.cuda.stream(streams[i]):
                devs[i].copy_(hosts[i], non_blocking=pinned)
        torch.cuda.synchronize(device=device)

    def _d2h_multi_stream() -> None:
        for i in range(num_streams):
            with torch.cuda.stream(streams[i]):
                hosts[i].copy_(devs[i], non_blocking=pinned)
        torch.cuda.synchronize(device=device)

    if mode == "threaded":

        def _h2d_threaded() -> None:
            with ThreadPoolExecutor(max_workers=num_streams) as pool:
                list(
                    pool.map(
                        lambda i: devs[i].copy_(hosts[i], non_blocking=pinned),
                        range(num_streams),
                    )
                )
            torch.cuda.synchronize(device=device)

        def _d2h_threaded() -> None:
            with ThreadPoolExecutor(max_workers=num_streams) as pool:
                list(
                    pool.map(
                        lambda i: hosts[i].copy_(devs[i], non_blocking=pinned),
                        range(num_streams),
                    )
                )
            torch.cuda.synchronize(device=device)

        return _TransferPlan(
            total, pinned, num_streams, mode, _h2d_threaded, _d2h_threaded
        )

    return _TransferPlan(
        total, pinned, num_streams, "multi_stream", _h2d_multi_stream, _d2h_multi_stream
    )


def _build_single(nbytes: int, device: str, pinned: bool) -> _TransferPlan:
    host = torch.empty(nbytes, dtype=torch.int8, pin_memory=pinned)
    dev = torch.empty(nbytes, dtype=torch.int8, device=device)

    def _h2d() -> None:
        dev.copy_(host, non_blocking=pinned)
        torch.cuda.synchronize(device=device)

    def _d2h() -> None:
        host.copy_(dev, non_blocking=pinned)
        torch.cuda.synchronize(device=device)

    return _TransferPlan(nbytes, pinned, 1, "single", _h2d, _d2h)


class HostDeviceBandwidthBenchmark(Benchmark):
    name = "host_device_bw"
    category = "memory"
    description = (
        "H2D/D2H bandwidth (pinned/pageable) via multi-stream parallel copies."
    )

    def run(self, ctx: RunContext) -> list[BenchmarkResult]:
        device = ctx.torch_device
        nbytes = int(ctx.sizes.get("transfer_bytes", 256 * 1024 * 1024))
        num_streams = int(ctx.sizes.get("transfer_streams", 8))
        mode = str(ctx.sizes.get("transfer_mode", "multi_stream"))
        timer = CudaTimer(ctx.warmup, ctx.rep, flush_l2=False, device=device)

        results: list[BenchmarkResult] = []
        peak = self._theoretical(ctx)

        for pinned in (True, False):
            try:
                plan = _build_plan(nbytes, device, pinned, num_streams, mode)
            except Exception as e:
                results.append(self._err("h2d", pinned, nbytes, e, num_streams, mode))
                results.append(self._err("d2h", pinned, nbytes, e, num_streams, mode))
                continue

            for direction, fn in (("h2d", plan.h2d_fn), ("d2h", plan.d2h_fn)):
                try:
                    stats = timer.time(fn)
                except Exception as e:
                    results.append(
                        self._err(
                            direction, pinned, plan.nbytes, e, num_streams, mode
                        )
                    )
                    continue
                results.append(
                    self._ok(
                        direction,
                        pinned,
                        plan.nbytes,
                        stats,
                        ctx,
                        peak,
                        num_streams=plan.num_streams,
                        mode=plan.mode,
                    )
                )

        torch.cuda.empty_cache()
        return results

    def _ok(
        self,
        direction,
        pinned,
        nbytes,
        stats,
        ctx,
        peak,
        *,
        num_streams: int,
        mode: str,
    ) -> BenchmarkResult:
        gbps = nbytes / (stats.median_ms * 1e-3) / _GB
        tag = "pinned" if pinned else "pageable"
        # Link ceiling applies to both; pageable rarely reaches it due to CPU staging.
        theo = peak.value
        theo_src = peak.source
        theo_detail = peak.detail
        if not pinned and theo_detail:
            theo_detail = f"{theo_detail}; pageable CPU staging (link ceiling)"

        return BenchmarkResult(
            name=f"{direction}_{tag}",
            category=self.category,
            value=gbps,
            unit="GB/s",
            config={
                "bytes": nbytes,
                "pinned": pinned,
                "direction": direction,
                "streams": num_streams,
                "mode": mode,
            },
            timing=stats,
            theoretical=theo,
            theoretical_source=theo_src,
            theoretical_detail=theo_detail,
            extra={"link": self._link_label(ctx)},
        )

    def _err(
        self, direction, pinned, nbytes, exc, num_streams: int, mode: str
    ) -> BenchmarkResult:
        tag = "pinned" if pinned else "pageable"
        return BenchmarkResult(
            name=f"{direction}_{tag}",
            category=self.category,
            value=0.0,
            unit="GB/s",
            config={
                "bytes": nbytes,
                "pinned": pinned,
                "direction": direction,
                "streams": num_streams,
                "mode": mode,
            },
            error=f"{type(exc).__name__}: {exc}",
        )

    @staticmethod
    def _theoretical(ctx: RunContext):
        from syspeek.theoretical import SpecSourceMode, resolve_theoretical

        mode = ctx.spec_source or SpecSourceMode.AUTO_FALLBACK
        return resolve_theoretical(ctx.device, "host_device_bandwidth_gbps", mode)

    @staticmethod
    def _link_label(ctx: RunContext) -> str:
        from syspeek.platforms import get_platform

        return get_platform(ctx.device).host_device_link_label()
