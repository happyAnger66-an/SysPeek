"""GPU timing utilities based on CUDA events.

Provides a :class:`CudaTimer` that runs warmup iterations, then measures a
callable over many repetitions using ``torch.cuda.Event`` pairs. Optionally
flushes the L2 cache before each timed iteration so measurements reflect a
cold-cache (DRAM-bound) scenario rather than cache residency.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import torch

from syspeek.core.result import TimingStats


class CudaTimer:
    """Times a callable on the GPU using CUDA events."""

    def __init__(
        self,
        warmup: int = 10,
        rep: int = 50,
        flush_l2: bool = True,
        device: str = "cuda:0",
    ) -> None:
        self.warmup = warmup
        self.rep = rep
        self.flush_l2 = flush_l2
        self.device = device
        self._l2_buffer: Optional[torch.Tensor] = None
        if flush_l2:
            self._l2_buffer = self._make_l2_buffer()

    def _make_l2_buffer(self) -> torch.Tensor:
        """Allocate a buffer ~2x the device L2 cache to evict it on demand."""
        l2_bytes = 0
        try:
            props = torch.cuda.get_device_properties(self.device)
            l2_bytes = int(getattr(props, "l2_cache_size", 0) or 0)
        except Exception:
            l2_bytes = 0
        if l2_bytes <= 0:
            l2_bytes = 64 * 1024 * 1024  # 64MB fallback
        size = int(l2_bytes * 2)
        return torch.empty(size, dtype=torch.int8, device=self.device)

    def _clear_l2(self) -> None:
        if self._l2_buffer is not None:
            self._l2_buffer.zero_()

    def time(
        self,
        fn: Callable[[], Any],
        setup: Optional[Callable[[], Any]] = None,
    ) -> TimingStats:
        """Measure ``fn`` ``rep`` times after ``warmup`` iterations.

        ``setup`` (optional) runs before each iteration and is *not* timed;
        use it to reset state (e.g. restore output buffers) when needed.
        """
        torch.cuda.synchronize(self.device)
        for _ in range(self.warmup):
            if setup is not None:
                setup()
            if self.flush_l2:
                self._clear_l2()
            fn()
        torch.cuda.synchronize(self.device)

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.rep)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.rep)]

        for i in range(self.rep):
            if setup is not None:
                setup()
            if self.flush_l2:
                self._clear_l2()
            start_events[i].record()
            fn()
            end_events[i].record()

        torch.cuda.synchronize(self.device)
        samples = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        return TimingStats(samples_ms=samples)
