"""GEMM throughput benchmark (achieved FLOPS across dtypes).

Measures matrix-multiply throughput via cuBLAS (``torch.matmul`` / ``_int_mm`` /
``_scaled_mm``). This reflects **achievable** framework throughput, not NVIDIA
datasheet **sparse** peaks (FP4 / structured sparsity).

Optimizations (see ``--gemm-fast``, no L2 flush during timing):
  - Compute timing disables L2 flush (unlike memory cold-cache probes).
  - ``gemm_fast``: FP16/BF16 Tensor Core fast accumulation (when supported).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, Optional

import torch

from syspeek.benchmarks.base import Benchmark
from syspeek.core.result import BenchmarkResult, RunContext
from syspeek.core.timing import CudaTimer

_DTYPE_UNIT = {
    "fp32": "TFLOPS",
    "tf32": "TFLOPS",
    "fp16": "TFLOPS",
    "bf16": "TFLOPS",
    "int8": "TOPS",
    "fp8": "TFLOPS",
}
_DEFAULT_DTYPES = ["fp32", "tf32", "fp16", "bf16", "int8", "fp8"]


@contextmanager
def _tf32_context(enabled: bool) -> Iterator[None]:
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    prev_cd = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_mm
        torch.backends.cudnn.allow_tf32 = prev_cd


@contextmanager
def _fast_accum_context(enabled: bool) -> Iterator[None]:
    """Enable FP16/BF16 Tensor Core fast accumulation (FP16/BF16 acc, not FP32)."""
    prev_fp16 = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    prev_bf16 = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    if enabled:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = prev_fp16
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = prev_bf16


def _make_matmul(
    dtype: str,
    m: int,
    n: int,
    k: int,
    device: str,
    *,
    fast_accum: bool = False,
):
    """Return a zero-arg callable performing one GEMM, or None if unsupported."""
    if dtype in ("fp16", "bf16", "fp32", "tf32"):
        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "tf32": torch.float32,
        }[dtype]
        a = torch.randn(m, k, dtype=torch_dtype, device=device)
        b = torch.randn(k, n, dtype=torch_dtype, device=device)
        c = torch.empty(m, n, dtype=torch_dtype, device=device)
        use_tf32 = dtype == "tf32"
        use_fast = fast_accum and dtype in ("fp16", "bf16")

        def fn():
            with _tf32_context(use_tf32):
                with _fast_accum_context(use_fast):
                    torch.matmul(a, b, out=c)

        return fn

    if dtype == "int8":
        a = torch.randint(-8, 8, (m, k), dtype=torch.int8, device=device)
        b = torch.randint(-8, 8, (k, n), dtype=torch.int8, device=device)
        torch._int_mm(a, b)

        def fn():
            torch._int_mm(a, b)

        return fn

    if dtype == "fp8":
        fp8 = getattr(torch, "float8_e4m3fn", None)
        if fp8 is None:
            return None
        a = torch.randn(m, k, device=device).to(fp8)
        b = torch.randn(n, k, device=device).to(fp8).t()
        scale = torch.tensor(1.0, device=device)
        # bf16 output is common in inference; fp16 may be closer to peak on some stacks.
        out_dtype = torch.bfloat16
        torch._scaled_mm(a, b, scale_a=scale, scale_b=scale, out_dtype=out_dtype)

        def fn():
            torch._scaled_mm(a, b, scale_a=scale, scale_b=scale, out_dtype=out_dtype)

        return fn

    return None


class GemmFlopsBenchmark(Benchmark):
    name = "gemm_flops"
    category = "compute"
    description = "Achieved GEMM throughput (cuBLAS via PyTorch; dense, no sparsity)."

    def run(self, ctx: RunContext) -> list[BenchmarkResult]:
        device = ctx.torch_device
        size = int(ctx.sizes.get("gemm_size", 8192))
        fast_accum = bool(ctx.sizes.get("gemm_fast", False))
        dtypes = ctx.dtypes or _DEFAULT_DTYPES
        # L2 flush measures cold DRAM; for compute GEMM it artificially lowers TFLOPS.
        timer = CudaTimer(ctx.warmup, ctx.rep, flush_l2=False, device=device)

        results: list[BenchmarkResult] = []
        for dtype in dtypes:
            unit = _DTYPE_UNIT.get(dtype, "TFLOPS")
            m = n = k = self._fit_size(size, dtype, device, fast_accum)
            accum_label = self._accum_label(dtype, fast_accum)
            try:
                fn = self._build(dtype, m, n, k, device, fast_accum=fast_accum)
            except Exception as e:
                results.append(
                    BenchmarkResult(
                        name=f"gemm_{dtype}",
                        category=self.category,
                        value=0.0,
                        unit=unit,
                        config=self._config(m, n, k, dtype, accum_label, fast_accum),
                        error=f"unsupported: {type(e).__name__}: {e}",
                    )
                )
                continue
            if fn is None:
                results.append(
                    BenchmarkResult(
                        name=f"gemm_{dtype}",
                        category=self.category,
                        value=0.0,
                        unit=unit,
                        config={"dtype": dtype},
                        error="unsupported on this build/GPU",
                    )
                )
                continue

            stats = timer.time(fn)
            flop = 2.0 * m * n * k
            tflops = flop / (stats.median_ms * 1e-3) / 1e12
            peak = self._theoretical(ctx, dtype, fast_accum=fast_accum)
            results.append(
                BenchmarkResult(
                    name=f"gemm_{dtype}",
                    category=self.category,
                    value=tflops,
                    unit=unit,
                    config=self._config(m, n, k, dtype, accum_label, fast_accum),
                    timing=stats,
                    theoretical=peak.value,
                    theoretical_source=peak.source,
                    theoretical_detail=peak.detail,
                )
            )
            del fn
            torch.cuda.empty_cache()
        return results

    def _build(
        self, dtype, m, n, k, device, *, fast_accum: bool
    ) -> Optional[Callable]:
        return _make_matmul(dtype, m, n, k, device, fast_accum=fast_accum)

    @staticmethod
    def _accum_label(dtype: str, fast_accum: bool) -> str:
        if dtype in ("fp16", "bf16"):
            return "fp16/bf16_acc" if fast_accum else "fp32_acc"
        if dtype == "tf32":
            return "tf32_tensor"
        return "default"

    @staticmethod
    def _config(
        m: int, n: int, k: int, dtype: str, accum: str, fast_accum: bool
    ) -> dict:
        return {
            "M": m,
            "N": n,
            "K": k,
            "dtype": dtype,
            "accum": accum,
            "gemm_fast": fast_accum,
        }

    @staticmethod
    def _fit_size(size: int, dtype: str, device: str, fast_accum: bool) -> int:
        cur = size
        while cur >= 512:
            try:
                fn = _make_matmul(dtype, cur, cur, cur, device, fast_accum=fast_accum)
                if fn is not None:
                    fn()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                return cur
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                cur //= 2
            except Exception:
                return cur
        return cur

    @staticmethod
    def _theoretical(ctx: RunContext, dtype: str, *, fast_accum: bool = False):
        from syspeek.theoretical import SpecSourceMode, resolve_theoretical

        mode = ctx.spec_source or SpecSourceMode.AUTO_FALLBACK
        metric = f"flops_{dtype}_tflops"
        if fast_accum and dtype in ("fp16", "bf16"):
            metric = f"flops_{dtype}_fast_tflops"
        return resolve_theoretical(ctx.device, metric, mode)
