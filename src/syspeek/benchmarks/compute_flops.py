"""GEMM throughput benchmark (achieved FLOPS across dtypes).

Measures matrix-multiply throughput, which is the standard proxy for achievable
compute performance and exercises the Tensor Cores for low-precision dtypes.
FLOPs per multiply of ``[M,K] x [K,N]`` is ``2*M*N*K``.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from syspeek.benchmarks.base import Benchmark
from syspeek.core.result import BenchmarkResult, RunContext
from syspeek.core.timing import CudaTimer

# dtype key -> (theoretical-metric suffix, headline unit)
_DTYPE_UNIT = {
    "fp32": "TFLOPS",
    "tf32": "TFLOPS",
    "fp16": "TFLOPS",
    "bf16": "TFLOPS",
    "int8": "TOPS",
    "fp8": "TFLOPS",
}
_DEFAULT_DTYPES = ["fp32", "tf32", "fp16", "bf16", "int8", "fp8"]


def _make_matmul(dtype: str, m: int, n: int, k: int, device: str):
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

        def fn():
            prev = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = use_tf32
            try:
                torch.matmul(a, b, out=c)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = prev

        return fn

    if dtype == "int8":
        a = torch.randint(-8, 8, (m, k), dtype=torch.int8, device=device)
        b = torch.randint(-8, 8, (k, n), dtype=torch.int8, device=device)
        # _int_mm validates support/shape lazily; probe once.
        torch._int_mm(a, b)

        def fn():
            torch._int_mm(a, b)

        return fn

    if dtype == "fp8":
        fp8 = getattr(torch, "float8_e4m3fn", None)
        if fp8 is None:
            return None
        a = torch.randn(m, k, device=device).to(fp8)
        # _scaled_mm wants the second operand column-major.
        b = torch.randn(n, k, device=device).to(fp8).t()
        scale = torch.tensor(1.0, device=device)
        torch._scaled_mm(a, b, scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)

        def fn():
            torch._scaled_mm(a, b, scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)

        return fn

    return None


class GemmFlopsBenchmark(Benchmark):
    name = "gemm_flops"
    category = "compute"
    description = "Achieved GEMM throughput across dtypes (fp32/tf32/fp16/bf16/int8/fp8)."

    def run(self, ctx: RunContext) -> list[BenchmarkResult]:
        device = ctx.torch_device
        size = int(ctx.sizes.get("gemm_size", 8192))
        dtypes = ctx.dtypes or _DEFAULT_DTYPES
        timer = CudaTimer(ctx.warmup, ctx.rep, ctx.flush_l2, device)

        results: list[BenchmarkResult] = []
        for dtype in dtypes:
            unit = _DTYPE_UNIT.get(dtype, "TFLOPS")
            m = n = k = self._fit_size(size, dtype, device)
            try:
                fn = self._build(dtype, m, n, k, device)
            except Exception as e:  # unsupported dtype/op on this GPU
                results.append(
                    BenchmarkResult(
                        name=f"gemm_{dtype}",
                        category=self.category,
                        value=0.0,
                        unit=unit,
                        config={"M": m, "N": n, "K": k, "dtype": dtype},
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
            peak = self._theoretical(ctx, dtype)
            results.append(
                BenchmarkResult(
                    name=f"gemm_{dtype}",
                    category=self.category,
                    value=tflops,
                    unit=unit,
                    config={"M": m, "N": n, "K": k, "dtype": dtype},
                    timing=stats,
                    theoretical=peak.value,
                    theoretical_source=peak.source,
                    theoretical_detail=peak.detail,
                )
            )
            del fn
            torch.cuda.empty_cache()
        return results

    def _build(self, dtype, m, n, k, device) -> Optional[Callable]:
        return _make_matmul(dtype, m, n, k, device)

    @staticmethod
    def _fit_size(size: int, dtype: str, device: str) -> int:
        """Shrink the GEMM size on OOM so the benchmark still produces a number."""
        cur = size
        while cur >= 512:
            try:
                fn = _make_matmul(dtype, cur, cur, cur, device)
                if fn is not None:
                    fn()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                return cur
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                cur //= 2
            except Exception:
                # Non-OOM error: let the caller surface it at the chosen size.
                return cur
        return cur

    @staticmethod
    def _theoretical(ctx: RunContext, dtype: str):
        from syspeek.theoretical import SpecSourceMode, resolve_theoretical

        mode = ctx.spec_source or SpecSourceMode.AUTO_FALLBACK
        return resolve_theoretical(ctx.device, f"flops_{dtype}_tflops", mode)
