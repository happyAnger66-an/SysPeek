"""Theoretical hardware peaks for efficiency (%) calculation.

Two sources are supported:

* **fixed** — values from ``_TABLE`` (vendor specs, manually curated).
* **auto** — derived at runtime from CUDA device properties + nvidia-smi clocks.

Use :class:`SpecSourceMode` to pick a strategy; ``auto_fallback`` tries auto first
and falls back to fixed when auto derivation fails for a metric.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from syspeek.core.device import DeviceInfo


class SpecSourceMode(str, Enum):
    """How theoretical peaks are resolved."""

    AUTO = "auto"
    FIXED = "fixed"
    AUTO_FALLBACK = "auto_fallback"


@dataclass(frozen=True)
class TheoreticalPeak:
    """A theoretical peak with provenance."""

    value: Optional[float]
    source: Optional[str] = None  # "auto" | "fixed" | None
    detail: Optional[str] = None  # human-readable derivation / table key

    @property
    def has_value(self) -> bool:
        return self.value is not None and self.value > 0


# FP32 CUDA cores per SM (dense, no sparsity).
_CORES_PER_SM: dict[tuple[int, int], int] = {
    (12, 0): 128,  # Blackwell consumer (approx.)
    (9, 0): 128,   # Hopper
    (8, 9): 128,   # Ada
    (8, 6): 128,   # Ampere
    (8, 0): 64,    # Ampere (A100 uses 64 FP32/SM in some docs; keep conservative)
    (7, 5): 64,    # Turing
    (7, 0): 64,    # Volta
}

# GDDR effective data-rate factor: bandwidth = bus_bits * mem_clock_mhz * factor / 8000
# GDDR6X (sm_89 consumer): factor=16 → 192 * 1313 * 16 / 8000 ≈ 504 GB/s
_GDDR_RATE_FACTOR: dict[tuple[int, int], int] = {
    (8, 9): 16,  # GDDR6X
    (8, 6): 16,  # GDDR6X on many Ampere cards
    (7, 5): 8,   # GDDR6
}

# LPDDR5 effective factor for integrated (Jetson): conservative default
_LPDDR_RATE_FACTOR = 4

# Bus width (bits) hints when nvidia-smi does not expose memory.busWidth.
_BUS_WIDTH_HINTS: dict[str, int] = {
    "rtx 4090": 384,
    "rtx 4080": 256,
    "rtx 4070": 192,
    "rtx 4060": 128,
    "rtx 3090": 384,
    "rtx 3080": 320,
    "rtx 3070": 256,
    "rtx 3060": 192,
}

# GPU-side PCIe lane count (may be lower than motherboard link width).
_GPU_PCIE_LANE_HINTS: dict[str, int] = {
    "rtx 4090": 16,
    "rtx 4080": 16,
    "rtx 4070": 8,
    "rtx 4060": 8,
    "rtx 3090": 16,
    "rtx 3080": 16,
    "rtx 3070": 16,
    "rtx 3060": 16,
}

# Per-lane peak throughput (GB/s, one direction), 128b/130b encoding.
_PCIE_LANE_GBPS: dict[int, float] = {
    1: 0.250,
    2: 0.500,
    3: 0.985,
    4: 1.969,
    5: 3.938,
}

# NOTE: dense (no 2:4 sparsity) figures. Fill in / correct per your silicon.
_TABLE: dict[str, dict[str, float]] = {
    "rtx 4070": {
        "mem_bandwidth_gbps": 504.2,
        "pcie_bandwidth_gbps": 15.75,  # Gen4 x8, one direction
        "flops_fp32_tflops": 29.1,
        "flops_tf32_tflops": 29.1,
        "flops_fp16_tflops": 58.2,  # cuBLAS FP32 acc (GeForce Ada)
        "flops_bf16_tflops": 58.2,
        "flops_fp16_fast_tflops": 116.3,  # FP16 acc peak (not default torch path)
        "flops_bf16_fast_tflops": 116.3,
        "flops_int8_tflops": 232.0,
        "flops_fp8_tflops": 232.0,
    },
    # Jetson T5000 (Thor) — MAXN, **dense** GEMM (no 2:4 sparsity). See DS-11945-001.
    # Marketing "2070 TFLOPS" is sparse FP4; do not compare directly to SysPeek dense GEMM.
    "jetson t5000": {
        "mem_bandwidth_gbps": 273.0,
        "flops_fp32_tflops": 8.064,  # 2560 CUDA cores @ 1.575 GHz MAXN
        "flops_tf32_tflops": 42.0,  # dense TF32 tensor (cuBLAS), below sparse peak
        "flops_fp16_tflops": 128.0,  # dense FP16, FP32 accumulate (cuBLAS default)
        "flops_bf16_tflops": 128.0,
        "flops_fp16_fast_tflops": 258.0,  # dense FP16/BF16 acc (~½ sparse FP16 517)
        "flops_bf16_fast_tflops": 258.0,
        "flops_int8_tflops": 517.0,  # dense MAXN (datasheet dense FP8 ≡ INT8)
        "flops_fp8_tflops": 517.0,
    },
    "thor": {
        "mem_bandwidth_gbps": 273.0,
        "flops_fp32_tflops": 8.064,
        "flops_tf32_tflops": 42.0,
        "flops_fp16_tflops": 128.0,
        "flops_bf16_tflops": 128.0,
        "flops_fp16_fast_tflops": 258.0,
        "flops_bf16_fast_tflops": 258.0,
        "flops_int8_tflops": 517.0,
        "flops_fp8_tflops": 517.0,
    },
}


# Known CUDA core counts when device props only expose SM count.
_CUDA_CORE_COUNT_HINTS: dict[str, int] = {
    "jetson t5000": 2560,
    "t5000": 2560,
    "thor": 2560,
    "jetson t4000": 1536,
    "t4000": 1536,
}


def _cuda_core_count(device: "DeviceInfo") -> Optional[int]:
    hint_key = _match_table_key(device)
    if hint_key and hint_key in _CUDA_CORE_COUNT_HINTS:
        return _CUDA_CORE_COUNT_HINTS[hint_key]
    name = device.name.lower()
    board = str(device.extra.get("board_model", "")).lower()
    for key, cores in _CUDA_CORE_COUNT_HINTS.items():
        if key in name or key in board:
            return cores
    cpsm = _cores_per_sm(device.compute_capability)
    if cpsm is None:
        return None
    return device.multi_processor_count * cpsm


def _bus_width_bits(device: "DeviceInfo") -> Optional[int]:
    smi = _query_nvidia_smi(device.index, "memory.busWidth")
    if smi:
        try:
            w = int(float(smi[0]))
            if w > 0:
                return w
        except ValueError:
            pass
    key = _match_table_key(device)
    if key and key in _BUS_WIDTH_HINTS:
        return _BUS_WIDTH_HINTS[key]
    name = device.name.lower()
    for hint_key, width in _BUS_WIDTH_HINTS.items():
        if hint_key in name:
            return width
    # Jetson unified memory
    if device.platform == "jetson" or device.is_integrated:
        return 256
    return None


def _mem_bandwidth_gbps(bus_bits: int, mem_mhz: float, cc: tuple[int, int]) -> float:
    """Convert bus width + memory clock to peak GB/s."""
    if mem_mhz > 5000:
        # Recent drivers: clocks.max.memory is an effective aggregate rate.
        return bus_bits * mem_mhz * 2 / 8000.0
    factor = _LPDDR_RATE_FACTOR if cc >= (9, 0) else _GDDR_RATE_FACTOR.get(cc, 16)
    return bus_bits * mem_mhz * factor / 8000.0


def _match_table_key(device: "DeviceInfo") -> Optional[str]:
    name = device.name.lower()
    for key in _TABLE:
        if key in name:
            return key
    board = str(device.extra.get("board_model", "")).lower()
    for key in _TABLE:
        if key in board:
            return key
    return None


def _lookup_fixed(device: "DeviceInfo", metric: str) -> TheoreticalPeak:
    key = _match_table_key(device)
    if key is None:
        return TheoreticalPeak(None, None, "no fixed table entry for this device")
    val = _TABLE[key].get(metric)
    if val is None:
        return TheoreticalPeak(None, None, f"fixed table '{key}' has no '{metric}'")
    return TheoreticalPeak(val, "fixed", f"table:{key}")


def _query_nvidia_smi(device_index: int, fields: str) -> Optional[list[str]]:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
                "-i",
                str(device_index),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        line = out.stdout.strip().splitlines()[0]
        return [p.strip() for p in line.split(",")]
    except Exception:
        return None


def _is_consumer_geforce(device: "DeviceInfo") -> bool:
    n = device.name.lower()
    if device.is_integrated or device.platform == "jetson":
        return False
    datacenter_markers = ("tesla", "a100", "h100", "h200", "l40", "a10", "a30", "a40")
    if any(m in n for m in datacenter_markers):
        return False
    return any(m in n for m in ("geforce", "rtx", "gtx", "quadro"))


def _cores_per_sm(cc: tuple[int, int]) -> Optional[int]:
    if cc in _CORES_PER_SM:
        return _CORES_PER_SM[cc]
    major, minor = cc
    if major >= 8:
        return 128
    if major >= 7:
        return 64
    return None


def _derive_fp32_tflops(device: "DeviceInfo") -> TheoreticalPeak:
    cores = _cuda_core_count(device)
    if cores is None or cores <= 0:
        return TheoreticalPeak(None, None, "unknown CUDA core count for this device")

    smi = _query_nvidia_smi(device.index, "clocks.max.sm")
    if not smi:
        smi = _query_nvidia_smi(device.index, "clocks.max.graphics")
    if not smi:
        return TheoreticalPeak(None, None, "nvidia-smi clock query failed")

    try:
        sm_mhz = float(smi[0])
    except ValueError:
        return TheoreticalPeak(None, None, f"invalid SM clock: {smi[0]!r}")

    if sm_mhz <= 0:
        return TheoreticalPeak(None, None, "SM clock is zero")

    tflops = round(cores * 2.0 * (sm_mhz * 1e6) / 1e12, 2)
    detail = f"{cores} CUDA cores, max SM clock {sm_mhz:.0f} MHz (nvidia-smi)"
    return TheoreticalPeak(tflops, "auto", detail)


def _derive_compute_auto(device: "DeviceInfo", metric: str) -> TheoreticalPeak:
    if not metric.startswith("flops_") or not metric.endswith("_tflops"):
        return TheoreticalPeak(None, None, f"unsupported compute metric: {metric}")

    body = metric[len("flops_") : -len("_tflops")]
    fast = body.endswith("_fast")
    dtype = body[:-5] if fast else body

    base = _derive_fp32_tflops(device)
    if not base.has_value:
        return base

    fp32 = base.value
    assert fp32 is not None
    consumer = _is_consumer_geforce(device)
    cc = device.compute_capability

    if dtype == "fp32":
        mult = 1.0
    elif dtype == "tf32":
        mult = 4.0 if not consumer else 4.0  # tensor TF32 vs CUDA
    elif dtype in ("fp16", "bf16"):
        if fast:
            mult = 8.0 if cc >= (9, 0) else 4.0
        else:
            mult = 2.0 if consumer else 4.0
    elif dtype in ("int8", "fp8"):
        if consumer and cc >= (8, 0):
            mult = 8.0
        elif cc >= (9, 0):
            mult = 16.0 if fast else 8.0
        else:
            mult = 4.0
    else:
        return TheoreticalPeak(None, None, f"unsupported dtype: {dtype}")

    val = round(fp32 * mult, 2)
    acc_note = "FP16/BF16 acc" if fast and dtype in ("fp16", "bf16") else (
        "FP32 acc" if dtype in ("fp16", "bf16") else "tensor path"
    )
    detail = f"{base.detail}; {dtype} ×{mult:g} ({acc_note})"
    return TheoreticalPeak(val, "auto", detail)


def _gpu_pcie_lane_cap(device: "DeviceInfo") -> Optional[int]:
    key = _match_table_key(device)
    if key and key in _GPU_PCIE_LANE_HINTS:
        return _GPU_PCIE_LANE_HINTS[key]
    name = device.name.lower()
    for hint_key, lanes in _GPU_PCIE_LANE_HINTS.items():
        if hint_key in name:
            return lanes
    return None


def _derive_pcie_bandwidth_auto(device: "DeviceInfo") -> TheoreticalPeak:
    """Peak host<->device link bandwidth (GB/s, one direction)."""
    if device.is_integrated or device.platform == "jetson":
        # Integrated: transfers share LPDDR with the GPU; use device mem as upper bound.
        mem = _derive_memory_auto(device)
        if mem.has_value:
            return TheoreticalPeak(
                mem.value,
                "auto",
                f"shared memory (Jetson); {mem.detail}",
            )
        return TheoreticalPeak(None, None, "integrated memory bandwidth unknown")

    smi = _query_nvidia_smi(
        device.index,
        "pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current",
    )
    if not smi or len(smi) < 2:
        return TheoreticalPeak(None, None, "nvidia-smi PCIe link query failed")

    try:
        gen = int(float(smi[0]))
        link_width = int(float(smi[1]))
        gen_cur = int(float(smi[2])) if len(smi) > 2 else gen
        width_cur = int(float(smi[3])) if len(smi) > 3 else link_width
    except ValueError:
        return TheoreticalPeak(None, None, f"invalid PCIe fields: {smi}")

    rate = _PCIE_LANE_GBPS.get(gen)
    if rate is None or link_width <= 0:
        return TheoreticalPeak(None, None, f"unsupported PCIe gen {gen}")

    gpu_cap = _gpu_pcie_lane_cap(device)
    effective_width = min(link_width, gpu_cap) if gpu_cap else link_width
    gbps = round(effective_width * rate, 2)
    cap_note = f", GPU cap x{gpu_cap}" if gpu_cap and gpu_cap < link_width else ""
    detail = (
        f"PCIe Gen{gen} x{effective_width} ({effective_width * rate:.2f} GB/s/dir); "
        f"link x{link_width}{cap_note}; active Gen{gen_cur} x{width_cur}"
    )
    return TheoreticalPeak(gbps, "auto", detail)


def _derive_memory_auto(device: "DeviceInfo") -> TheoreticalPeak:
    bus_bits = _bus_width_bits(device)
    if bus_bits is None:
        return TheoreticalPeak(
            None, None, "memory bus width unknown (no nvidia-smi field / hint)"
        )

    smi = _query_nvidia_smi(device.index, "clocks.max.memory")
    if not smi:
        return TheoreticalPeak(None, None, "nvidia-smi memory clock query failed")

    try:
        mem_mhz = float(smi[0])
    except ValueError:
        return TheoreticalPeak(None, None, f"invalid memory clock: {smi[0]!r}")

    if mem_mhz <= 0:
        return TheoreticalPeak(None, None, "memory clock is zero")

    cc = device.compute_capability
    gbps = _mem_bandwidth_gbps(bus_bits, mem_mhz, cc)
    mem_type = "LPDDR (est.)" if device.is_integrated or device.platform == "jetson" else "GDDR"
    detail = f"{bus_bits}-bit {mem_type}, mem clock {mem_mhz:.0f} MHz"
    return TheoreticalPeak(round(gbps, 2), "auto", detail)


def _derive_auto(device: "DeviceInfo", metric: str) -> TheoreticalPeak:
    if metric == "mem_bandwidth_gbps":
        return _derive_memory_auto(device)
    if metric in ("pcie_bandwidth_gbps", "host_device_bandwidth_gbps"):
        return _derive_pcie_bandwidth_auto(device)
    if metric.startswith("flops_"):
        return _derive_compute_auto(device, metric)
    return TheoreticalPeak(None, None, f"no auto derivation for '{metric}'")


def lookup_theoretical(
    device: "DeviceInfo",
    metric: str,
    mode: SpecSourceMode = SpecSourceMode.AUTO_FALLBACK,
) -> TheoreticalPeak:
    """Resolve the theoretical peak for ``metric`` using ``mode``."""
    if mode == SpecSourceMode.FIXED:
        return _lookup_fixed(device, metric)

    auto = _derive_auto(device, metric)
    if mode == SpecSourceMode.AUTO:
        return auto

    # AUTO_FALLBACK
    if auto.has_value:
        return auto
    fixed = _lookup_fixed(device, metric)
    if fixed.has_value:
        reason = auto.detail or "auto derivation failed"
        return TheoreticalPeak(
            fixed.value,
            "fixed",
            f"{fixed.detail}; fallback ({reason})",
        )
    return TheoreticalPeak(None, None, auto.detail or fixed.detail)


def resolve_theoretical(
    device: "DeviceInfo",
    metric: str,
    mode: SpecSourceMode,
) -> TheoreticalPeak:
    """Convenience wrapper used by benchmarks."""
    return lookup_theoretical(device, metric, mode)
