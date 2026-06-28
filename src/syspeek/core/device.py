"""Device detection and static hardware information."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch


@dataclass
class DeviceInfo:
    """Static information about the GPU under test."""

    index: int
    name: str
    compute_capability: tuple[int, int]
    total_memory_bytes: int
    multi_processor_count: int
    l2_cache_size_bytes: int
    is_integrated: bool  # True for Jetson / iGPU sharing system memory
    platform: str = "desktop"  # "desktop" | "jetson"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def sm_arch(self) -> str:
        major, minor = self.compute_capability
        return f"sm_{major}{minor}"

    @property
    def total_memory_gb(self) -> float:
        return self.total_memory_bytes / (1024**3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "compute_capability": f"{self.compute_capability[0]}.{self.compute_capability[1]}",
            "sm_arch": self.sm_arch,
            "total_memory_gb": round(self.total_memory_gb, 2),
            "multi_processor_count": self.multi_processor_count,
            "l2_cache_size_mb": round(self.l2_cache_size_bytes / (1024**2), 2),
            "is_integrated": self.is_integrated,
            "platform": self.platform,
            "extra": self.extra,
        }


def _read_jetson_model() -> Optional[str]:
    """Return the Jetson board model string if running on a Tegra platform."""
    # JetPack writes a release file; device-tree exposes the board model.
    nv_release = Path("/etc/nv_tegra_release")
    dt_model = Path("/proc/device-tree/model")
    if dt_model.exists():
        try:
            model = dt_model.read_bytes().decode("utf-8", "ignore").strip("\x00").strip()
            if model:
                return model
        except Exception:
            pass
    if nv_release.exists():
        try:
            return nv_release.read_text().strip().splitlines()[0]
        except Exception:
            return "Jetson"
    return None


def is_jetson() -> bool:
    """Heuristic: detect an NVIDIA Jetson / Tegra integrated platform."""
    if os.environ.get("SYSPEEK_FORCE_JETSON") == "1":
        return True
    if Path("/etc/nv_tegra_release").exists():
        return True
    model = _read_jetson_model()
    if model and ("jetson" in model.lower() or "tegra" in model.lower() or "thor" in model.lower()):
        return True
    return False


def detect_device(index: int = 0) -> DeviceInfo:
    """Build a :class:`DeviceInfo` for the GPU at ``index``."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device available. SysPeek requires a working CUDA GPU."
        )

    props = torch.cuda.get_device_properties(index)
    jetson = is_jetson()
    model = _read_jetson_model() if jetson else None

    extra: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "driver_total_memory_bytes": int(props.total_memory),
    }
    if model:
        extra["board_model"] = model

    return DeviceInfo(
        index=index,
        name=props.name,
        compute_capability=(props.major, props.minor),
        total_memory_bytes=int(props.total_memory),
        multi_processor_count=int(props.multi_processor_count),
        l2_cache_size_bytes=int(getattr(props, "l2_cache_size", 0) or 0),
        is_integrated=jetson or bool(getattr(props, "is_integrated", False)),
        platform="jetson" if jetson else "desktop",
        extra=extra,
    )
