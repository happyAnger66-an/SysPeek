"""NVIDIA Jetson (integrated GPU) platform, e.g. Jetson Thor.

Key differences from a discrete GPU:
  - The GPU is integrated and shares physical memory (LPDDR) with the CPU.
    "H2D"/"D2H" copies do not cross PCIe; they exercise the memory subsystem,
    and zero-copy / pinned memory behaves differently.
  - ``total_memory`` reported by CUDA is carved out of system RAM.
  - Power/clock telemetry comes from tegrastats / sysfs, not nvidia-smi.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from syspeek.platforms.base import Platform


class JetsonPlatform(Platform):
    """Integrated Tegra GPU platform with shared system memory."""

    kind = "jetson"

    @property
    def uses_dedicated_link(self) -> bool:
        return False

    def host_device_link_label(self) -> str:
        return "shared-mem (LPDDR)"

    def read_telemetry(self) -> dict[str, float]:
        """Best-effort power read from the Jetson INA3221 sysfs rails."""
        telemetry: dict[str, float] = {}
        # Sum power across hwmon rails when exposed (path varies by JetPack).
        for base in Path("/sys/bus/i2c/drivers/ina3221").glob("*/hwmon/hwmon*"):
            total_mw = 0.0
            found = False
            for power_file in base.glob("power*_input"):
                try:
                    total_mw += float(power_file.read_text().strip())
                    found = True
                except Exception:
                    continue
            if found:
                telemetry["power_w"] = total_mw / 1000.0
                break
        return telemetry

    def notes(self) -> list[str]:
        return [
            "Integrated GPU: host<->device copies traverse shared LPDDR, not PCIe.",
            "For peak GEMM, set power mode to MAXN: sudo nvpmodel -m 0 && sudo jetson_clocks.",
            "NVIDIA marketing TFLOPS (e.g. 2070 sparse FP4) ≠ SysPeek dense cuBLAS GEMM.",
        ]
