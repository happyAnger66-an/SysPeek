"""Discrete-GPU desktop/server platform (e.g. RTX 4070)."""

from __future__ import annotations

import shutil
import subprocess
from typing import Optional

from syspeek.platforms.base import Platform


class DesktopPlatform(Platform):
    """Discrete NVIDIA GPU connected over PCIe with dedicated VRAM."""

    kind = "desktop"

    def host_device_link_label(self) -> str:
        return "PCIe"

    def read_telemetry(self) -> dict[str, float]:
        """Read power/clocks/temperature via nvidia-smi when available."""
        if shutil.which("nvidia-smi") is None:
            return {}
        query = "power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu"
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self.device.index),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
        except Exception:
            return {}
        fields = [f.strip() for f in out.stdout.strip().splitlines()[0].split(",")]
        keys = ["power_w", "sm_clock_mhz", "mem_clock_mhz", "temp_c", "util_pct"]
        telemetry: dict[str, float] = {}
        for k, v in zip(keys, fields):
            try:
                telemetry[k] = float(v)
            except ValueError:
                continue
        return telemetry

    def notes(self) -> list[str]:
        return []
