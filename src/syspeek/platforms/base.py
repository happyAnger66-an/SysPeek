"""Base platform abstraction."""

from __future__ import annotations

from typing import Any, Optional

from syspeek.core.device import DeviceInfo
from syspeek.theoretical import SpecSourceMode, lookup_theoretical


class Platform:
    """Default platform behaviour (discrete-GPU assumptions)."""

    kind = "generic"

    def __init__(self, device: DeviceInfo) -> None:
        self.device = device

    # --- host <-> device transfer semantics ---------------------------------

    @property
    def uses_dedicated_link(self) -> bool:
        """True if host<->device crosses a dedicated link (e.g. PCIe)."""
        return not self.device.is_integrated

    def host_device_link_label(self) -> str:
        """Human-readable label for the host<->device path."""
        return "PCIe"

    # --- theoretical peaks ----------------------------------------------------

    def theoretical(
        self,
        metric: str,
        mode: SpecSourceMode = SpecSourceMode.AUTO_FALLBACK,
        **kwargs: Any,
    ) -> Optional[float]:
        """Return a known theoretical peak for ``metric`` in its native unit."""
        peak = lookup_theoretical(self.device, metric, mode=mode, **kwargs)
        return peak.value

    # --- telemetry (optional, best-effort) -----------------------------------

    def read_telemetry(self) -> dict[str, float]:
        """Best-effort power/clock/temperature snapshot. Empty if unsupported."""
        return {}

    def notes(self) -> list[str]:
        """Platform-specific caveats to surface in the report."""
        return []
