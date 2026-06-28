"""Platform abstraction layer.

A *platform* captures behaviour that differs between hardware classes but is
orthogonal to individual benchmarks: how host<->device transfers should be
labelled, where known theoretical peaks come from, and how to read power/clock
telemetry. Benchmarks stay device-agnostic; the platform annotates results.
"""

from __future__ import annotations

from syspeek.core.device import DeviceInfo
from syspeek.platforms.base import Platform
from syspeek.platforms.desktop import DesktopPlatform
from syspeek.platforms.jetson import JetsonPlatform

__all__ = ["Platform", "DesktopPlatform", "JetsonPlatform", "get_platform"]


def get_platform(device: DeviceInfo) -> Platform:
    """Return the platform implementation matching ``device``."""
    if device.platform == "jetson" or device.is_integrated:
        return JetsonPlatform(device)
    return DesktopPlatform(device)
