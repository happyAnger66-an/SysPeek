"""SysPeek: a cross-GPU hardware benchmark tool.

Measures *achieved* hardware metrics on NVIDIA GPUs:
  - Compute throughput (GEMM TFLOPS across dtypes)
  - Host<->Device bandwidth (PCIe / shared-memory on Jetson)
  - On-device memory bandwidth (GDDR / LPDDR on GPU)
  - Kernel launch latency

Designed to run on both discrete GPUs (e.g. RTX 4070) and integrated
Jetson platforms (e.g. Jetson Thor) via a thin platform-abstraction layer.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
