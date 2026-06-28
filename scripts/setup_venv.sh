#!/usr/bin/env bash
# Create a project-local venv and install SysPeek in editable mode.
#
# Usage:
#   ./scripts/setup_venv.sh              # use .venv in repo root
#   ./scripts/setup_venv.sh /path/to/venv
#
# PyTorch is installed separately — pick CUDA index for your platform:
#   Desktop (CUDA 12.x): cu128
#   Jetson Thor (CUDA 13): cu130

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${1:-${ROOT}/.venv}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating venv at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install -U pip setuptools wheel

# Lightweight runtime deps (torch installed separately — large CUDA wheels).
python -m pip install "click>=8.0" "rich>=13.0" "pydantic>=2.0" "numpy>=1.24" "safetensors>=0.4"

# Editable install without re-resolving deps (torch may already exist in venv).
python -m pip install -e "${ROOT}" --no-deps

echo ""
echo "Done. Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "Install PyTorch with CUDA support (pick index for your platform):"
  echo "  Desktop x86_64 (CUDA 12.x): pip install torch --index-url https://download.pytorch.org/whl/cu128"
  echo "  Jetson Thor (CUDA 13):     pip install torch --index-url https://download.pytorch.org/whl/cu130"
  echo ""
fi
echo "Then run:"
echo "  syspeek info"
echo "  syspeek run"
