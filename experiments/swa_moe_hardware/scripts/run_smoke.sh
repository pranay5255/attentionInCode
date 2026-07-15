#!/usr/bin/env bash
set -euo pipefail

HARDWARE="${1:-all}"
OUTPUT_DIR="${2:-runs/swa_moe_hardware}"

uv run --extra modal modal run -m \
  experiments.swa_moe_hardware.modal_runner \
  --preset smoke \
  --hardware "${HARDWARE}" \
  --output-dir "${OUTPUT_DIR}"
