#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  printf 'Usage: %s CONFIG_JSON [HARDWARE] [OUTPUT_DIR]\n' "$0" >&2
  exit 2
fi

CONFIG_PATH="$1"
HARDWARE="${2:-all}"
OUTPUT_DIR="${3:-runs/swa_moe_hardware}"

uv run --extra modal modal run -m \
  experiments.swa_moe_hardware.modal_runner \
  --config-path "${CONFIG_PATH}" \
  --hardware "${HARDWARE}" \
  --output-dir "${OUTPUT_DIR}"
