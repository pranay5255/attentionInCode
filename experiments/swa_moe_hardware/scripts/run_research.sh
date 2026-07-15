#!/usr/bin/env bash
set -euo pipefail

uv run --extra modal modal run -m \
  experiments.swa_moe_hardware.research_runner::app.research \
  "$@"
