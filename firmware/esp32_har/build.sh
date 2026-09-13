#!/usr/bin/env bash
# Build the firmware exactly as used for the reported numbers.
# -O2 instead of the Arduino default -Os: ~7x lower inference latency for +14 KB flash (see report).
set -euo pipefail
cd "$(dirname "$0")"
arduino-cli compile --fqbn esp32:esp32:esp32 \
  --build-property "compiler.optimization_flags=-O2" \
  --output-dir build . | grep -E "Sketch uses|Global variables"
