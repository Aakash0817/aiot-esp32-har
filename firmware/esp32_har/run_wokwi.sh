#!/usr/bin/env bash
# Headless Wokwi run that reproduces results/device_log.txt. Needs WOKWI_CLI_TOKEN (free, wokwi.com/dashboard/ci)
# and wokwi-cli (curl -L https://wokwi.com/ci/install.sh | sh).
set -euo pipefail
cd "$(dirname "$0")"
: "${WOKWI_CLI_TOKEN:?set WOKWI_CLI_TOKEN}"
wokwi-cli --timeout 240000 --scenario replay.scenario.yaml --serial-log-file ../../results/device_log.txt .
