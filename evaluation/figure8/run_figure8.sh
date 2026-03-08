#!/usr/bin/env bash
set -euo pipefail

# Set the port directly here when needed.
PORT="30001"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_SCRIPT="${SCRIPT_DIR}/e2e_llama2_tree_verify_sweep.py"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_tree_from_csv.py"

if [[ ! -f "${SWEEP_SCRIPT}" ]]; then
  echo "Error: script not found: ${SWEEP_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${PLOT_SCRIPT}" ]]; then
  echo "Error: script not found: ${PLOT_SCRIPT}" >&2
  exit 1
fi

echo "[1/2] Running tree/verify sweep..."
python "${SWEEP_SCRIPT}" --port "${PORT}"

echo "[2/2] Plotting figure from CSV..."
python "${PLOT_SCRIPT}"

echo "Figure 8 pipeline finished."
