#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <gpu_id> <port>"
  echo "Example: $0 0 30000"
  exit 1
fi

GPU_ID="$1"
PORT="$2"

if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
  echo "Error: port must be an integer, got: $PORT" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA2_WO_SCRIPT="${SCRIPT_DIR}/e2e_llama2_wo_standalone.py"
LLAMA2_STANDALONE_SCRIPT="${SCRIPT_DIR}/e2e_llama2_standalone.py"

if [[ ! -f "$LLAMA2_WO_SCRIPT" ]]; then
  echo "Error: script not found: $LLAMA2_WO_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$LLAMA2_STANDALONE_SCRIPT" ]]; then
  echo "Error: script not found: $LLAMA2_STANDALONE_SCRIPT" >&2
  exit 1
fi

echo "[1/2] Running in current environment: e2e_llama2_wo_standalone.py (GPU=${GPU_ID}, port=${PORT})"
CUDA_VISIBLE_DEVICES="$GPU_ID" python "$LLAMA2_WO_SCRIPT" --port "$PORT"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found, cannot switch to sglang054 environment" >&2
  exit 1
fi

echo "[2/2] Running in sglang054 environment: e2e_llama2_standalone.py (GPU=${GPU_ID}, port=${PORT})"
CUDA_VISIBLE_DEVICES="$GPU_ID" conda run -n sglang054 python "$LLAMA2_STANDALONE_SCRIPT" --port "$PORT"

echo "Table 5 llama2 experiments finished."
