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
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
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
# CUDA_VISIBLE_DEVICES="$GPU_ID" python "$LLAMA2_WO_SCRIPT" --port "$PORT"

DEFAULT_SGLANG054_PYTHON="${REPO_ROOT}/sglang054/bin/python"
SGLANG054_PYTHON="${SGLANG054_PYTHON:-$DEFAULT_SGLANG054_PYTHON}"
if [[ ! -x "$SGLANG054_PYTHON" ]]; then
  echo "Error: sglang054 venv python not found or not executable: $SGLANG054_PYTHON" >&2
  echo "Hint: export SGLANG054_PYTHON=/path/to/sglang054/bin/python" >&2
  echo "Hint: default looked at: ${DEFAULT_SGLANG054_PYTHON}" >&2
  exit 1
fi

echo "[2/2] Running in sglang054 venv: e2e_llama2_standalone.py (GPU=${GPU_ID}, port=${PORT})"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$SGLANG054_PYTHON" "$LLAMA2_STANDALONE_SCRIPT" --port "$PORT"

echo "Table 5 llama2 experiments finished."
