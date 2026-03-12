#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PARALLEL_SCRIPT="${SCRIPT_DIR}/run_table4_parallel.sh"
LLAMA2_SCRIPT="${SCRIPT_DIR}/run_llama2_table4.sh"
LLAMA3_SCRIPT="${SCRIPT_DIR}/run_llama3_table4.sh"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_table4_from_csv.py"

if [[ ! -f "$PARALLEL_SCRIPT" ]]; then
  echo "Error: script not found: $PARALLEL_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$LLAMA2_SCRIPT" ]]; then
  echo "Error: script not found: $LLAMA2_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$LLAMA3_SCRIPT" ]]; then
  echo "Error: script not found: $LLAMA3_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$PLOT_SCRIPT" ]]; then
  echo "Error: script not found: $PLOT_SCRIPT" >&2
  exit 1
fi

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <CUDA_VISIBLE_DEVICES>" >&2
  echo "Example: $0 0" >&2
  echo "Example: $0 0,1" >&2
  exit 1
fi

CUDA_DEVICES="$1"
CUDA_DEVICES="${CUDA_DEVICES//[[:space:]]/}"

if [[ -z "$CUDA_DEVICES" ]]; then
  echo "Error: CUDA_VISIBLE_DEVICES cannot be empty." >&2
  exit 1
fi
if ! [[ "$CUDA_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "Error: invalid CUDA_VISIBLE_DEVICES format: $CUDA_DEVICES" >&2
  echo "Expected examples: 0 or 0,1 or 2,3,4" >&2
  exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "$CUDA_DEVICES"
GPU_COUNT="${#GPU_IDS[@]}"

echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES (count=${GPU_COUNT})"

if [[ "$GPU_COUNT" -ge 2 ]]; then
  GPU0="${GPU_IDS[0]}"
  GPU1="${GPU_IDS[1]}"
  echo "Detected >=2 GPUs, running in parallel mode..."
  echo "Using first two GPUs: llama2 -> ${GPU0}, llama3 -> ${GPU1}"
  bash "$PARALLEL_SCRIPT" "$GPU0" "$GPU1"
else
  GPU0="${GPU_IDS[0]}"
  echo "Detected 1 GPU, running in serial mode on GPU ${GPU0}..."
  bash "$LLAMA2_SCRIPT" "$GPU0" 30000
  bash "$LLAMA3_SCRIPT" "$GPU0" 30000
fi

echo "Generating table4 PDF..."
python "$PLOT_SCRIPT"

echo "Table 4 pipeline finished."
