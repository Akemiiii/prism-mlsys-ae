#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GPU_LLAMA2="${1:-0}"
GPU_LLAMA3="${2:-1}"

if ! [[ "$GPU_LLAMA2" =~ ^[0-9]+$ ]]; then
  echo "Error: llama2 gpu id must be an integer, got: $GPU_LLAMA2" >&2
  exit 1
fi
if ! [[ "$GPU_LLAMA3" =~ ^[0-9]+$ ]]; then
  echo "Error: llama3 gpu id must be an integer, got: $GPU_LLAMA3" >&2
  exit 1
fi

echo "Starting parallel Table 5 runs..."
echo "llama2 -> GPU ${GPU_LLAMA2} (port 30000), llama3 -> GPU ${GPU_LLAMA3} (port 30001)"

start_llama2="$(date +%s)"
./run_llama2_table5.sh "$GPU_LLAMA2" 30000 &
pid_llama2=$!
echo "llama2 started (pid=${pid_llama2})"

start_llama3="$(date +%s)"
./run_llama3_table5.sh "$GPU_LLAMA3" 30001 &
pid_llama3=$!
echo "llama3 started (pid=${pid_llama3})"

status_llama2=0
status_llama3=0

wait "$pid_llama2"
status_llama2=$?
end_llama2="$(date +%s)"
elapsed_llama2=$((end_llama2 - start_llama2))

wait "$pid_llama3"
status_llama3=$?
end_llama3="$(date +%s)"
elapsed_llama3=$((end_llama3 - start_llama3))

echo "----------------------------------------"
echo "llama2 status: ${status_llama2}, elapsed: ${elapsed_llama2}s"
echo "llama3 status: ${status_llama3}, elapsed: ${elapsed_llama3}s"

if [[ "$status_llama2" -eq 0 && "$status_llama3" -eq 0 ]]; then
  echo "Both parallel runs finished successfully."
  exit 0
fi

echo "One or more parallel runs failed."
exit 1
