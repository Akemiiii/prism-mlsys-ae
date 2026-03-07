#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting parallel Table 5 runs..."

start_llama2="$(date +%s)"
./run_llama2_table5.sh 0 30000 &
pid_llama2=$!
echo "llama2 started (pid=${pid_llama2})"

start_llama3="$(date +%s)"
./run_llama3_table5.sh 1 30001 &
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
