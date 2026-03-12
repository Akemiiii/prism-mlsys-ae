#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="Akem1i/Prism-AE"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/../models"

mkdir -p "${TARGET_DIR}"
hf download "${MODEL_ID}" --repo-type model --local-dir "${TARGET_DIR}"

echo "Models downloaded to: ${TARGET_DIR}"
ln -s ${TARGET_DIR}/Meta-Llama-3-8B-Instruct ${TARGET_DIR}/Llama-3-8B-Instruct
