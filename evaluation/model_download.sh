#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="Akemiiii/Prism-AE"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/../models"

mkdir -p "${TARGET_DIR}"
modelscope download --model "${MODEL_ID}" --local_dir "${TARGET_DIR}"

echo "Models downloaded to: ${TARGET_DIR}"
ln -s ${TARGET_DIR}/Meta-Llama-3-8B-Instruct ${TARGET_DIR}/Llama-3-8B-Instruct
