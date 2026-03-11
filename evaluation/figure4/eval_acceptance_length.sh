#!/usr/bin/env bash
#SBATCH -J PRISM
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 128
#SBATCH --gres=gpu:8

set -euo pipefail

# -----------------------------
# Default config (overridable by args/env)
# -----------------------------
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
PROJECT="${PROJECT:-Llama-3-8B}"
MODEL="${MODEL:-Eagle3}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
BENCHES="${BENCHES:-}"

usage() {
  cat <<'EOF'
Usage:
  bash eval_acceptance_length.sh [options]

Options:
  --gpus <list>          CUDA_VISIBLE_DEVICES, e.g. 0,1,2,3
  --project <name>       PROJECT, e.g. Llama-2-7B / Llama-3-8B
  --model <name>         MODEL, e.g. PRISM / Eagle2 / Eagle3 / HASS
  --run-tag <tag>        RUN_TAG for eval_all output folder
  --benches <list>       Comma-separated benchmarks to run, e.g. mt_bench,humaneval
                         Available: mt_bench, humaneval, gsm8k, alpaca, sum, qa
                         Default: all 6 benchmarks
  -h, --help             Show help
EOF
}

# -----------------------------
# Parse CLI arguments
# -----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
    --project)
      PROJECT="$2"; shift 2 ;;
    --model)
      MODEL="$2"; shift 2 ;;
    --run-tag)
      RUN_TAG="$2"; shift 2 ;;
    --benches)
      BENCHES="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1 ;;
  esac
done

# -----------------------------
# Resolve project root robustly for both local and sbatch
# Priority:
# 1) PRISM_AE_ROOT (if manually set)
# 2) git toplevel from submit/current directory
# 3) fallback by relative path from BASE_DIR
# -----------------------------
BASE_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -n "${PRISM_AE_ROOT:-}" ]]; then
  ROOT_DIR="$(cd "${PRISM_AE_ROOT}" && pwd)"
elif git -C "${BASE_DIR}" rev-parse --show-toplevel >/dev/null 2>&1; then
  ROOT_DIR="$(git -C "${BASE_DIR}" rev-parse --show-toplevel)"
else
  ROOT_DIR="$(cd "${BASE_DIR}/../.." && pwd)"
fi

# Directory for figure4 outputs
FIG4_DIR="${ROOT_DIR}/evaluation/figure4"

# Codebase directories
PRISM_DIR="${ROOT_DIR}/PRISM"
PRISM_LEGACY_DIR="${ROOT_DIR}/PRISM_legacy"
EAGLE3_DIR="${ROOT_DIR}/Eagle3"

# -----------------------------
# Select codebase by model:
#   MODEL == PRISM  (case-insensitive) => PRISM
#   MODEL == Eagle3 (case-insensitive) => Eagle3
#   Otherwise                          => PRISM_legacy
# -----------------------------
MODEL_UPPER="$(echo "${MODEL}" | tr '[:lower:]' '[:upper:]')"
if [[ "${MODEL_UPPER}" == "PRISM" ]]; then
  CODEBASE_DIR="${PRISM_DIR}"
  ENGINE_NAME="PRISM"
elif [[ "${MODEL_UPPER}" == "EAGLE3" ]]; then
  CODEBASE_DIR="${EAGLE3_DIR}"
  ENGINE_NAME="Eagle3"
else
  CODEBASE_DIR="${PRISM_LEGACY_DIR}"
  ENGINE_NAME="PRISM_legacy"
fi

EVAL_SH="${CODEBASE_DIR}/eval_all.sh"

# -----------------------------
# Validate paths
# -----------------------------
if [[ ! -d "${CODEBASE_DIR}" ]]; then
  echo "[FATAL] ${ENGINE_NAME} directory not found: ${CODEBASE_DIR}"
  exit 1
fi

if [[ ! -f "${EVAL_SH}" ]]; then
  echo "[FATAL] eval_all.sh not found: ${EVAL_SH}"
  exit 1
fi

# -----------------------------
# Run evaluation
# -----------------------------
echo "[INFO] Start run"
echo "  BASE_DIR=${BASE_DIR}"
echo "  ROOT_DIR=${ROOT_DIR}"
echo "  FIG4_DIR=${FIG4_DIR}"
echo "  ENGINE=${ENGINE_NAME}"
echo "  CODEBASE_DIR=${CODEBASE_DIR}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  PROJECT=${PROJECT}"
echo "  MODEL=${MODEL}"
echo "  RUN_TAG=${RUN_TAG}"
echo "  BENCHES=${BENCHES:-<all>}"

pushd "${CODEBASE_DIR}" >/dev/null

set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
PROJECT="${PROJECT}" \
MODEL="${MODEL}" \
RUN_TAG="${RUN_TAG}" \
BENCHES="${BENCHES}" \
bash "${EVAL_SH}" "${PROJECT}" "${MODEL}" "${BENCHES}"
EVAL_RC=$?
set -e

popd >/dev/null

# -----------------------------
# Sync outputs to evaluation/figure4
# Source: <CODEBASE_DIR>/outputs/${PROJECT}/${MODEL}/${RUN_TAG}
# Target: <FIG4_DIR>/outputs/${PROJECT}/${MODEL}/${RUN_TAG}
# -----------------------------
SRC_RESULT="${CODEBASE_DIR}/outputs/${PROJECT}/${MODEL}/${RUN_TAG}"
DST_BASE="${FIG4_DIR}/outputs/${PROJECT}/${MODEL}"
DST_RESULT="${DST_BASE}/${RUN_TAG}"

mkdir -p "${DST_BASE}"

if [[ -d "${SRC_RESULT}" ]]; then
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${SRC_RESULT}/" "${DST_RESULT}/"
  else
    cp -a "${SRC_RESULT}" "${DST_RESULT}"
  fi
  ln -sfn "${DST_RESULT}" "${DST_BASE}/latest"
  echo "[INFO] Output synced to: ${DST_RESULT}"
  echo "[INFO] Latest link: ${DST_BASE}/latest"
else
  echo "[WARN] Source result not found: ${SRC_RESULT}"
fi

if [[ "${EVAL_RC}" -ne 0 ]]; then
  echo "[FINAL] eval_all.sh failed with code ${EVAL_RC}"
  exit "${EVAL_RC}"
fi

echo "[FINAL] Done."
