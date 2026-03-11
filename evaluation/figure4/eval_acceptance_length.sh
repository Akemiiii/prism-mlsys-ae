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
PROJECT="${PROJECT:-}"
MODEL="${MODEL:-}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
BENCHES="${BENCHES:-}"

# -----------------------------
# Conda environment names
# -----------------------------
CONDA_ENV_DEFAULT="Specdec"
CONDA_ENV_EAGLE3="Eagle3"

# Initialize conda for this shell session (required for conda activate in scripts)
if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
elif [[ -d "${HOME}/miniconda3" ]]; then
  CONDA_BASE="${HOME}/miniconda3"
elif [[ -d "${HOME}/anaconda3" ]]; then
  CONDA_BASE="${HOME}/anaconda3"
else
  echo "[FATAL] Cannot locate conda installation."
  exit 1
fi

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Activate the default environment at startup
conda activate "${CONDA_ENV_DEFAULT}"
echo "[INFO] Activated default conda env: ${CONDA_ENV_DEFAULT}"

# -----------------------------
# Valid combinations
# -----------------------------
ALL_PROJECTS=("Llama-2-7B" "Llama-3-8B")
MODELS_LLAMA_2_7B=("Eagle2" "HASS-1" "PRISM")
MODELS_LLAMA_3_8B=("Eagle2" "HASS-1" "HASS-2" "PRISM" "Eagle3")

get_valid_models() {
  local proj="$1"
  case "${proj}" in
    Llama-2-7B) echo "${MODELS_LLAMA_2_7B[@]}" ;;
    Llama-3-8B) echo "${MODELS_LLAMA_3_8B[@]}" ;;
    *)
      echo "[FATAL] Unknown project: ${proj}" >&2
      return 1 ;;
  esac
}

is_valid_combo() {
  local proj="$1" mdl="$2"
  local valid_models
  valid_models=($(get_valid_models "${proj}")) || return 1
  for m in "${valid_models[@]}"; do
    [[ "${m}" == "${mdl}" ]] && return 0
  done
  return 1
}

usage() {
  cat <<'EOF'
Usage:
  bash eval_acceptance_length.sh [options]

Options:
  --gpus <list>          CUDA_VISIBLE_DEVICES, e.g. 0,1,2,3
  --project <name>       PROJECT filter, e.g. Llama-2-7B / Llama-3-8B
                         Default: iterate over all projects
  --model <name>         MODEL filter, e.g. PRISM / Eagle2 / Eagle3 / HASS-1 / HASS-2
                         Default: iterate over all valid models per project
  --run-tag <tag>        RUN_TAG for eval_all output folder
  --benches <list>       Comma-separated benchmarks to run, e.g. mt_bench,humaneval
                         Available: mt_bench, humaneval, gsm8k, alpaca, sum, qa
                         Default: all 6 benchmarks
  -h, --help             Show help

Valid combinations:
  Llama-2-7B : Eagle2, HASS-1, PRISM
  Llama-3-8B : Eagle2, HASS-1, HASS-2, PRISM, Eagle3
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
# Build list of (PROJECT, MODEL) combos to run
# -----------------------------
COMBOS=()

if [[ -n "${PROJECT}" && -n "${MODEL}" ]]; then
  if ! is_valid_combo "${PROJECT}" "${MODEL}"; then
    echo "[FATAL] Invalid combination: PROJECT=${PROJECT}, MODEL=${MODEL}"
    usage
    exit 1
  fi
  COMBOS+=("${PROJECT}|${MODEL}")

elif [[ -n "${PROJECT}" ]]; then
  valid_models=($(get_valid_models "${PROJECT}"))
  for m in "${valid_models[@]}"; do
    COMBOS+=("${PROJECT}|${m}")
  done

elif [[ -n "${MODEL}" ]]; then
  for p in "${ALL_PROJECTS[@]}"; do
    if is_valid_combo "${p}" "${MODEL}"; then
      COMBOS+=("${p}|${MODEL}")
    fi
  done
  if [[ ${#COMBOS[@]} -eq 0 ]]; then
    echo "[FATAL] MODEL=${MODEL} is not valid for any project."
    usage
    exit 1
  fi

else
  for p in "${ALL_PROJECTS[@]}"; do
    valid_models=($(get_valid_models "${p}"))
    for m in "${valid_models[@]}"; do
      COMBOS+=("${p}|${m}")
    done
  done
fi

echo "========================================"
echo "[INFO] Will run ${#COMBOS[@]} combination(s):"
for combo in "${COMBOS[@]}"; do
  IFS='|' read -r _p _m <<< "${combo}"
  echo "  PROJECT=${_p}  MODEL=${_m}"
done
echo "  RUN_TAG=${RUN_TAG}"
echo "  BENCHES=${BENCHES:-<all>}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "========================================"

# -----------------------------
# Core evaluation function
# -----------------------------
run_eval() {
  local CUR_PROJECT="$1"
  local CUR_MODEL="$2"

  # Select codebase by model
  local CUR_MODEL_UPPER
  CUR_MODEL_UPPER="$(echo "${CUR_MODEL}" | tr '[:lower:]' '[:upper:]')"

  local CODEBASE_DIR ENGINE_NAME NEED_ENV_SWITCH=false
  if [[ "${CUR_MODEL_UPPER}" == "PRISM" ]]; then
    CODEBASE_DIR="${PRISM_DIR}"
    ENGINE_NAME="PRISM"
  elif [[ "${CUR_MODEL_UPPER}" == "EAGLE3" ]]; then
    CODEBASE_DIR="${EAGLE3_DIR}"
    ENGINE_NAME="Eagle3"
    NEED_ENV_SWITCH=true
  else
    CODEBASE_DIR="${PRISM_LEGACY_DIR}"
    ENGINE_NAME="PRISM_legacy"
  fi

  local EVAL_SH="${CODEBASE_DIR}/eval_all.sh"

  # Validate paths
  if [[ ! -d "${CODEBASE_DIR}" ]]; then
    echo "[ERROR] ${ENGINE_NAME} directory not found: ${CODEBASE_DIR}"
    return 1
  fi
  if [[ ! -f "${EVAL_SH}" ]]; then
    echo "[ERROR] eval_all.sh not found: ${EVAL_SH}"
    return 1
  fi

  # Switch conda environment if needed
  if [[ "${NEED_ENV_SWITCH}" == true ]]; then
    echo "[INFO] Switching conda env: ${CONDA_ENV_DEFAULT} -> ${CONDA_ENV_EAGLE3}"
    conda activate "${CONDA_ENV_EAGLE3}"
  fi

  # Run evaluation
  echo "----------------------------------------"
  echo "[INFO] Start: PROJECT=${CUR_PROJECT}  MODEL=${CUR_MODEL}"
  echo "  ENGINE=${ENGINE_NAME}"
  echo "  CODEBASE_DIR=${CODEBASE_DIR}"
  echo "  CONDA_ENV=${CONDA_DEFAULT_ENV:-unknown}"

  pushd "${CODEBASE_DIR}" >/dev/null

  local EVAL_RC=0
  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  PROJECT="${CUR_PROJECT}" \
  MODEL="${CUR_MODEL}" \
  RUN_TAG="${RUN_TAG}" \
  BENCHES="${BENCHES}" \
  bash "${EVAL_SH}" "${CUR_PROJECT}" "${CUR_MODEL}" "${BENCHES}"
  EVAL_RC=$?
  set -e

  popd >/dev/null

  # Switch conda environment back if we changed it
  if [[ "${NEED_ENV_SWITCH}" == true ]]; then
    echo "[INFO] Switching conda env back: ${CONDA_ENV_EAGLE3} -> ${CONDA_ENV_DEFAULT}"
    conda activate "${CONDA_ENV_DEFAULT}"
  fi

  # Sync outputs to evaluation/figure4
  local SRC_RESULT="${CODEBASE_DIR}/outputs/${CUR_PROJECT}/${CUR_MODEL}/${RUN_TAG}"
  local DST_BASE="${FIG4_DIR}/outputs/${CUR_PROJECT}/${CUR_MODEL}"
  local DST_RESULT="${DST_BASE}/${RUN_TAG}"

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
    echo "[ERROR] eval_all.sh failed with code ${EVAL_RC} for PROJECT=${CUR_PROJECT} MODEL=${CUR_MODEL}"
  fi

  return "${EVAL_RC}"
}

# -----------------------------
# Iterate over all combos
# -----------------------------
TOTAL=${#COMBOS[@]}
PASSED=0
FAILED=0
FAILED_LIST=()

for idx in "${!COMBOS[@]}"; do
  IFS='|' read -r CUR_PROJECT CUR_MODEL <<< "${COMBOS[$idx]}"
  echo ""
  echo "========================================"
  echo "[PROGRESS] ($((idx + 1))/${TOTAL})  PROJECT=${CUR_PROJECT}  MODEL=${CUR_MODEL}"
  echo "========================================"

  if run_eval "${CUR_PROJECT}" "${CUR_MODEL}"; then
    PASSED=$((PASSED + 1))
  else
    FAILED=$((FAILED + 1))
    FAILED_LIST+=("${CUR_PROJECT}|${CUR_MODEL}")
  fi
done

# -----------------------------
# Summary
# -----------------------------
echo ""
echo "========================================"
echo "[SUMMARY] Total=${TOTAL}  Passed=${PASSED}  Failed=${FAILED}"
if [[ ${FAILED} -gt 0 ]]; then
  echo "[SUMMARY] Failed combinations:"
  for f in "${FAILED_LIST[@]}"; do
    IFS='|' read -r _p _m <<< "${f}"
    echo "  PROJECT=${_p}  MODEL=${_m}"
  done
fi
echo "========================================"

if [[ ${FAILED} -gt 0 ]]; then
  exit 1
fi

echo "[FINAL] All done."
