#!/usr/bin/env bash
#SBATCH -J PRISM
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --gres=gpu:2

set -u
set -o pipefail

# =========================
# Config (overridable)
# =========================
# Optional args:
#   sbatch run_eval.sbatch <PROJECT> <MODEL> [BENCHES]
#   BENCHES is a comma-separated list, e.g. "mt_bench,humaneval,gsm8k"
PROJECT="${1:-${PROJECT:-Llama-3-8B}}"
MODEL="${2:-${MODEL:-Eagle3}}"
BENCHES_ARG="${3:-${BENCHES:-}}"

REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MODEL_ROOT="$(cd "${REPO_ROOT}/.." && pwd)/models"

# Keep PROJECT for drafter/config layout
CKPT_ROOT="${MODEL_ROOT}/${PROJECT}-drafter"
EA_CONFIG_PATH="${REPO_ROOT}/train/${PROJECT}/${MODEL}_config.json"

# Resolve base model path by project alias
# Priority:
# 1) BASE_MODEL_PATH env override
# 2) Auto mapping by PROJECT
if [[ -n "${BASE_MODEL_PATH:-}" ]]; then
  BASE_MODEL_PATH="${BASE_MODEL_PATH}"
else
  PROJECT_KEY="$(echo "${PROJECT}" | tr '[:upper:]' '[:lower:]')"
  case "${PROJECT_KEY}" in
    llama-2-7b|llama-2-7b-instruct|llama-2-7b-chat-hf)
      BASE_MODEL_PATH="${MODEL_ROOT}/Llama-2-7b-chat-hf"
      ;;
    llama-3-8b|llama-3-8b-instruct)
      BASE_MODEL_PATH="${MODEL_ROOT}/Llama-3-8B-Instruct"
      ;;
    *)
      BASE_MODEL_PATH="${MODEL_ROOT}/${PROJECT}-Instruct"
      ;;
  esac
fi

# =========================
# Resolve generation script by PROJECT + MODEL
# =========================
# Rule:
#   MODEL == Eagle3 (only with Llama-3-8B) => gen_ea_answer_llama3chat.py
#   PROJECT ~ Llama-3-*                     => gen_ea_answer_llama3chat.py
#   PROJECT ~ Llama-2-*                     => gen_ea_answer_llama2chat.py
#   fallback                                => gen_ea_answer_llama3chat.py
# =========================
MODEL_KEY="$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]')"
PROJECT_KEY="$(echo "${PROJECT}" | tr '[:upper:]' '[:lower:]')"

if [[ "${MODEL_KEY}" == "eagle3" ]]; then
  GEN_SCRIPT="${REPO_ROOT}/evaluation/gen_ea_answer_llama3chat.py"
else
  case "${PROJECT_KEY}" in
    llama-2-7b|llama-2-7b-instruct|llama-2-7b-chat-hf)
      GEN_SCRIPT="${REPO_ROOT}/evaluation/gen_ea_answer_llama2chat.py"
      ;;
    llama-3-8b|llama-3-8b-instruct)
      GEN_SCRIPT="${REPO_ROOT}/evaluation/gen_ea_answer_llama3chat.py"
      ;;
    *)
      GEN_SCRIPT="${REPO_ROOT}/evaluation/gen_ea_answer_llama3chat.py"
      echo "[WARN] Unknown PROJECT '${PROJECT}', defaulting to gen_ea_answer_llama3chat.py"
      ;;
  esac
fi

# =========================
# Resolve benchmarks
# =========================
ALL_BENCHES=("mt_bench" "humaneval" "gsm8k" "alpaca" "sum" "qa")

if [[ -n "${BENCHES_ARG}" ]]; then
  IFS=',' read -r -a BENCHES <<< "${BENCHES_ARG}"
  # Validate user-specified benchmarks
  for b in "${BENCHES[@]}"; do
    found=0
    for ab in "${ALL_BENCHES[@]}"; do
      if [[ "${b}" == "${ab}" ]]; then
        found=1
        break
      fi
    done
    if [[ "${found}" -eq 0 ]]; then
      echo "[FATAL] Unknown benchmark '${b}'. Available: ${ALL_BENCHES[*]}"
      exit 1
    fi
  done
else
  BENCHES=("${ALL_BENCHES[@]}")
fi

# =========================
# Output layout (simplified)
# =========================
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${REPO_ROOT}/outputs/${PROJECT}/${MODEL}/${RUN_TAG}"
LOG_DIR="${RESULT_ROOT}/logs"

mkdir -p "${RESULT_ROOT}" "${LOG_DIR}" "${REPO_ROOT}/outputs/${PROJECT}/${MODEL}"
ln -sfn "${RESULT_ROOT}" "${REPO_ROOT}/outputs/${PROJECT}/${MODEL}/latest"

# Make benchmark folders land under RESULT_ROOT
cd "${RESULT_ROOT}"

# =========================
# Tasks
# =========================
SUFFIXES=("100k" "200k" "400k" "600k" "800k")
TEMPS=("0.0" "1.0")

# =========================
# GPU parse
# =========================
GPU_STR="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a GPUS <<< "${GPU_STR}"
NUM_GPUS="${#GPUS[@]}"

if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "[FATAL] No GPU found from CUDA_VISIBLE_DEVICES='${GPU_STR}'"
  exit 1
fi

if [[ ! -d "${CKPT_ROOT}" ]]; then
  echo "[FATAL] CKPT_ROOT not found: ${CKPT_ROOT}"
  exit 1
fi

if [[ ! -f "${EA_CONFIG_PATH}" ]]; then
  echo "[FATAL] EA_CONFIG_PATH not found: ${EA_CONFIG_PATH}"
  exit 1
fi

if [[ ! -d "${BASE_MODEL_PATH}" ]]; then
  echo "[FATAL] BASE_MODEL_PATH not found: ${BASE_MODEL_PATH}"
  exit 1
fi

if [[ ! -f "${GEN_SCRIPT}" ]]; then
  echo "[FATAL] GEN_SCRIPT not found: ${GEN_SCRIPT}"
  exit 1
fi

TMP_QUEUE_DIR="$(mktemp -d -p "${RESULT_ROOT}" "gpu_queues_${SLURM_JOB_ID:-$$}_XXXX")"
cleanup() { rm -rf "${TMP_QUEUE_DIR}"; }
trap cleanup EXIT

echo "=================================================="
echo "start time: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "LOG_DIR=${LOG_DIR}"
echo "PROJECT=${PROJECT}"
echo "MODEL=${MODEL}"
echo "GEN_SCRIPT=${GEN_SCRIPT}"
echo "BENCHES=${BENCHES[*]}"
echo "CUDA_VISIBLE_DEVICES(raw)=${GPU_STR}"
echo "Parsed GPUs: ${GPUS[*]}"
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "CKPT_ROOT=${CKPT_ROOT}"
echo "EA_CONFIG_PATH=${EA_CONFIG_PATH}"
echo "BASE_MODEL_PATH=${BASE_MODEL_PATH}"
echo "=================================================="

# =========================
# Helper
# =========================
get_ea_model_path() {
  local name="$1"
  # IMPORTANT: CKPT_ROOT already includes ${PROJECT}
  echo "${CKPT_ROOT}/${name}"
}

# =========================
# Copy config to each checkpoint dir
# =========================
for sfx in "${SUFFIXES[@]}"; do
  NAME="${MODEL}-${sfx}"
  EA_MODEL_PATH="$(get_ea_model_path "${NAME}")"

  if [[ ! -d "${EA_MODEL_PATH}" ]]; then
    echo "[WARN] Missing model dir, skip config copy: ${EA_MODEL_PATH}"
    continue
  fi

  cp -f "${EA_CONFIG_PATH}" "${EA_MODEL_PATH}/config.json"
done

# =========================
# Build per-GPU queues (strict serial per GPU)
# =========================
for ((i=0; i<NUM_GPUS; i++)); do
  : > "${TMP_QUEUE_DIR}/gpu_${i}.queue"
done

job_idx=0
for sfx in "${SUFFIXES[@]}"; do
  NAME="${MODEL}-${sfx}"
  for bench_name in "${BENCHES[@]}"; do
    for temperature in "${TEMPS[@]}"; do
      gpu_idx=$((job_idx % NUM_GPUS))
      printf "%s\t%s\t%s\n" "${NAME}" "${bench_name}" "${temperature}" >> "${TMP_QUEUE_DIR}/gpu_${gpu_idx}.queue"
      echo "[DISPATCH] job=${job_idx} gpu=${GPUS[$gpu_idx]} name=${NAME} bench=${bench_name} temp=${temperature}"
      job_idx=$((job_idx + 1))
    done
  done
done
echo "[INFO] Total jobs: ${job_idx}"

# =========================
# Single task runner
# =========================
run_one_task() {
  local gpu="$1"
  local name="$2"
  local bench_name="$3"
  local temperature="$4"

  local ea_model_path
  ea_model_path="$(get_ea_model_path "${name}")"

  local log_file="${LOG_DIR}/${name}__${bench_name}__temp-${temperature}__gpu-${gpu}.log"
  local ld_json="${bench_name}/${name}-temperature-${temperature}.jsonl"

  # Baseline compatible locations
  local baseline_json_new="${bench_name}/Naive-temperature-${temperature}.jsonl"
  local baseline_json_old="${bench_name}/${PROJECT}/Naive-temperature-${temperature}.jsonl"
  local baseline_json=""

  local task_fail=0

  {
    echo "--------------------------------------------------"
    echo "start: $(date)"
    echo "gpu=${gpu}, name=${name}, bench=${bench_name}, temp=${temperature}"
    echo "EA_MODEL_PATH=${ea_model_path}"
    echo "GEN_SCRIPT=${GEN_SCRIPT}"
    echo "CWD=${PWD}"

    if [[ ! -d "${ea_model_path}" ]]; then
      echo "[ERROR] Missing model dir: ${ea_model_path}"
      task_fail=1
    fi

    if [[ "${task_fail}" -eq 0 ]]; then
      if ! CUDA_VISIBLE_DEVICES="${gpu}" python "${GEN_SCRIPT}" \
        --ea-model-path "${ea_model_path}" \
        --base-model-path "${BASE_MODEL_PATH}" \
        --model-id "${name}" \
        --bench-name "${bench_name}" \
        --total-token 60 \
        --depth 5 \
        --top-k 10 \
        --temperature "${temperature}" \
	--use_eagle3
      then
        echo "[ERROR] ${GEN_SCRIPT##*/} failed"
        task_fail=1
      fi
    fi

    # Optional baseline generation (disabled by default to avoid overwrite race)
    # if [[ "${task_fail}" -eq 0 ]]; then
    #   if ! CUDA_VISIBLE_DEVICES="${gpu}" python "${REPO_ROOT}/evaluation/gen_baseline_answer_llama3chat.py" \
    #     --ea-model-path "${ea_model_path}" \
    #     --base-model-path "${BASE_MODEL_PATH}" \
    #     --model-id "Naive" \
    #     --bench-name "${bench_name}" \
    #     --temperature "${temperature}"
    #   then
    #     echo "[ERROR] gen_baseline_answer_llama3chat.py failed"
    #     task_fail=1
    #   fi
    # fi

    if [[ "${task_fail}" -eq 0 ]]; then
      if [[ ! -f "${ld_json}" ]]; then
        echo "[ERROR] LD output not found: ${ld_json}"
        task_fail=1
      fi
    fi

    if [[ "${task_fail}" -eq 0 ]]; then
      if [[ -f "${baseline_json_new}" ]]; then
        baseline_json="${baseline_json_new}"
      elif [[ -f "${baseline_json_old}" ]]; then
        baseline_json="${baseline_json_old}"
      else
        baseline_json=""
      fi

      if [[ -n "${baseline_json}" ]]; then
        if ! python "${REPO_ROOT}/evaluation/summary.py" \
          --model_path "${BASE_MODEL_PATH}" \
          --baseline_json "${baseline_json}" \
          --LD_json "${ld_json}"
        then
          echo "[ERROR] summary.py failed"
          task_fail=1
        fi
      else
        echo "[WARN] Baseline file not found, skip summary"
      fi
    fi

    if [[ "${task_fail}" -eq 0 ]]; then
      if ! python "${REPO_ROOT}/evaluation/conditional_accept_rate.py" \
        --input_file "${ld_json}"
      then
        echo "[ERROR] conditional_accept_rate.py failed"
        task_fail=1
      fi
    fi

    if [[ "${task_fail}" -eq 0 ]]; then
      echo "done: $(date)"
    else
      echo "failed: $(date)"
    fi
  } > "${log_file}" 2>&1

  return "${task_fail}"
}

# =========================
# One worker per GPU (strict serial on same GPU)
# =========================
gpu_worker() {
  local gpu_idx="$1"
  local gpu="${GPUS[$gpu_idx]}"
  local queue_file="${TMP_QUEUE_DIR}/gpu_${gpu_idx}.queue"
  local worker_fail=0

  echo "[WORKER-${gpu}] start queue: ${queue_file}"

  while IFS=$'\t' read -r name bench_name temperature; do
    [[ -z "${name}" ]] && continue
    echo "[WORKER-${gpu}] run name=${name} bench=${bench_name} temp=${temperature}"

    if ! run_one_task "${gpu}" "${name}" "${bench_name}" "${temperature}"; then
      worker_fail=1
      echo "[WORKER-${gpu}] task failed: name=${name} bench=${bench_name} temp=${temperature}"
    fi
  done < "${queue_file}"

  echo "[WORKER-${gpu}] finished"
  return "${worker_fail}"
}

# Launch workers
pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  gpu_worker "${i}" &
  pids+=("$!")
done

# Wait all workers
any_fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    any_fail=1
  fi
done

echo "end time: $(date)"
echo "RESULT_ROOT=${RESULT_ROOT}"

if [[ "${any_fail}" -eq 1 ]]; then
  echo "[FINAL] Some tasks failed. Check logs in: ${LOG_DIR}"
  exit 1
else
  echo "[FINAL] All tasks finished successfully. Logs in: ${LOG_DIR}"
fi
