#!/usr/bin/env bash
# Run Hugging Face Transformers benchmark workloads.

set -euo pipefail

WORKLOAD="${1:-gpt_j_6b}"

JAX_LAB_DIR="${PWD}"
PYTHON="${PYTHON:-python3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

TARGET="hf_transformers_release"
TARGET_DIR="${JAX_LAB_DIR}/targets/${TARGET}"
RUN_DIR="${TARGET_DIR}/run_artifacts/${WORKLOAD}"

mkdir -p "${RUN_DIR}"

source "${JAX_LAB_DIR}/utilities/install_jax_wheels.sh"

for file in \
  "${TARGET_DIR}/benchspec.yml" \
  "${TARGET_DIR}/partitions.py" \
  "${TARGET_DIR}/run_clm_mp.py" \
  "${TARGET_DIR}/run_t5_mlm_flax.py"; do
  [[ -f "${file}" ]] || {
    echo "missing required file: ${file}" >&2
    exit 2
  }
done

"${PYTHON}" -m pip install \
  "transformers<5" \
  datasets \
  ml_dtypes \
  opt_einsum \
  optax \
  flax

FLAX_T5="$("${PYTHON}" -c "import transformers.models.t5.modeling_flax_t5 as m; print(m.__file__)")"
sed -i 's/a_max=/max=/g; s/a_min=/min=/g' "${FLAX_T5}"

export PY_COLORS=1
export TF_CPP_MIN_LOG_LEVEL=0
export JAX_ENABLE_X64=0
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_PREALLOCATE=false

MODEL_RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

set +e

case "${WORKLOAD}" in
  gpt_j_6b)
    BENCHMARK_COMMAND="${PYTHON} ${TARGET_DIR}/run_clm_mp.py \
      --model_name_or_path EleutherAI/gpt-j-6b \
      --dataset_name Salesforce/wikitext \
      --dataset_config_name wikitext-103-raw-v1 \
      --do_train \
      --do_eval \
      --block_size 512 \
      --num_train_epochs 0 \
      --per_device_eval_batch_size 1 \
      --overwrite_output_dir \
      --output_dir /tmp/flax-clm \
      --dtype float16 \
      --use_train_set_for_eval \
      --batch_num 13"
    ;;

  flan_t5_large)
    mkdir -p /tmp/flan-t5-large_eval

    BENCHMARK_COMMAND="${PYTHON} ${TARGET_DIR}/run_t5_mlm_flax.py \
      --dataset_name Salesforce/wikitext \
      --dataset_config_name wikitext-103-raw-v1 \
      --model_name_or_path google/flan-t5-large \
      --output_dir /tmp/flan-t5-large_eval \
      --do_eval \
      --num_train_epochs 0 \
      --logging_steps 1 \
      --overwrite_output_dir \
      --per_device_eval_batch_size 1 \
      --max_seq_length 512 \
      --dtype bfloat16 \
      --use_train_set_for_eval \
      --batch_num 13"
    ;;

  *)
    echo "unknown workload: ${WORKLOAD}" >&2
    exit 1
    ;;
esac

${BENCHMARK_COMMAND} 2>&1 | tee /tmp/hf_transformers_run.log
RUN_CODE=${PIPESTATUS[0]}

set -e

MODEL_RUN_COMPLETED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

EVAL_TIME_SECONDS="$(
  grep "eval_times" /tmp/hf_transformers_run.log \
    | tail -1 \
    | cut -d "," -f 2 \
    | xargs || true
)"

if [[ -z "${EVAL_TIME_SECONDS}" ]]; then
  echo "failed to parse eval_time_seconds" >&2
  RUN_CODE=1
  EVAL_TIME_SECONDS=0
fi

echo "benchmark_metric eval_time_seconds=${EVAL_TIME_SECONDS}" \
  | tee -a /tmp/hf_transformers_run.log

CMP_CODE=0

"${PYTHON}" "${JAX_LAB_DIR}/utilities/benchcmp.py" \
  --benchspec "${TARGET_DIR}/benchspec.yml" \
  --workload "${WORKLOAD}" \
  --log /tmp/hf_transformers_run.log \
  > /tmp/benchmark_results.json || CMP_CODE=$?

"${PYTHON}" "${JAX_LAB_DIR}/utilities/collect_bench_manifest.py" \
  --combo "${COMBO:?}" \
  --target "${TARGET}" \
  --workload "${WORKLOAD}" \
  --run-code "${RUN_CODE}" \
  --model-run-started-at "${MODEL_RUN_STARTED_AT}" \
  --model-run-completed-at "${MODEL_RUN_COMPLETED_AT}" \
  --benchmark-repo-commit "package" \
  --benchmark-command "${BENCHMARK_COMMAND}" \
  --config benchspec="${TARGET_DIR}/benchspec.yml" \
  > /tmp/benchmark_manifest.json

"${PYTHON}" "${JAX_LAB_DIR}/utilities/collect_run_manifest.py" \
  --runner "${RUNNER:?}" \
  --python-version "${PYTHON_VERSION}" \
  --rocm-version "${ROCM_VERSION:?}" \
  --rocm-tag "${ROCM_TAG:?}" \
  --cmp-code "${CMP_CODE}" \
  --extra /tmp/benchmark_manifest.json \
  --extra /tmp/benchmark_results.json \
  --out "${RUN_DIR}/result.json"

rm -f \
  /tmp/hf_transformers_run.log \
  /tmp/benchmark_manifest.json \
  /tmp/benchmark_results.json

exit $(( RUN_CODE != 0 || CMP_CODE != 0 ))
