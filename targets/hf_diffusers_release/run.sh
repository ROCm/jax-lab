#!/usr/bin/env bash
# Run Hugging Face Diffusers inference benchmark workload.

set -euo pipefail

WORKLOAD="${1:-stable_diffusion}"

JAX_LAB_DIR="${PWD}"
PYTHON="${PYTHON:-python3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

TARGET="hf_diffusers_release"
TARGET_DIR="${JAX_LAB_DIR}/targets/${TARGET}"
RUN_DIR="${TARGET_DIR}/run_artifacts/${WORKLOAD}"

mkdir -p "${RUN_DIR}"

source "${JAX_LAB_DIR}/utilities/install_jax_wheels.sh"

for file in \
  "${TARGET_DIR}/stable_diffusion.py" \
  "${TARGET_DIR}/benchspec.yml"; do
  [[ -f "${file}" ]] || {
    echo "missing required file: ${file}" >&2
    exit 2
  }
done

case "${WORKLOAD}" in
  stable_diffusion)
    ;;
  *)
    echo "unknown workload: ${WORKLOAD}" >&2
    exit 1
    ;;
esac

"${PYTHON}" -m pip install \
  "transformers<5" \
  "diffusers<1.0" \
  ml_dtypes \
  opt_einsum \
  flax

export PY_COLORS=1
export TF_CPP_MIN_LOG_LEVEL=0
export JAX_ENABLE_X64=0
export XLA_PYTHON_CLIENT_ALLOCATOR=bfc
export XLA_PYTHON_CLIENT_PREALLOCATE=false

MODEL_RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
BENCHMARK_COMMAND="${PYTHON} ${TARGET_DIR}/stable_diffusion.py"

set +e
${BENCHMARK_COMMAND} 2>&1 | tee /tmp/stable_diff_run.log
RUN_CODE=${PIPESTATUS[0]}
set -e

AVG_INFERENCE_TIME_EXCLUDE_FIRST="$(
  grep "Average Inference time(exclude first)" /tmp/stable_diff_run.log \
    | tail -1 \
    | awk '{print $(NF-1)}'
)"

if [[ -z "${AVG_INFERENCE_TIME_EXCLUDE_FIRST}" ]]; then
  echo "failed to parse avg_inference_time_exclude_first" >&2
  RUN_CODE=1
fi

echo "benchmark_metric avg_inference_time_exclude_first=${AVG_INFERENCE_TIME_EXCLUDE_FIRST}" \
  | tee -a /tmp/stable_diff_run.log

MODEL_RUN_COMPLETED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

CMP_CODE=0
"${PYTHON}" "${JAX_LAB_DIR}/utilities/benchcmp.py" \
  --benchspec "${TARGET_DIR}/benchspec.yml" \
  --workload "${WORKLOAD}" \
  --log /tmp/stable_diff_run.log \
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
  /tmp/stable_diff_run.log \
  /tmp/benchmark_manifest.json \
  /tmp/benchmark_results.json

exit $(( RUN_CODE != 0 || CMP_CODE != 0 ))
