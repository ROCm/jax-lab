#!/usr/bin/env bash
# Run MaxText release benchmark workload.

set -euo pipefail

WORKLOAD="${1:-gemma3_4b}"

JAX_LAB_DIR="${PWD}"
PYTHON="${PYTHON:-python3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

TARGET="maxtext_release"
TARGET_DIR="${JAX_LAB_DIR}/targets/${TARGET}"
RUN_DIR="${TARGET_DIR}/run_artifacts/${WORKLOAD}"

mkdir -p "${RUN_DIR}"

source "${JAX_LAB_DIR}/utilities/install_jax_wheels.sh"

if [[ ! -d "${TARGET_DIR}/maxtext/.git" ]]; then
  git clone \
    --depth 1 \
    --branch rocm-main \
    https://github.com/ROCm/maxtext.git \
    "${TARGET_DIR}/maxtext"
fi

for file in \
  "${TARGET_DIR}/requirements.txt" \
  "${TARGET_DIR}/configs/${WORKLOAD}.yml" \
  "${TARGET_DIR}/benchspec.yml"; do
  [[ -f "${file}" ]] || {
    echo "missing required file: ${file}" >&2
    exit 2
  }
done

"${PYTHON}" -m pip install -r "${TARGET_DIR}/requirements.txt"

export PY_COLORS=1
export TF_CPP_MIN_LOG_LEVEL=0
export JAX_ENABLE_X64=0
export XLA_PYTHON_CLIENT_ALLOCATOR=bfc
export XLA_PYTHON_CLIENT_PREALLOCATE=false

MODEL_RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
BENCHMARK_COMMAND="${PYTHON} -m maxtext.trainers.pre_train.train ${TARGET_DIR}/configs/${WORKLOAD}.yml"

set +e
pushd "${TARGET_DIR}/maxtext/src" >/dev/null

${BENCHMARK_COMMAND} 2>&1 | tee /tmp/maxtext_run.log
RUN_CODE=${PIPESTATUS[0]}

popd >/dev/null
set -e

MODEL_RUN_COMPLETED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
BENCHMARK_REPO_COMMIT="$(git -C "${TARGET_DIR}/maxtext" rev-parse HEAD)"

CMP_CODE=0
"${PYTHON}" "${JAX_LAB_DIR}/utilities/benchcmp.py" \
  --benchspec "${TARGET_DIR}/benchspec.yml" \
  --workload "${WORKLOAD}" \
  --log /tmp/maxtext_run.log \
  > /tmp/benchmark_results.json || CMP_CODE=$?

"${PYTHON}" "${JAX_LAB_DIR}/utilities/collect_bench_manifest.py" \
  --combo "${COMBO:?}" \
  --target "${TARGET}" \
  --workload "${WORKLOAD}" \
  --run-code "${RUN_CODE}" \
  --model-run-started-at "${MODEL_RUN_STARTED_AT}" \
  --model-run-completed-at "${MODEL_RUN_COMPLETED_AT}" \
  --benchmark-repo-commit "${BENCHMARK_REPO_COMMIT}" \
  --benchmark-command "${BENCHMARK_COMMAND}" \
  --config workload="${TARGET_DIR}/configs/${WORKLOAD}.yml" \
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

rm -f /tmp/maxtext_run.log /tmp/benchmark_manifest.json /tmp/benchmark_results.json

exit $(( RUN_CODE != 0 || CMP_CODE != 0 ))
