#!/usr/bin/env bash
# Run Flax release benchmark workload.

set -euo pipefail

WORKLOAD="${1:-convolution}"

JAX_LAB_DIR="${PWD}"
PYTHON="${PYTHON:-python3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

TARGET="flax_release"
TARGET_DIR="${JAX_LAB_DIR}/targets/${TARGET}"
RUN_DIR="${TARGET_DIR}/run_artifacts/${WORKLOAD}"

mkdir -p "${RUN_DIR}"

source "${JAX_LAB_DIR}/utilities/install_jax_wheels.sh"

if [[ ! -d "${TARGET_DIR}/flax/.git" ]]; then
  git clone \
    --depth 1 \
    --branch main \
    https://github.com/google/flax.git \
    "${TARGET_DIR}/flax"
fi

[[ -f "${TARGET_DIR}/benchspec.yml" ]] || {
  echo "missing required file: ${TARGET_DIR}/benchspec.yml" >&2
  exit 2
}

python3 -m pip install -q uv

REQ_FILE="${TARGET_DIR}/base-requirements.txt"

(
  cd "${TARGET_DIR}/flax"
  uv tree --depth 1 \
    | sed 's/[├└──]//g' \
    | awk '{print $1}' \
    | grep -viE '^(flax|jax|torch)\b' \
    | grep -viF '(*)'
) > "${REQ_FILE}"

grep -vi '^tensorflow' "${REQ_FILE}" > /tmp/base.txt
{
  echo "tensorflow==2.19.1"
  echo "tensorflow-datasets"
  echo "importlib_resources"
} >> /tmp/base.txt
mv /tmp/base.txt "${REQ_FILE}"

pip3 install -r "${REQ_FILE}"

export PY_COLORS=1
export TF_CPP_MIN_LOG_LEVEL=0
export JAX_ENABLE_X64=0
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_PREALLOCATE=false

MODEL_RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

set +e
pushd "${TARGET_DIR}/flax" >/dev/null

case "${WORKLOAD}" in
  convolution)
    sed -i \
      's/^[[:space:]]*export_mgr\.save.*/  pass  # export_mgr.save skipped by jax-lab CI/' \
      examples/mnist/train.py

    cd examples/mnist
    BENCHMARK_COMMAND="${PYTHON} main.py --workdir=/tmp/mnist --config=configs/default.py"

    ${BENCHMARK_COMMAND} 2>&1 | tee /tmp/flax_run.log
    RUN_CODE=${PIPESTATUS[0]}
    LOSS="$(grep "test_loss" /tmp/flax_run.log | tail -1 | awk '{print $12}')"
    ;;

  resnet)
    cp -f "${TARGET_DIR}/imagenette.py" examples/imagenet/configs/imagenette.py

    sed -i \
      '/dataset_builder = tfds.builder(config.dataset)/a\  dataset_builder.download_and_prepare()' \
      examples/imagenet/train.py

    sed -i \
      's/if jax.process_index() == 0 and config.profile:/if jax.process_index() == 0 and hasattr(config, "profile") and config.profile:/' \
      examples/imagenet/train.py

    cd examples/imagenet
    BENCHMARK_COMMAND="${PYTHON} main.py --workdir=/tmp/imagenette --config=configs/imagenette.py"

    ${BENCHMARK_COMMAND} 2>&1 | tee /tmp/flax_run.log
    RUN_CODE=${PIPESTATUS[0]}
    LOSS="$(grep "eval epoch" /tmp/flax_run.log | tail -1 | awk '{print $9}')"
    ;;

  nlp_seq)
    cd examples/nlp_seq

    wget -O ud-treebanks-v2.0.tgz \
      https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/e3706e21-06e1-42de-88d4-93b12d5bcff1/content

    tar xzf ud-treebanks-v2.0.tgz

    BENCHMARK_COMMAND="${PYTHON} train.py --batch_size=64 --num_train_steps=18000 --eval_frequency=500 --model_dir=/tmp/model_dir --dev=ud-treebanks-v2.0/UD_Ancient_Greek/grc-ud-dev.conllu --train=ud-treebanks-v2.0/UD_Ancient_Greek/grc-ud-train.conllu"

    ${BENCHMARK_COMMAND} 2>&1 | tee /tmp/flax_run.log
    RUN_CODE=${PIPESTATUS[0]}
    LOSS="$(grep "eval in step:" /tmp/flax_run.log | tail -1 | awk '{print $10}')"
    ;;

  *)
    echo "unknown workload: ${WORKLOAD}" >&2
    exit 1
    ;;
esac

popd >/dev/null
set -e

MODEL_RUN_COMPLETED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
BENCHMARK_REPO_COMMIT="$(git -C "${TARGET_DIR}/flax" rev-parse HEAD)"

echo "benchmark_metric final_loss=${LOSS}" | tee -a /tmp/flax_run.log

CMP_CODE=0
"${PYTHON}" "${JAX_LAB_DIR}/utilities/benchcmp.py" \
  --benchspec "${TARGET_DIR}/benchspec.yml" \
  --workload "${WORKLOAD}" \
  --log /tmp/flax_run.log \
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

rm -f /tmp/flax_run.log /tmp/benchmark_manifest.json /tmp/benchmark_results.json

exit $(( RUN_CODE != 0 || CMP_CODE != 0 ))
