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
BENCHMARK_LOG="/tmp/${TARGET}-${WORKLOAD}.log"

mkdir -p "${RUN_DIR}"
rm -f "${RUN_DIR}/result.json"

source "${JAX_LAB_DIR}/utilities/benchmark_logging.sh"

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
  echo "tensorflow-datasets==4.9.10"
  echo "tensorflow-metadata==1.17.3"
  echo "importlib_resources"
} >> /tmp/base.txt
mv /tmp/base.txt "${REQ_FILE}"

pip3 install -r "${REQ_FILE}"

if [[ -n "${REFERENCE_JAXLIB_VERSION:-}" ]]; then
  case "${WORKLOAD}" in
    convolution) dataset=mnist ;;
    resnet) dataset=imagenette ;;
  esac
  if [[ -n "${dataset:-}" ]]; then
    python3 -c 'import sys, tensorflow_datasets as tfds; tfds.builder(sys.argv[1]).download_and_prepare()' "${dataset}"
  fi
fi

export PY_COLORS=1
export TF_CPP_MIN_LOG_LEVEL=0
export JAX_ENABLE_X64=0
export XLA_PYTHON_CLIENT_ALLOCATOR=bfc
export XLA_PYTHON_CLIENT_PREALLOCATE=false

pushd "${TARGET_DIR}/flax" >/dev/null

case "${WORKLOAD}" in
  convolution)
    sed -i \
      's/^[[:space:]]*export_mgr\.save.*/  pass  # export_mgr.save skipped by jax-lab CI/' \
      examples/mnist/train.py

    cd examples/mnist
    BASELINE_OUTPUT_FLAG=--workdir
    BENCHMARK_CMD=(
      "${PYTHON}" main.py
      --workdir=/tmp/mnist
      --config=configs/default.py
    )
    ;;

  resnet)
    cp -f "${TARGET_DIR}/imagenette.py" examples/imagenet/configs/imagenette.py

    sed -i '/^  dataset_builder.download_and_prepare()$/d' examples/imagenet/train.py
    if [[ -z "${REFERENCE_JAXLIB_VERSION:-}" ]]; then
      sed -i \
        '/dataset_builder = tfds.builder(config.dataset)/a\  dataset_builder.download_and_prepare()' \
        examples/imagenet/train.py
    fi

    sed -i \
      's/if jax.process_index() == 0 and config.profile:/if jax.process_index() == 0 and hasattr(config, "profile") and config.profile:/' \
      examples/imagenet/train.py

    cd examples/imagenet
    BASELINE_OUTPUT_FLAG=--workdir
    BENCHMARK_CMD=(
      "${PYTHON}" main.py
      --workdir=/tmp/imagenette
      --config=configs/imagenette.py
    )
    ;;

  nlp_seq)
    cd examples/nlp_seq

    wget -O ud-treebanks-v2.0.tgz \
      https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/e3706e21-06e1-42de-88d4-93b12d5bcff1/content

    tar xzf ud-treebanks-v2.0.tgz

    BASELINE_OUTPUT_FLAG=--model_dir
    BENCHMARK_CMD=(
      "${PYTHON}" train.py
      --batch_size=64
      --num_train_steps=18000
      --eval_frequency=500
      --model_dir=/tmp/model_dir
      --dev=ud-treebanks-v2.0/UD_Ancient_Greek/grc-ud-dev.conllu
      --train=ud-treebanks-v2.0/UD_Ancient_Greek/grc-ud-train.conllu
    )
    ;;

  *)
    echo "unknown workload: ${WORKLOAD}" >&2
    exit 1
    ;;
esac

append_metric_from_log() {
  local log_file="$1" loss
  case "${WORKLOAD}" in
    convolution)
      loss="$(grep 'test_loss' "${log_file}" | tail -1 | awk '{gsub(/,/, "", $12); print $12}' || true)"
      ;;
    resnet)
      loss="$(grep 'eval epoch' "${log_file}" | tail -1 | awk '{print $9}' || true)"
      ;;
    nlp_seq)
      loss="$(grep 'eval in step:' "${log_file}" | tail -1 | awk '{print $10}' || true)"
      ;;
  esac
  if [[ -z "${loss}" ]]; then
    echo "failed to parse final_loss" >&2
    show_log_tail "${log_file}"
    return 1
  fi
  echo "benchmark_metric final_loss=${loss}" >> "${log_file}"
}

run_with_baseline \
  "${BENCHMARK_LOG}" \
  "${TARGET}/${WORKLOAD}" \
  "${TARGET_DIR}/benchspec.yml" \
  "${WORKLOAD}" \
  final_loss \
  "${BENCHMARK_CMD[@]}"

popd >/dev/null
BENCHMARK_REPO_COMMIT="$(git -C "${TARGET_DIR}/flax" rev-parse HEAD)"

CMP_CODE=0
"${PYTHON}" "${JAX_LAB_DIR}/utilities/benchcmp.py" \
  --benchspec "${TARGET_DIR}/benchspec.yml" \
  --workload "${WORKLOAD}" \
  --log "${BENCHMARK_LOG}" \
  "${BENCHCMP_ARGS[@]}" \
  > /tmp/benchmark_results.json || {
  CMP_CODE=$?
  show_log_tail "${BENCHMARK_LOG}"
}

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

rm -f "${BENCHMARK_LOG}" /tmp/benchmark_manifest.json /tmp/benchmark_results.json

exit $(( RUN_CODE != 0 || CMP_CODE != 0 ))
