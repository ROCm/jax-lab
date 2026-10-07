#!/usr/bin/env bash
# Shared reference/candidate execution after a target prepares its command.

_install_benchmark_environment() {
  local selection="$1"
  (cd "${JAX_LAB_DIR}" && JAXLIB_VERSION="${selection}" \
    bash utilities/install_jax_wheels.sh) || return $?
  install_benchmark_requirements
}

_benchmark_once() {
  local log="$1" label="$2" output_dir="$3"
  shift 3
  local -a command=("$@")
  local i found=0
  mkdir -p "${output_dir}" || return $?
  if [[ -n "${BASELINE_OUTPUT_FLAG:-}" ]]; then
    for i in "${!command[@]}"; do
      if [[ "${command[i]}" == "${BASELINE_OUTPUT_FLAG}="* ]]; then
        command[i]="${BASELINE_OUTPUT_FLAG}=${output_dir}"
        found=1
        break
      elif [[ "${command[i]}" == "${BASELINE_OUTPUT_FLAG}" && $((i + 1)) -lt ${#command[@]} ]]; then
        command[i+1]="${output_dir}"
        found=1
        break
      fi
    done
    if [[ "${found}" != 1 ]]; then
      echo "missing benchmark output flag: ${BASELINE_OUTPUT_FLAG}" >&2
      return 2
    fi
  fi
  benchmark_command_string "${command[@]}"
  MODEL_RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  RUN_CODE=0
  run_with_log "${log}" "${label}" "${command[@]}" || RUN_CODE=$?
  if [[ "${RUN_CODE}" == 0 ]] && declare -F append_metric_from_log >/dev/null; then
    append_metric_from_log "${log}" || RUN_CODE=$?
  fi
  MODEL_RUN_COMPLETED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return 0
}

prepare_reference_baseline() {
  local log="$1" label="$2" spec="$3" workload="$4" i
  shift 4
  BASELINE_TMP="$(mktemp -d)" || return $?
  trap 'rm -rf "${BASELINE_TMP}"' EXIT
  _install_benchmark_environment latest_release || return $?
  local -a reference_results=()
  for i in 1 2 3; do
    local reference_log="${RUN_DIR}/logs/reference-${i}.log"
    _benchmark_once "${reference_log}" "${label} reference ${i}" \
      "${BASELINE_TMP}/reference-${i}" "$@" || return $?
    if [[ "${RUN_CODE}" != 0 ]]; then
      echo "reference ${i} failed; candidate will not run" >&2
      return "${RUN_CODE}"
    fi
    local reference_json="${BASELINE_TMP}/reference-${i}.json"
    "${PYTHON}" "${JAX_LAB_DIR}/utilities/benchcmp.py" \
      --benchspec "${spec}" --workload "${workload}" \
      --log "${reference_log}" --skip-comparison > "${reference_json}" || return $?
    reference_results+=("${reference_json}")
    rm -rf "${BASELINE_TMP}/reference-${i}"
  done
  "${PYTHON}" "${JAX_LAB_DIR}/utilities/benchcmp.py" \
    --benchspec "${spec}" --workload "${workload}" \
    --calculate-baseline "${reference_results[@]}" || return $?
}

run_candidate_benchmark() {
  local log="$1" label="$2"
  shift 2
  _install_benchmark_environment "${JAXLIB_VERSION:-head}" || return $?
  _benchmark_once "${log}" "${label}" "${BASELINE_TMP}/candidate" "$@"
}
