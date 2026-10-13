#!/usr/bin/env bash
# Capture benchmark output; quiet CI success, tail on failure.

show_log_tail() {
  local log_file="$1"
  local line_count="${2:-${JAX_LAB_LOG_TAIL_LINES:-2}}"

  if [[ ! -s "${log_file}" ]]; then
    echo "benchmark log is empty: ${log_file}" >&2
    return
  fi

  echo "last ${line_count} lines from ${log_file}:" >&2
  tail -n "${line_count}" "${log_file}" >&2 || true
}

run_with_log() (
  set +e

  local log_file="$1"
  local label="$2"
  shift 2

  local status=0
  local tee_status=0
  local -a pipeline_status=()

  mkdir -p "$(dirname "${log_file}")"
  : > "${log_file}"

  if [[ "${GITHUB_ACTIONS:-false}" == "true" &&
        "${JAX_LAB_VERBOSE:-0}" != "1" ]]; then
    "$@" >"${log_file}" 2>&1
    status=$?
  else
    "$@" 2>&1 | tee "${log_file}"
    pipeline_status=("${PIPESTATUS[@]}")

    status="${pipeline_status[0]}"
    tee_status="${pipeline_status[1]}"

    if [[ "${status}" -eq 0 && "${tee_status}" -ne 0 ]]; then
      status="${tee_status}"
    fi
  fi

  if [[ "${status}" -ne 0 ]]; then
    echo "${label} failed with exit code ${status}" >&2

    if [[ "${GITHUB_ACTIONS:-false}" == "true" &&
          "${JAX_LAB_VERBOSE:-0}" != "1" ]]; then
      show_log_tail "${log_file}"
    fi
  fi

  exit "${status}"
)

benchmark_command_string() {
  printf -v BENCHMARK_COMMAND '%q ' "$@"
  BENCHMARK_COMMAND="${BENCHMARK_COMMAND% }"
}

# Called after a target has prepared its dependencies and benchmark command.
# Targets may define append_metric_from_log(log) and set
# BASELINE_OUTPUT_FLAG to keep per-run output separate.
_benchmark_once() {
  local log="$1" label="$2" output_dir="$3"
  shift 3
  local -a command=("$@")
  local i found=0

  if [[ -n "${output_dir}" && -n "${BASELINE_OUTPUT_FLAG:-}" ]]; then
    mkdir -p "${output_dir}"
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
  if declare -F append_metric_from_log >/dev/null; then
    append_metric_from_log "${log}" || RUN_CODE=1
  fi
  MODEL_RUN_COMPLETED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

run_with_baseline() {
  local log="$1" label="$2" spec="$3" workload="$4" metric="$5"
  shift 5
  local candidate_version="${JAXLIB_VERSION:-head}" i value baseline_value
  local -a references=()
  BASELINE_TMP=""
  BENCHCMP_ARGS=(--skip-comparison)

  if [[ -n "${REFERENCE_JAXLIB_VERSION:-}" ]]; then
    BASELINE_TMP="$(mktemp -d)"
    trap 'rm -rf "${BASELINE_TMP}"' EXIT
    export JAXLIB_VERSION="${REFERENCE_JAXLIB_VERSION}"
    (cd "${JAX_LAB_DIR}" && bash "${JAX_LAB_DIR}/utilities/install_jax_wheels.sh") || return $?

    for i in 1 2 3; do
      _benchmark_once "${log}" "${label} reference ${i}" \
        "${BASELINE_TMP}/reference-${i}" "$@" || return $?
      if [[ "${RUN_CODE}" -ne 0 ]]; then
        echo "reference ${i} failed" >&2
        return 1
      fi
      value="$("${PYTHON:-python3}" "${JAX_LAB_DIR}/utilities/benchcmp.py" \
        --benchspec "${spec}" --workload "${workload}" --log "${log}" \
        --skip-comparison --value-for "${metric}")" || return $?
      references+=("${value}")
      echo "reference ${i} complete"
      rm -rf "${BASELINE_TMP}/reference-${i}"
    done

    baseline_value="$("${PYTHON:-python3}" -c \
      'import statistics, sys; print(statistics.median(map(float, sys.argv[1:])))' \
      "${references[@]}")" || return $?
    echo "reference median computed"
    BENCHCMP_ARGS=(--baseline-value "${baseline_value}")
  fi

  export JAXLIB_VERSION="${candidate_version}"
  (cd "${JAX_LAB_DIR}" && bash "${JAX_LAB_DIR}/utilities/install_jax_wheels.sh") || return $?
  _benchmark_once "${log}" "${label}" \
    "${BASELINE_TMP:+${BASELINE_TMP}/candidate}" "$@" || return $?

  if [[ -n "${BASELINE_TMP}" ]]; then
    rm -rf "${BASELINE_TMP}"
    trap - EXIT
  fi
}
