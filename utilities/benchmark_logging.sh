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

