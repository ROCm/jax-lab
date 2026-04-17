#!/usr/bin/env bash
set -euo pipefail

# Run a MaxText workload inside the prepared container.
#
# Responsibilities:
# - keep a shared repo checkout under runs/maxtext/repo
# - optionally install target-level Python requirements
# - resolve workload files under configs/
# - load workload environment from <workload>.env.sh
# - write per-run logs and metadata under /run_artifacts
# - use executor.py to run MaxText.train
#
# Notes:
# - repo and branch are fixed for this target on purpose
# - workload has a default and can be overridden
# - repo is cloned only if missing, or if --reclone is given
# - if repo already exists and --reclone is not given, it is left untouched

ROOT=/workspace
TARGET=maxtext
TARGET_DIR="$ROOT/targets/$TARGET"
EXECUTOR="$TARGET_DIR/executor.py"

# Fixed source for this target.
REPO_URL="https://github.com/ROCm/maxtext.git"
BRANCH="main"

# Default workload. Override with --workload <name>.
WORKLOAD="llama3_8b"

# Force a fresh clone of the shared repo checkout.
RECLONE=0

# Extra args forwarded after "--" to MaxText.train.
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workload)
      WORKLOAD="$2"
      shift 2
      ;;
    --reclone)
      RECLONE=1
      shift
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# launch.py mounts the current run directory here.
RUN_DIR=/run_artifacts

# Shared target state and shared repo checkout.
RUN_ROOT="$ROOT/runs/$TARGET"
REPO_DIR="$RUN_ROOT/repo"

# Per-run files.
LOG_FILE="$RUN_DIR/run.log"
META_FILE="$RUN_DIR/meta.txt"

# Workload files follow a simple convention.
ENV_FILE="$TARGET_DIR/configs/$WORKLOAD.env.sh"
CFG_FILE="$TARGET_DIR/configs/$WORKLOAD.yml"

[[ -f "$ENV_FILE" ]] || { echo "error: missing env file: $ENV_FILE" >&2; exit 2; }
[[ -f "$CFG_FILE" ]] || { echo "error: missing config file: $CFG_FILE" >&2; exit 2; }

mkdir -p "$RUN_ROOT" "$RUN_DIR"

# Recreate the repo only when explicitly requested.
if [[ "$RECLONE" -eq 1 ]]; then
  rm -rf "$REPO_DIR"
fi

# First run (or after --reclone): clone the repo.
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi


REPO_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
START_TIME="$(date -Iseconds)"

# Small metadata file for debugging and later parsing.
{
  echo "start_time=$START_TIME"
  echo "target=$TARGET"
  echo "workload=$WORKLOAD"
  echo "repo_url=$REPO_URL"
  echo "repo_branch=$BRANCH"
  echo "repo_commit=$REPO_COMMIT"
  echo "repo_dir=$REPO_DIR"
  echo "run_dir=$RUN_DIR"
  echo "requirements_file=$REQUIREMENTS_FILE"
  echo "env_file=$ENV_FILE"
  echo "config_file=$CFG_FILE"
  echo "cmd=python3 -m MaxText.train $CFG_FILE ${EXTRA_ARGS[*]}"
} > "$META_FILE"

# Load workload-specific environment in the current shell.
cd "$REPO_DIR"
source "$ENV_FILE"

# Run the workload and keep both console output and a persistent log.
STATUS=0
python3 "$EXECUTOR" \
  --target-dir "$TARGET_DIR" \
  --repo-dir "$REPO_DIR" \
  --workload "$WORKLOAD" \
  -- "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE" || STATUS=$?

END_TIME="$(date -Iseconds)"

{
  echo "end_time=$END_TIME"
  echo "exit_code=$STATUS"
} >> "$META_FILE"

exit "$STATUS"
