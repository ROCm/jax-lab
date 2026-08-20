# Runtime Baseline Comparison

The comparison baseline is derived from three reference runs in the same CI
job. Reference and candidate runs use the same runner, container, target
checkout, configuration, and prepared dependencies. Only the installed wheel set
changes.

## Execution

1. Prepare the target once.
2. Install the reference wheels once.
3. Run the reference benchmark three times with `--skip-comparison`.
4. Validate the reference results and calculate the median with
   `--calculate-baseline`.
5. Install the candidate wheels once.
6. Run the candidate benchmark through the normal comparison path.
7. Produce one final `result.json`.

Reusable caches and input data are shared. Run-specific outputs and checkpoints
remain isolated.

## `benchspec.yml`

Metric definitions, aggregation rules, comparison direction, and thresholds
remain unchanged. Baseline entries are resolved during execution:

```yaml
metrics:
  tflops_regression_pct:
    role: comparison
    from: median_tflops_per_device
    better: higher
    threshold: 10
    baselines:
      gemma3_4b:
      llama3_1_8b:
      # ...
```

## Reference runs

`--skip-comparison` produces sample and aggregate metrics without evaluating
comparison metrics. It can also support baseline-only jobs that persist reference
results for later use.

All three reference runs must succeed. Each result must contain exactly one valid
aggregate named by the comparison metric's `from` field in `benchspec.yml`.

`--calculate-baseline` validates the three aggregates, computes their median, and
sets the active workload baseline. Missing or invalid results stop execution
before the candidate run.

CI reports reference summaries and relevant failure output. Intermediate
reference logs and JSON files are not uploaded.

## Candidate comparison

The candidate run uses the existing comparison path (no `--skip-comparison`).
`baseline_for()` reads the runtime baseline written after the reference phase.

Only the candidate writes the final result:

```text
targets/<target>/run_artifacts/${WORKLOAD}/result.json
```

`MODEL_RUN_*` timestamps cover the candidate execution only.

## Workflow

The workflow supplies separate reference and candidate wheel selections:

```yaml
REFERENCE_JAXLIB_VERSION: release
REFERENCE_S3_WHEELS_URI: ${{ vars.REFERENCE_S3_WHEELS_URI }}

# Candidate wheels
JAXLIB_VERSION: ${{ github.event_name == 'schedule' && 'head' || inputs['jaxlib-version'] || 'head' }}
```

`run.sh` is invoked once. It prepares the target once, installs each wheel set
once, executes the reference benchmark three times, and executes the candidate
benchmark once.

Artifact and database upload behavior remains unchanged. Only the final
candidate result is persisted.

Reference aggregates may later be loaded from the database without changing the
candidate comparison path.
