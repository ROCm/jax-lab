# JAX-LAB

A JAX‑based experimentation lab for systematic benchmarking of configurable model variants on AMD GPU architectures, aiming reproducible and comparable performance analysis.

The repository is organized around benchmark targets under `targets/`,
shared infrastructure under `utilities/`, GitHub Actions workflows under
`.github/workflows/`.

```text
.github/workflows/
targets/
utilities/
launch.py
README.md
```

Each benchmark workload produces exactly one immutable `result.json` artifact.
This artifact is the single source of truth for benchmark metadata, execution
metadata, metric values, and regression comparison results. Database uploaders
only consume `result.json` and do not reconstruct metadata from CI environment
variables during ingestion.

Benchmark targets live under:

```text
targets/<target_name>/
```

A target defines workload execution, metric extraction, aggregation rules,
comparison logic, and final artifact generation. Typical target structure:

```text
targets/maxtext_release/
  run.sh
  benchspec.yml
  requirements.txt
  configs/
    gemma3_4b.yml
```

Current targets include:

```text
targets/maxtext_release/
targets/flax_release/
```

`run.sh` executes the workload and produces the final `result.json`.
`benchspec.yml` defines benchmark metrics, aggregation rules, baselines,
thresholds, and comparison behavior. Optional workload configuration can be
stored under `configs/`, while target-specific Python dependencies may be
defined in `requirements.txt`.

Metrics are defined declaratively in `benchspec.yml`.

```yaml
metrics:
  tflops_per_device:
    role: sample
    lines: "completed_step"
    step: "step"
    value: "TFLOP/s/device"

  median_tflops_per_device:
    role: aggregate
    from: tflops_per_device
    op: median
    skip_first: 4

  tflops_regression_pct:
    role: comparison
    from: median_tflops_per_device
    baseline: 700
    threshold: 10
    better: higher
```

Three metric roles are supported:
- `sample`: raw values extracted from benchmark logs
- `aggregate`: derived metrics computed from samples
- `comparison`: regression or comparison metrics

Comparison metrics may additionally store observed values, baseline metadata,
threshold metadata, and comparison direction.

Example comparison result:

```json
{
  "metric": "tflops_regression_pct",
  "role": "comparison",
  "value": 1.8,
  "observed_value": 726.4,
  "baseline_value": 700,
  "threshold_pct": 10,
  "better": "higher"
}
```

Example `result.json` structure:

```json
{
  "schema_version": 1,
  "run_started_at": "...",
  "run_completed_at": "...",
  "github_repository": "...",
  "github_sha": "...",
  "python_version": "3.12",
  "rocm_version": "7.2.0",
  "target": "maxtext_release",
  "workload": "gemma3_4b",
  "combo": "8gpu-maxtext_release-gemma3_4b-py3.12",
  "run_key": "...",
  "run_code": 0,
  "cmp_code": 0,
  "results": [
    {
      "metric": "tflops_per_device",
      "role": "sample",
      "step": 12,
      "value": 728.1
    },
    {
      "metric": "median_tflops_per_device",
      "role": "aggregate",
      "value": 726.4
    },
    {
      "metric": "tflops_regression_pct",
      "role": "comparison",
      "value": 1.8,
      "observed_value": 726.4,
      "baseline_value": 700,
      "threshold_pct": 10,
      "better": "higher"
    }
  ]
}
```

Benchmark results are uploaded into a generic schema consisting of:

```text
jax_ci_benchmark_runs
jax_ci_benchmark_metrics
jax_ci_benchmark_results
```

The schema is intentionally target-agnostic so that new benchmark targets and
workloads can be added without requiring schema changes.

Benchmarks are primarily executed through GitHub Actions workflows. Nightly
workflows upload benchmark results automatically, while manual runs only upload
results when `send-to-db=true` is specified. Each matrix entry uploads a single
artifact containing `result.json`.

Local experimentation and debugging can still be performed through `launch.py`.

```bash
python3 launch.py run \
  --target maxtext_release \
  --image ghcr.io/rocm/jax-base-ubu24.rocm720:latest \
  --workload gemma3_4b
```

