# JAX-LAB

JAX-LAB is a JAX benchmarking framework for reproducible and comparable evaluation
of configurable workloads on AMD GPUs.

## Repository layout

```text
.github/workflows/  CI workflows
targets/            Benchmark targets and workload configurations
utilities/          Wheel installation, metric processing, manifests, and database upload
docs/               Design documentation
launch.py           Local Docker entry point
```

## Result contract

Each completed workload run produces one final `result.json`. It is the
canonical persisted record of benchmark metadata, execution metadata, metric
values, and comparison results.

Artifact upload and database ingestion consume only the final `result.json`.
Intermediate logs and reference results are not part of the persisted result
contract.

## Targets

| Target | Workloads |
|--------|-----------|
| `maxtext_release` | `gemma3_4b`, `llama3_1_8b`, `mixtral_8x7b`, `deepseek2_16b`, `qwen3_14b`, `gpt_oss_20b`, `olmo3_7b` |
| `flax_release` | `convolution`, `resnet`, `nlp_seq` |
| `hf_transformers_release` | `gpt_j_6b`, `flan_t5_large` |
| `hf_diffusers_release` | `stable_diffusion` |

## Target structure

Benchmark targets live under `targets/<target_name>/`.

A target owns workload-specific execution and configuration. Shared utilities
handle metric extraction, aggregation, comparison, manifest collection, and
database ingestion.

```text
targets/maxtext_release/
├── run.sh
├── benchspec.yml
├── requirements.txt
└── configs/
    └── gemma3_4b.yml
```

- `run.sh` prepares and executes the workload and writes the final `result.json`.
- `benchspec.yml` defines metrics, aggregation rules, comparison behavior,
  thresholds, and workload baseline entries.
- `configs/` contains optional workload-specific configuration.
- `requirements.txt` contains optional target-specific Python dependencies.

## Benchmark specification

Metrics are defined declaratively in `benchspec.yml`.

```yaml
schema_version: 1

model_domain: llm
workload_type: train
benchmark_goal: throughput

metrics:
  tflops_per_device:
    role: sample
    lines: "completed step"
    step: step
    value: "TFLOP/s/device"
    better: higher

  step_time_seconds:
    role: sample
    lines: "completed step"
    step: step
    value: "seconds"
    better: lower

  median_tflops_per_device:
    role: aggregate
    from: tflops_per_device
    op: median
    skip_first: 3

  median_step_time_seconds:
    role: aggregate
    from: step_time_seconds
    op: median
    skip_first: 3

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

Three metric roles are supported:

- `sample`: raw values extracted from benchmark logs
- `aggregate`: values derived from sample metrics
- `comparison`: an observed aggregate compared with its baseline

Generated comparison rows may include `observed_value`, `baseline_value`,
`baseline_metric`, `threshold_pct`, and comparison direction.

## Runtime baseline comparison

Targets using runtime baselines derive the active baseline from reference runs
before evaluating the candidate.

See [`docs/baseline_comparison.md`](docs/baseline_comparison.md) for the
execution and failure model.

## Database

Benchmark results are stored in a target-agnostic schema:

```text
jax_ci_benchmark_runs
jax_ci_benchmark_metrics
jax_ci_benchmark_results
```

New targets, workloads, and metrics can be added without target-specific schema
changes.

## GitHub Actions

Benchmarks are primarily executed through GitHub Actions.

Scheduled runs upload results automatically. Manual runs upload results only when
`send-to-db=true`.

Each matrix entry publishes one artifact containing the final `result.json`.

## Local runs

`launch.py` supports local Docker-based execution and debugging:

```bash
python3 launch.py run \
  --target maxtext_release \
  --image ghcr.io/rocm/jax-base-ubu24.therock-7.14:latest \
  --workload gemma3_4b
```

Some targets and wheel sources may require additional environment variables or
credentials.

## License

Apache License 2.0. See [LICENSE](LICENSE).
