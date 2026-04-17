# JAX-LAB

A JAX‑based experimentation lab for systematic benchmarking of configurable model variants on AMD GPU architectures, aiming reproducible and comparable performance analysis.

## Self-Contained Model Runner
This repository provides a simple and reproducible way to run model workloads (such as MaxText) using Docker. The design is intentionally minimal: each component is explicit, easy to debug, and easy to extend.
At a high level, the system separates image preparation from workload execution:
- Docker images define the runtime environment (ROCm, JAX, system dependencies)
- Targets define how a model is executed
- Runs store outputs, logs, and metadata

All orchestration is handled by `launch.py`.


### Image Preparation (build).
The `build` command prepares reusable Docker images.
If a target contains a `Dockerfile`, it is used to `build` the image. When `--image` is provided, it is passed as `BASE_IMAGE` and overrides the parent image defined in the Dockerfile.
If a target does not have a `Dockerfile`, `build` can still create a new tag from an existing image.
If no target is specified, `build` acts as a simple retag command.

Examples:
```bash
# build using a base image
python3 launch.py build \
 --target maxtext \
 --image rocm/base:latest

# build with explicit output tag
python3 launch.py build \
 --target maxtext \
 --image rocm/base:latest \
 --tag my/maxtext:setup

# retag an existing image
python3 launch.py build \
 --image my/local:img \
 --tag my/local:stable
```

## Running Workloads (run).
The `run` command executes a workload using an existing image. It never builds.
A typical run looks like:
```
python3 launch.py run \
 --target maxtext \
 --image ghcr.io/rocm/jax-base-ubu24.rocm720:latest \
 --workload llama3_8b
```

During execution:
1. A container is started from the given `image`
2. The repository is mounted into the `container`
3. A shared checkout is prepared under `runs/<target>/repo`
4. A per-run directory is created under `runs/<target>/<run-id>/`
5. The target entrypoint is executed: `targets/<target>/run.sh`

Any arguments after `--` are forwarded directly to the target.`
`launch.py` does not interpret target-specific flags.


### Targets
Targets define how a workload is executed. Each target is fully self-contained:
```
targets/<target>/
 Dockerfile #TODO:
 run.sh
 executor.py
 requirements.txt
 configs/
```
- run.sh prepares the environment (repo, dependencies, env vars)
- executor.py launches the actual training process
- requirements.txt defines additional Python dependencies
- configs/ contains workload definitions

This structure allows targets to be reused and modified independently.

### Runs and Artifacts
All runtime data is stored under:
```
runs/<target>/
 repo/           # shared repository checkout
 <run-id>/       # logs, metadata, outputs
```

This avoids duplicating repositories while keeping runs isolated and reproducible.