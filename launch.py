#!/usr/bin/env python3

"""
(Prepare images) and run targets where:
- image = environment (rocm, (jax, deps))
- target = what to run (scripts, configs)

Commands:
- build: prepare an image (build or retag)
- run: run a target using an existing image
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGETS = ROOT / "targets"
RUNS = ROOT / "runs"


def fail(msg: str) -> None:
    """Print an error and exit."""
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(2)


def run_cmd(label: str, cmd: list[str], check: bool = True):
    """Print and execute a command."""
    print(f"[{label}]")
    print(" ".join(cmd))
    return subprocess.run(cmd, check=check)


def build(target: str | None, image: str | None, tag: str | None) -> str:
    """
    Build modes:
    - target build:
        use targets/<target>/Dockerfile
        (override BASE_IMAGE via --image)
    - retag:
        if no target is given, just tag an existing image
    """
    if not target:
        if not image:
            fail("build without --target requires --image")
        if not tag:
            fail("build without --target requires --tag")
        run_cmd("docker-tag", ["docker", "tag", image, tag])
        return tag

    tdir = TARGETS / target
    if not tdir.is_dir():
        fail(f"target not found: {target}")

    dockerfile = tdir / "Dockerfile"
    out = tag or f"local/{target}:dev"

    # target build with Dockerfile
    if dockerfile.exists():
        cmd = [
            "docker",
            "build",
            "-t",
            out,
            "-f",
            str(dockerfile),
        ]

        # Optional parent image override. If omitted, the Dockerfile uses
        # its own default base image logic.
        if image:
            cmd += ["--build-arg", f"BASE_IMAGE={image}"]

        cmd += [str(ROOT)]
        run_cmd("docker-build", cmd)
        return out

    # target without Dockerfile: just create a target-scoped tag if an image is provided
    if not image:
        fail(f"target {target} has no Dockerfile, so --image is required")

    run_cmd("docker-tag", ["docker", "tag", image, out])
    return out


def run(target: str, image: str, workload: str | None, extra: list[str]) -> int:
    """Run a target using an image.

    This never builds.
    It mounts the repo and calls targets/<target>/run.sh.
    """
    tdir = TARGETS / target
    if not tdir.is_dir():
        fail(f"target not found: {target}")

    run_sh = tdir / "run.sh"
    if not run_sh.exists():
        fail(f"missing run.sh in {target}")

    if extra and extra[0] == "--":
        extra = extra[1:]

    name = workload or "run"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS / target / f"{name}-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_cmd_list = [
        "docker",
        "run",
        "-d",
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri",
        "--group-add",
        "video",
        "--ipc=host",
        "-v",
        f"{ROOT}:/workspace",
        "-v",
        f"{run_dir}:/run_artifacts",
        "-e",
        "RUN_ARTIFACTS=/run_artifacts",
        "--name",
        f"{target}-{ts}",
        image,
        "tail",
        "-f",
        "/dev/null",
    ]

    result = subprocess.run(run_cmd_list, capture_output=True, text=True, check=True)
    container_id = result.stdout.strip()

    exec_cmd = [
        "docker",
        "exec",
        container_id,
        "/bin/bash",
        f"/workspace/targets/{target}/run.sh",
    ]

    if workload:
        exec_cmd += ["--workload", workload]

    exec_cmd += extra

    return run_cmd("docker-exec", exec_cmd, check=False).returncode


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare images and run targets.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Prepare an image for reuse.")
    b.add_argument("--target", help="Target under targets/<target>.")
    b.add_argument(
        "--image",
        help="Parent image. The target Dockerfile already defines its own base.",
    )
    b.add_argument(
        "--tag",
        help="Output image tag. Defaults to local/<target>:dev when --target is set.",
    )

    r = sub.add_parser("run", help="Run a target using an existing image.")
    r.add_argument("--target", required=True, help="Target under targets/<target>.")
    r.add_argument("--image", required=True, help="Image to run with.")
    r.add_argument("--workload", help="Workload forwarded to run.sh.")
    r.add_argument("args", nargs=argparse.REMAINDER, help="Args forwarded after --.")

    args = p.parse_args()
    RUNS.mkdir(exist_ok=True)

    if args.cmd == "build":
        image = build(args.target, args.image, args.tag)
        print(image)
        return

    sys.exit(run(args.target, args.image, args.workload, args.args))


if __name__ == "__main__":
    main()
