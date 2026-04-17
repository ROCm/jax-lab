#!/usr/bin/env python3

"""
Run a MaxText workload config.

Responsibilities:
- resolve configs/<workload>.yml
- forward extra args (optional)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def fail(msg: str) -> None:
    """Print an error and exit."""
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a MaxText workload.")
    parser.add_argument("--target-dir", required=True, help="Path to targets/maxtext.")
    parser.add_argument(
        "--repo-dir", required=True, help="Path to the MaxText checkout."
    )
    parser.add_argument(
        "--workload", required=True, help="Workload name under configs/."
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded after '--' to MaxText train.",
    )
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    work_dir = repo_dir / "src"
    cfg = Path(args.target_dir) / "configs" / f"{args.workload}.yml"

    if not cfg.exists():
        fail(f"missing config file: {cfg}")
    if not work_dir.is_dir():
        fail(f"missing src dir: {work_dir}")

    extra = args.args
    if extra and extra[0] == "--":
        extra = extra[1:]

    cmd = ["python3", "-m", "maxtext.trainers.pre_train.train", str(cfg), *extra]

    print("[command]")
    print(" ".join(cmd))

    subprocess.run(cmd, cwd=work_dir, check=True)


if __name__ == "__main__":
    main()
