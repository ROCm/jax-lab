#!/usr/bin/env python3
"""Collect benchmark-specific manifest fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def read_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(errors="replace")) or {}


def workload_metadata(benchspec: dict, workload: str) -> dict:
    workload_cfg = (benchspec.get("workloads") or {}).get(workload, {})
    return {
        "model_domain": workload_cfg.get("model_domain")
        or benchspec.get("model_domain"),
        "workload_type": workload_cfg.get("workload_type")
        or benchspec.get("workload_type"),
        "benchmark_goal": workload_cfg.get("benchmark_goal")
        or benchspec.get("benchmark_goal"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--run-code", required=True, type=int)
    parser.add_argument("--model-run-started-at", required=True)
    parser.add_argument("--model-run-completed-at", required=True)
    parser.add_argument("--benchmark-repo-commit", required=True)
    parser.add_argument("--benchmark-command", required=True)
    parser.add_argument("--config", action="append", default=[])
    args = parser.parse_args()

    configs = {}
    for item in args.config:
        name, path = item.split("=", 1)
        configs[name] = read_yaml(path)

    benchspec = configs.get("benchspec", {})
    meta = workload_metadata(benchspec, args.workload)

    print(
        json.dumps(
            {
                "combo": args.combo,
                "target": args.target,
                "workload": args.workload,
                **meta,
                "run_code": args.run_code,
                "model_run_started_at": args.model_run_started_at,
                "model_run_completed_at": args.model_run_completed_at,
                "benchmark_repo_commit": args.benchmark_repo_commit,
                "benchmark_command": args.benchmark_command,
                "benchmark_configs_json": configs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
