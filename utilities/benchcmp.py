#!/usr/bin/env python3
"""Compare benchmark metrics from a run log against benchspec.yml."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Optional

import yaml


def read_text(path: str) -> str:
    return Path(path).read_text(errors="replace")


def parse_number(line: str, key: str) -> Optional[float]:
    match = re.search(
        rf"{re.escape(key)}\s*[:=]\s*([-+]?[0-9]*\.?[0-9]+)",
        line,
    )
    return float(match.group(1)) if match else None


def result_row(metric: str, role: str, value: float, *, step: int = 0, **extra):
    row = {
        "metric": metric,
        "role": role,
        "step": int(step or 0),
        "value": float(value),
    }
    row.update({k: v for k, v in extra.items() if v is not None})
    return row


def parse_samples(log: str, metric: dict) -> list[dict]:
    rows = []

    for line in log.splitlines():
        if metric["lines"] not in line:
            continue

        value = parse_number(line, metric["value"])
        if value is None:
            continue

        step = 0
        if metric.get("step"):
            parsed_step = parse_number(line, metric["step"])
            step = int(parsed_step) if parsed_step is not None else 0

        rows.append({"step": step, "value": value})

    return rows


def baseline_for(metric: dict, workload: str) -> float:
    if "baseline" in metric:
        return float(metric["baseline"])

    key = workload.replace("-", "_").replace(".", "_")
    baselines = metric.get("baselines") or {}

    if key not in baselines:
        raise ValueError(f"no baseline for workload {workload!r}")

    return float(baselines[key])


def aggregate(values: list[float], metric: dict) -> float:
    values = values[int(metric.get("skip_first", 0)) :]

    if not values:
        raise ValueError("no values available for aggregation")

    op = metric["op"]
    if op == "median":
        return float(statistics.median(values))
    if op == "mean":
        return float(statistics.mean(values))
    if op == "min":
        return float(min(values))
    if op == "max":
        return float(max(values))

    raise ValueError(f"unknown aggregate op: {op}")


def regression_percent(value: float, baseline: float) -> float:
    if baseline == 0:
        raise ValueError("baseline must be non-zero")
    return abs((value - baseline) / baseline) * 100.0


def is_regression(value: float, baseline: float, better: str) -> bool:
    if better == "lower":
        return value > baseline
    if better == "higher":
        return value < baseline
    raise ValueError(f"unknown direction: {better}")


def better_from_chain(spec: dict, from_name: str) -> Optional[str]:
    name = from_name
    seen = set()

    while name and name not in seen:
        seen.add(name)
        metric = spec["metrics"][name]

        if metric.get("better"):
            return metric["better"]

        if metric["role"] == "sample":
            break

        name = metric.get("from")

    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare benchmark log metrics against benchspec.yml"
    )
    parser.add_argument("--benchspec", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    spec = yaml.safe_load(read_text(args.benchspec)) or {}
    log = read_text(args.log)

    values = {}
    results = []
    cmp_code = 0

    for name, metric in spec["metrics"].items():
        role = metric["role"]

        if role == "sample":
            rows = parse_samples(log, metric)
            values[name] = [row["value"] for row in rows]

            if not rows:
                raise ValueError(f"no samples found for metric {name!r}")

            for row in rows:
                results.append(
                    result_row(
                        name,
                        role,
                        row["value"],
                        step=row["step"],
                        better=metric.get("better"),
                    )
                )

        elif role == "aggregate":
            value = aggregate(values[metric["from"]], metric)
            values[name] = [value]

            better = metric.get("better") or better_from_chain(spec, metric["from"])
            results.append(result_row(name, role, value, better=better))

        elif role == "comparison":
            from_name = metric["from"]
            measured = values[from_name][-1]
            baseline = baseline_for(metric, args.workload)
            better = metric.get("better") or better_from_chain(spec, from_name)

            if not better:
                raise ValueError(f"no better direction for comparison {name!r}")

            pct = regression_percent(measured, baseline)
            if is_regression(measured, baseline, better) and pct > float(
                metric["threshold"]
            ):
                cmp_code = 1

            values[name] = [pct]
            results.append(
                result_row(
                    name,
                    role,
                    pct,
                    better=better,
                    observed_value=measured,
                    baseline_value=baseline,
                    baseline_metric=from_name,
                    threshold_pct=float(metric["threshold"]),
                )
            )

        else:
            raise ValueError(f"unknown metric role: {role}")

    print(json.dumps({"cmp_code": cmp_code, "results": results}, indent=2))
    return cmp_code


if __name__ == "__main__":
    raise SystemExit(main())
