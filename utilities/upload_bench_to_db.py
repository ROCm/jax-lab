#!/usr/bin/env python3
"""Upload benchmark result.json into MySQL."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# pylint: disable=import-error
import mysql.connector
from mysql.connector import Error as MySQLError

BATCH_SIZE = 2000


def int_or_none(value):
    return int(value) if value not in (None, "") else None


def parse_iso_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(
        tzinfo=None
    )


def pipe_split(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [p for p in (x.strip() for x in raw.split("|")) if p]


def require_field(data: dict, key: str):
    value = data.get(key)
    if value in (None, ""):
        raise SystemExit(f"Benchmark result missing required field: {key}")
    return value


def packages_json_and_jax_version(
    raw: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    if not raw:
        return None, None

    packages = []
    jax_version = None

    for item in pipe_split(raw):
        name, sep, version = item.partition("==")
        name = name.strip()
        version = version.strip() if sep else None
        packages.append({"name": name, "version": version, "raw": item})
        if name == "jax" and version:
            jax_version = version

    return json.dumps(packages), jax_version


def wheels_json(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None

    wheels = []
    for line in pipe_split(raw):
        match = re.match(r"^([0-9a-fA-F]{64})\s+(.+)$", line)
        if match:
            wheels.append(
                {"sha256": match.group(1).lower(), "file": match.group(2).strip()}
            )
        else:
            wheels.append({"sha256": None, "file": line})
    return json.dumps(wheels)


def config_json(data: dict) -> Optional[str]:
    cfg = data.get("benchmark_configs_json")
    if cfg is None:
        return None
    return cfg if isinstance(cfg, str) else json.dumps(cfg)


def load_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_run_fields(data: dict, *, run_tag: str, gpu_tag: str) -> dict:
    packages_json, _ = packages_json_and_jax_version(data.get("jax_packages_raw"))

    return {
        "run_started_at": parse_iso_dt(data.get("run_started_at")),
        "run_completed_at": parse_iso_dt(data.get("run_completed_at")),
        "model_run_started_at": parse_iso_dt(data.get("model_run_started_at")),
        "model_run_completed_at": parse_iso_dt(data.get("model_run_completed_at")),
        "github_repository": data.get("github_repository"),
        "github_ref_name": data.get("github_ref_name"),
        "github_ref": data.get("github_ref"),
        "github_event_name": data.get("github_event_name"),
        "github_run_url": data.get("github_run_url"),
        "github_sha": data.get("github_sha"),
        "github_run_id": int_or_none(data.get("github_run_id")),
        "github_run_attempt": int_or_none(data.get("github_run_attempt")),
        "github_run_number": int_or_none(data.get("github_run_number")),
        "github_workflow": data.get("github_workflow"),
        "github_job": data.get("github_job"),
        "schema_version": int_or_none(require_field(data, "schema_version")),
        "is_nightly": data.get("is_nightly"),
        "jaxlib_version": data.get("jaxlib_version"),
        "run_key": require_field(data, "run_key"),
        "run_tag": run_tag,
        "combo": require_field(data, "combo"),
        "target": require_field(data, "target"),
        "workload": require_field(data, "workload"),
        "model_domain": data.get("model_domain"),
        "workload_type": data.get("workload_type"),
        "benchmark_goal": data.get("benchmark_goal"),
        "runner": data.get("runner"),
        "python_version": data.get("python_version"),
        "rocm_version": data.get("rocm_version"),
        "rocm_tag": data.get("rocm_tag"),
        "gpu_count": int_or_none(data.get("gpu_count")),
        "gpu_tag": gpu_tag,
        "run_code": int_or_none(data.get("run_code")),
        "cmp_code": int_or_none(data.get("cmp_code")),
        "benchmark_repo_commit": data.get("benchmark_repo_commit"),
        "benchmark_command": data.get("benchmark_command"),
        "base_image_name": data.get("base_image_name"),
        "base_image_digest": data.get("base_image_digest"),
        "packages_json": packages_json,
        "wheels_json": wheels_json(data.get("wheels_sha_raw")),
        "benchmark_configs_json": config_json(data),
    }


def connect():
    return mysql.connector.connect(
        host=os.environ["ROCM_JAX_DB_HOSTNAME"],
        user=os.environ["ROCM_JAX_DB_USERNAME"],
        password=os.environ["ROCM_JAX_DB_PASSWORD"],
        database=os.environ["ROCM_JAX_DB_NAME"],
        autocommit=False,
    )


def find_existing_run_id(cur, fields: dict) -> Optional[int]:
    cur.execute(
        """
        SELECT id
        FROM jax_ci_benchmark_runs
        WHERE run_key = %s AND combo = %s
        LIMIT 1
        """,
        (fields["run_key"], fields["combo"]),
    )
    row = cur.fetchone()
    return int(row[0]) if row else None


def insert_run(cur, fields: dict) -> int:
    fields = dict(fields)
    fields["ingested_at"] = datetime.now(timezone.utc).replace(tzinfo=None)

    cur.execute(
        """
        INSERT INTO jax_ci_benchmark_runs (
          run_started_at, run_completed_at,
          model_run_started_at, model_run_completed_at, ingested_at,
          github_repository, github_ref_name, github_ref, github_event_name,
          github_run_url, github_sha, github_run_id, github_run_attempt,
          github_run_number, github_workflow, github_job,
          schema_version, is_nightly, jaxlib_version, run_key, run_tag, combo,
          target, workload, model_domain, workload_type, benchmark_goal,
          runner, python_version, rocm_version, rocm_tag, gpu_count, gpu_tag,
          run_code, cmp_code,
          benchmark_repo_commit, benchmark_command,
          base_image_name, base_image_digest,
          packages_json, wheels_json, benchmark_configs_json
        ) VALUES (
          %(run_started_at)s, %(run_completed_at)s,
          %(model_run_started_at)s, %(model_run_completed_at)s, %(ingested_at)s,
          %(github_repository)s, %(github_ref_name)s, %(github_ref)s,
          %(github_event_name)s, %(github_run_url)s, %(github_sha)s,
          %(github_run_id)s, %(github_run_attempt)s, %(github_run_number)s,
          %(github_workflow)s, %(github_job)s,
          %(schema_version)s, %(is_nightly)s, %(jaxlib_version)s,
          %(run_key)s, %(run_tag)s, %(combo)s,
          %(target)s, %(workload)s, %(model_domain)s,
          %(workload_type)s, %(benchmark_goal)s,
          %(runner)s, %(python_version)s, %(rocm_version)s,
          %(rocm_tag)s, %(gpu_count)s, %(gpu_tag)s,
          %(run_code)s, %(cmp_code)s,
          %(benchmark_repo_commit)s, %(benchmark_command)s,
          %(base_image_name)s, %(base_image_digest)s,
          %(packages_json)s, %(wheels_json)s, %(benchmark_configs_json)s
        )
        """,
        fields,
    )
    return int(cur.lastrowid)


def sync_metrics_and_get_ids(cur, results: List[dict]) -> Dict[Tuple[str, str], int]:
    pairs = sorted(
        {(require_field(row, "metric"), require_field(row, "role")) for row in results}
    )
    if not pairs:
        return {}

    cur.execute("DROP TEMPORARY TABLE IF EXISTS tmp_benchmark_metrics_")
    cur.execute("""
        CREATE TEMPORARY TABLE tmp_benchmark_metrics_ (
          name VARCHAR(255) NOT NULL,
          role VARCHAR(32) NOT NULL,
          PRIMARY KEY (name, role)
        ) ENGINE=InnoDB
        """)
    cur.executemany(
        "INSERT IGNORE INTO tmp_benchmark_metrics_ (name, role) VALUES (%s, %s)",
        pairs,
    )

    cur.execute("""
        INSERT INTO jax_ci_benchmark_metrics (name, role)
        SELECT s.name, s.role
        FROM tmp_benchmark_metrics_ s
        LEFT JOIN jax_ci_benchmark_metrics m
          ON m.name = s.name AND m.role = s.role
        WHERE m.id IS NULL
        """)

    cur.execute("""
        SELECT m.id, s.name, s.role
        FROM tmp_benchmark_metrics_ s
        JOIN jax_ci_benchmark_metrics m
          ON m.name = s.name AND m.role = s.role
        """)
    return {(name, role): int(metric_id) for metric_id, name, role in cur.fetchall()}


def insert_results(
    cur,
    run_id: int,
    metric_ids: Dict[Tuple[str, str], int],
    results: List[dict],
) -> None:
    rows = []
    for row in results:
        metric = require_field(row, "metric")
        role = require_field(row, "role")
        rows.append(
            (
                run_id,
                metric_ids[(metric, role)],
                int_or_none(row.get("step")) or 0,
                float(require_field(row, "value")),
                row.get("observed_value"),
                row.get("better"),
                row.get("baseline_value"),
                row.get("baseline_metric"),
                row.get("threshold_pct"),
            )
        )

    sql = """
        INSERT INTO jax_ci_benchmark_results (
          run_id, metric_id, step, value, observed_value, better,
          baseline_value, baseline_metric, threshold_pct
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
          value = VALUES(value),
          observed_value = VALUES(observed_value),
          better = VALUES(better),
          baseline_value = VALUES(baseline_value),
          baseline_metric = VALUES(baseline_metric),
          threshold_pct = VALUES(threshold_pct)
    """
    for i in range(0, len(rows), BATCH_SIZE):
        cur.executemany(sql, rows[i : i + BATCH_SIZE])


def upload_benchmark_results(result_json: Path, *, run_tag: str, gpu_tag: str) -> None:
    data = load_result(result_json)
    fields = build_run_fields(data, run_tag=run_tag, gpu_tag=gpu_tag)
    results = data.get("results") or []

    conn = connect()
    cur = conn.cursor()

    try:
        existing_run_id = find_existing_run_id(cur, fields)
        if existing_run_id is not None:
            conn.rollback()
            print(
                "[DUPLICATE] run already exists: "
                f"run_id={existing_run_id} run_key={fields['run_key']} "
                f"combo={fields['combo']}"
            )
            return

        run_id = insert_run(cur, fields)

        if results:
            metric_ids = sync_metrics_and_get_ids(cur, results)
            insert_results(cur, run_id, metric_ids, results)

        conn.commit()
        print(
            f"[summary] run_id={run_id} combo={fields['combo']} "
            f"target={fields['target']} workload={fields['workload']} "
            f"results={len(results)}"
        )

    except MySQLError as e:
        conn.rollback()
        if getattr(e, "errno", None) == 1062:
            print(
                "[DUPLICATE] insert hit unique constraint: "
                f"run_key={fields['run_key']} combo={fields['combo']}"
            )
            return
        raise SystemExit(f"MySQL error: {e}") from e
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload benchmark result JSON to MySQL"
    )
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--gpu-tag", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    upload_benchmark_results(
        Path(args.result_json),
        run_tag=args.run_tag,
        gpu_tag=args.gpu_tag,
    )
