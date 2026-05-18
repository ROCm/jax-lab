#!/usr/bin/env python3
"""Collect final benchmark result manifest."""

import argparse
import json
import os
import re
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def env(name, default=""):
    return os.environ.get(name, default)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_date(started):
    if started:
        return datetime.fromisoformat(started.replace("Z", "+00:00")).strftime(
            "%Y-%m-%d"
        )
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def get_json(url, headers=None):
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode())


def get_headers(url, headers=None):
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.headers


def github_api_headers():
    headers = {"Accept": "application/vnd.github+json"}
    token = env("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def github_run_started_at():
    repo = env("GITHUB_REPOSITORY")
    run_id = env("GITHUB_RUN_ID")
    if not repo or not run_id:
        return ""
    try:
        data = get_json(
            f"https://api.github.com/repos/{repo}/actions/runs/{run_id}",
            headers=github_api_headers(),
        )
        return data.get("run_started_at") or ""
    except Exception:
        return ""


def gpu_count(runner):
    match = re.search(r"([0-9]+)gpu", runner or "")
    return int(match.group(1)) if match else None


def wheels_sha_raw():
    wheels = sorted(str(p) for p in Path("dist").glob("*.whl"))
    if not wheels:
        return ""
    p = subprocess.run(["sha256sum", *wheels], capture_output=True, text=True)
    return p.stdout.strip().replace("\n", "|").rstrip("|") if p.returncode == 0 else ""


def jax_packages_raw():
    p = subprocess.run(
        ["python3", "-m", "pip", "list", "--format=freeze"],
        capture_output=True,
        text=True,
    )
    lines = [
        line
        for line in p.stdout.splitlines()
        if re.match(r"^(jax|jaxlib)==", line)
        or ("pjrt" in line and "jax-rocm" in line)
        or ("plugin" in line and "jax-rocm" in line)
    ]
    return "|".join(lines)


def image_digest(image_name):
    if not image_name.startswith("ghcr.io/"):
        return ""
    repo = image_name.removeprefix("ghcr.io/").split(":", 1)[0]
    try:
        token = get_json(
            f"https://ghcr.io/token?service=ghcr.io&scope=repository:{repo}:pull"
        ).get("token", "")
        if not token:
            return ""
        headers = get_headers(
            f"https://ghcr.io/v2/{repo}/manifests/latest",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.docker.distribution.manifest.v2+json",
            },
        )
        return headers.get("Docker-Content-Digest", "")
    except Exception:
        return ""


def ordered_manifest(manifest):
    keys = (
        "schema_version",
        "run_started_at",
        "run_completed_at",
        "model_run_started_at",
        "model_run_completed_at",
        "github_run_url",
        "github_repository",
        "github_ref_name",
        "github_ref",
        "github_sha",
        "github_event_name",
        "github_run_id",
        "github_run_attempt",
        "github_run_number",
        "github_workflow",
        "github_job",
        "is_nightly",
        "jaxlib_version",
        "run_key",
        "combo",
        "target",
        "workload",
        "model_domain",
        "workload_type",
        "benchmark_goal",
        "runner",
        "python_version",
        "rocm_version",
        "rocm_tag",
        "gpu_count",
        "run_code",
        "cmp_code",
        "benchmark_repo_commit",
        "benchmark_command",
        "base_image_name",
        "base_image_digest",
        "jax_packages_raw",
        "wheels_sha_raw",
        "benchmark_configs_json",
        "results",
    )

    out = {key: manifest[key] for key in keys if key in manifest}
    for key, value in manifest.items():
        if key not in out:
            out[key] = value
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--rocm-version", required=True)
    parser.add_argument("--rocm-tag", required=True)
    parser.add_argument("--cmp-code", required=True, type=int)
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    started = github_run_started_at() or env("RUN_STARTED_AT")

    manifest = {
        "schema_version": 1,
        "run_started_at": started,
        "run_completed_at": utc_now(),
        "github_run_url": (
            f"https://github.com/{env('GITHUB_REPOSITORY')}"
            f"/actions/runs/{env('GITHUB_RUN_ID')}"
        ),
        "github_repository": env("GITHUB_REPOSITORY"),
        "github_ref_name": env("GITHUB_REF_NAME"),
        "github_ref": env("GITHUB_REF"),
        "github_sha": env("GITHUB_SHA"),
        "github_event_name": env("GITHUB_EVENT_NAME"),
        "github_run_id": env("GITHUB_RUN_ID"),
        "github_run_attempt": env("GITHUB_RUN_ATTEMPT"),
        "github_run_number": env("GITHUB_RUN_NUMBER"),
        "github_workflow": env("GITHUB_WORKFLOW"),
        "github_job": env("GITHUB_JOB"),
        "is_nightly": env("IS_NIGHTLY", "unknown"),
        "jaxlib_version": env("JAXLIB_VERSION"),
        "python_version": args.python_version,
        "rocm_version": args.rocm_version,
        "rocm_tag": args.rocm_tag,
        "gpu_count": gpu_count(args.runner),
        "runner": args.runner,
        "run_key": (
            f"{run_date(started)}_"
            f"{env('GITHUB_RUN_ID')}_"
            f"{env('GITHUB_RUN_ATTEMPT')}"
        ),
        "base_image_name": env("BASE_IMAGE_NAME"),
        "base_image_digest": env(
            "BASE_IMAGE_DIGEST",
            image_digest(env("BASE_IMAGE_NAME")),
        ),
        "wheels_sha_raw": wheels_sha_raw(),
        "jax_packages_raw": jax_packages_raw(),
    }

    for path in args.extra:
        manifest.update(json.loads(Path(path).read_text()))

    manifest["cmp_code"] = args.cmp_code

    Path(args.out).write_text(json.dumps(ordered_manifest(manifest), indent=2) + "\n")


if __name__ == "__main__":
    main()
