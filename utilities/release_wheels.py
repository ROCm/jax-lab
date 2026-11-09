#!/usr/bin/env python3
"""Download compatible ROCm wheels from the latest published ROCm/jax release."""

import json
import re
import sys
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename


def latest_release():
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "jax-lab"}
    request = urllib.request.Request(
        "https://api.github.com/repos/ROCm/jax/releases/latest", headers=headers
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        release = json.load(response)
    if release.get("draft") or release.get("prerelease"):
        raise ValueError("reference must be a published stable release")
    return release


def select_wheels(release, compatible_tags):
    urls = {asset["browser_download_url"] for asset in release.get("assets", [])
            if asset["name"].endswith(".whl")}
    # Some releases link externally hosted wheels in their release notes.
    urls.update(re.findall(r"https://[^\s<>\[\]()\"']+\.whl", release.get("body") or ""))
    candidates = {"jax-rocm7-plugin": [], "jax-rocm7-pjrt": []}
    for url in sorted(urls):
        name = unquote(Path(urlparse(url).path).name)
        try:
            distribution, version, _, tags = parse_wheel_filename(name)
        except ValueError:
            continue
        if distribution in candidates and tags.intersection(compatible_tags):
            candidates[distribution].append((name, url, version))
    selected = []
    for distribution, wheels in candidates.items():
        if len(wheels) != 1:
            raise ValueError(
                f"release {release.get('tag_name')}: expected one compatible "
                f"{distribution} wheel, found {len(wheels)}; "
                "inspect the release assets/notes before using it as a baseline"
            )
        selected.append(wheels[0])
    if selected[0][2] != selected[1][2]:
        raise ValueError("reference plugin and PJRT versions do not match")
    return selected


def main():
    destination = Path(sys.argv[1])
    release = latest_release()
    wheels = select_wheels(release, set(sys_tags()))
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Reference release: {release['tag_name']} ({release['html_url']})", file=sys.stderr)
    for name, url, _ in wheels:
        with urllib.request.urlopen(url, timeout=120) as response:
            with (destination / name).open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
    # Existing release installation uses matching upstream JAX packages.
    print(wheels[0][2].base_version)


if __name__ == "__main__":
    main()
