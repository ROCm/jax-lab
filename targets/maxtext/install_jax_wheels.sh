#!/usr/bin/env bash
set -euo pipefail

pip uninstall -y jax jaxlib jax-rocm7-plugin jax-rocm7-pjrt

# Install jax/jaxlib

PYTAG=cp312             # TODO: make parametric
ARCH=x86_64             # TODO: make parametric
PLATFORM=manylinux_2_27 # TODO: make parametric

gcloud storage cp gs://jax-nightly-artifacts/latest/jax*py3*none*any.whl .
gcloud storage cp gs://jax-nightly-artifacts/latest/jaxlib*${PYTAG}*${PLATFORM}*${ARCH}*.whl .

python3 -m pip install jaxlib-*.whl --no-deps
python3 -m pip install jax-*.whl --no-deps


# Install pjrt/plugin

PYTHON_MAJOR_MINOR=312
JAXCI_ROCM_VERSION=7
ROCM_WHEELS_BASE_URL="https://d22q5eopkfeftw.cloudfront.net"
RESOLVED_S3_URI=$(curl -fsSL "${ROCM_WHEELS_BASE_URL}/rocm-wheels/LATEST" | tr -d '[:space:]')
WHEELS_PATH="${RESOLVED_S3_URI#s3://jax-ci-amd/}"
WHEELS_URL="${ROCM_WHEELS_BASE_URL}/${WHEELS_PATH%/}"
LISTING=$(curl -fsSL "${WHEELS_URL}/")
FILES=$(echo "$LISTING" | grep -oE 'href="[^"]+\.whl"' | cut -d'"' -f2)
PJRT=$(echo "$FILES" | grep "jax_rocm${JAXCI_ROCM_VERSION}_pjrt-" | head -n1)
PLUGIN=$(echo "$FILES" | grep "jax_rocm${JAXCI_ROCM_VERSION}_plugin-" | grep "${PYTHON_MAJOR_MINOR}" | head -n1)

[[ -n "$PJRT" && -n "$PLUGIN" ]] || { echo "error: wheels not found"; exit 1; }
python3 -m pip install \
 "$WHEELS_URL/$PJRT" \
 "$WHEELS_URL/$PLUGIN"


# TE install, TODO
pip install https://github.com/ROCm/maxtext/releases/download/te-rocm-wheels-2026-04-13-098115728f7e/transformer_engine-2.12.0.dev0+9811572-1.mi355-cp312-cp312-linux_x86_64.whl
