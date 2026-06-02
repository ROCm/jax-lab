#!/usr/bin/env bash
# Install JAX ROCm wheels for benchmark runs.

set -euo pipefail

JAXLIB_VERSION="${JAXLIB_VERSION:-head}"

python3 -m pip uninstall -y \
  jax \
  jaxlib \
  jax-rocm7-plugin \
  jax-rocm7-pjrt || true

mkdir -p dist
rm -f dist/*.whl

case "${JAXLIB_VERSION}" in
  head)
    gcloud config set auth/disable_credentials True

    PYTAG="cp312"               #TODO: make parametric
    PLATFORM="manylinux_2_27"   #TODO: make parametric
    ARCH="x86_64"               #TODO: make parametric

    gcloud storage cp gs://jax-nightly-artifacts/latest/jax*py3*none*any.whl dist/
    gcloud storage cp \
      "gs://jax-nightly-artifacts/latest/jaxlib*${PYTAG}*${PLATFORM}*${ARCH}*.whl" \
      dist/

    python3 -m pip install dist/jaxlib-*.whl --no-deps
    python3 -m pip install dist/jax-*.whl --no-deps

    ROCM_WHEELS_BASE_URL="https://d22q5eopkfeftw.cloudfront.net"
    RESOLVED_S3_URI="$(curl -fsSL "${ROCM_WHEELS_BASE_URL}/rocm-wheels/LATEST" | tr -d '[:space:]')"
    WHEELS_PATH="${RESOLVED_S3_URI#s3://jax-ci-amd/}"
    WHEELS_URL="${ROCM_WHEELS_BASE_URL}/${WHEELS_PATH%/}"

    echo "Downloading ROCm wheels from ${WHEELS_URL}..."

    LISTING="$(curl -fsSL "${WHEELS_URL}/")"
    FILES="$(echo "${LISTING}" | grep -oE 'href="[^"]+\.whl"' | cut -d'"' -f2)"

    PJRT="$(echo "${FILES}" | grep "jax_rocm7_pjrt-" | head -n1)"
    PLUGIN="$(echo "${FILES}" | grep "jax_rocm7_plugin-" | grep "cp312" | head -n1)"

    [[ -n "${PJRT}" && -n "${PLUGIN}" ]] || {
      echo "error: ROCm wheels not found" >&2
      exit 1
    }

    curl -fsSL -o "dist/${PJRT}" "${WHEELS_URL}/${PJRT}"
    curl -fsSL -o "dist/${PLUGIN}" "${WHEELS_URL}/${PLUGIN}"

    python3 -m pip install "dist/${PJRT}" "dist/${PLUGIN}"
    ;;

  pypi_latest)
    python3 -m pip download \
      jax \
      jaxlib \
      jax-rocm7-pjrt \
      jax-rocm7-plugin \
      --dest dist/

    python3 -m pip install dist/*.whl
    ;;

  release)

    # Requires: S3_WHEELS_URI env var
    [[ -n "${S3_WHEELS_URI:-}" ]] || {
      echo "error: S3_WHEELS_URI required for release mode" >&2
      exit 2
    }

    PYTAG="cp312"
    ROCM_WHEELS_BASE_URL="https://d22q5eopkfeftw.cloudfront.net"

    # Download ROCm plugin/pjrt from S3 via CloudFront
    WHEELS_PATH="${S3_WHEELS_URI#s3://jax-ci-amd/}"
    WHEELS_URL="${ROCM_WHEELS_BASE_URL}/${WHEELS_PATH%/}"
    echo "Downloading ROCm wheels from ${WHEELS_URL}..."

    LISTING="$(curl -fsSL "${WHEELS_URL}/")"
    FILES="$(echo "${LISTING}" | grep -oE 'href="[^"]+\.whl"' | cut -d'"' -f2)"
    PJRT="$(echo "${FILES}" | grep "jax_rocm7_pjrt-" | head -n1)"
    PLUGIN="$(echo "${FILES}" | grep "jax_rocm7_plugin-" | grep "${PYTAG}" | head -n1)"
    [[ -n "${PJRT}" && -n "${PLUGIN}" ]] || {
      echo "error: ROCm wheels not found at ${WHEELS_URL}" >&2
      exit 1
    }

    # Derive JAX version from plugin wheel filename (strip .postN)
    JAX_VERSION="$(echo "${PLUGIN}" | grep -oP '\d+\.\d+\.\d+' | head -1)"
    echo "Derived JAX version: ${JAX_VERSION}"

    # Download pinned jax/jaxlib from PyPI
    python3 -m pip download "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}" \
      --no-deps --dest dist/
    python3 -m pip install dist/jaxlib-*.whl --no-deps
    python3 -m pip install dist/jax-*.whl --no-deps

    # Install ROCm wheels
    curl -fsSL -o "dist/${PJRT}" "${WHEELS_URL}/${PJRT}"
    curl -fsSL -o "dist/${PLUGIN}" "${WHEELS_URL}/${PLUGIN}"
    python3 -m pip install "dist/${PJRT}" "dist/${PLUGIN}"
    ;;

  *)
    echo "unknown JAXLIB_VERSION: ${JAXLIB_VERSION}" >&2
    exit 2
    ;;
esac
