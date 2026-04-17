#!/usr/bin/env bash
set -e

# Usage: ./run.sh <branch> <config.yml>

BRANCH="$1"
CONFIG="$2"

git checkout "$BRANCH"

# to temporarily test JAX091 release
pip install jax==0.9.1 jaxlib==0.9.1
pip install jax-rocm7-plugin
pip install jax-rocm7-pjrt

uv tree --depth 1 \
  | sed 's/[├└──]//g' \
  | awk '{print $1}' \
  | grep -vi maxtext \
  > base-requirements.txt

# Clean Noise
grep -vi '^tensorflow' base-requirements.txt > /tmp/base.txt
echo "tensorflow==2.19.1" >> /tmp/base.txt
mv /tmp/base.txt base-requirements.txt

pip3 install -r base-requirements.txt

python3 -m MaxText.train "$CONFIG"