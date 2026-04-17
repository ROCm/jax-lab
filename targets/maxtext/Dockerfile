# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_IMAGE=rocm/jax
FROM $BASE_IMAGE

WORKDIR /workspace
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

COPY targets/maxtext/install_jax.sh /tmp/install_jax.sh
RUN bash /tmp/install_jax.sh

COPY targets/maxtext/requirements.txt /tmp/requirements.txt
RUN if [ -f /tmp/requirements.txt ]; then python3 -m pip install -r /tmp/requirements.txt; fi