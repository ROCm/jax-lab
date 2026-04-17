ARG BASE_DOCKER=rocm/jax
FROM $BASE_DOCKER

# Install git.
RUN apt-get update && apt-get install -y git curl

# Install wheel as system packages
RUN pip3 install wheel --break-system-packages

RUN pip3 install uv --break-system-packages
