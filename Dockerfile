# Container image for the dancer/ PPO training package.
#
# Base on CUDA 12.4 runtime to match the torch cu124 pin in pyproject.toml.
# Driver requirement: >= 12.4.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=python3.12

# OS deps. python3.12 ships from the deadsnakes PPA on Ubuntu 22.04.
# libgl1 / libegl1 cover headless GL needs of mujoco-warp / Newton.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        libgl1 libegl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python

# uv (modern resolver, honours uv.lock).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer. pyroki lives in the `retarget` dep group (offline use only)
# and is intentionally excluded from the container — the trainer reads the
# pre-retargeted .npz directly.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-default-groups

# CLAUDE.md sharp edge: uv sync can pull a torch wheel without the right NCCL
# pin, manifesting as ImportError: libnccl.so.2 / undefined ncclCommWindow*.
# Force-reinstall the known-good combination.
RUN uv pip install --python /app/.venv/bin/python \
        nvidia-nccl-cu12==2.21.5 --force-reinstall \
    && uv pip install --python /app/.venv/bin/python \
        torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --reinstall

# Source + data. Order from least to most volatile for cache efficiency.
COPY retargeting/ ./retargeting/
COPY dancer/ ./dancer/
COPY data/robot/asimov-v1/ ./data/robot/asimov-v1/
COPY data/motions/asimov-v1-pyroki-full/ ./data/motions/asimov-v1-pyroki-full/

# Sanity check at build time.
RUN test -f data/robot/asimov-v1/xmls/asimov.xml \
        || (echo 'ERROR: data/robot/asimov-v1/xmls/asimov.xml missing.' && exit 1) \
    && test -f data/motions/asimov-v1-pyroki-full/dance1_subject3_keypoints_retargeted.npz \
        || (echo 'ERROR: motion .npz missing.' && exit 1)

ENV PATH="/app/.venv/bin:${PATH}"

# Azure ML overrides this at submit time; ENTRYPOINT here is for local
# `docker run ... key=value` testing.
ENTRYPOINT ["python", "-m", "dancer.train"]
CMD ["viewer.kind=null"]
