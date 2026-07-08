# syntax=docker/dockerfile:1.7

FROM runpod/base:1.0.7-cuda1281-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/huggingface_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    UV_INSTALL_DIR=/usr/local/bin \
    UV_PROJECT_ENVIRONMENT=/opt/personaplex-runpod/.venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/opt/personaplex-runpod/.venv/bin:/usr/local/bin:${PATH}"

WORKDIR /opt/personaplex-runpod

RUN apt-get update --yes \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential \
        libportaudio2 \
        libsndfile1 \
        pkg-config \
        python3 \
        python3-dev \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh

COPY pyproject.toml uv.lock ./
COPY moshi/pyproject.toml ./moshi/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-dev --no-install-workspace --compile-bytecode

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-dev --compile-bytecode \
    && chmod +x docker/runpod-start.sh docker/app-start.sh

EXPOSE 8888 8998

CMD ["/opt/personaplex-runpod/docker/runpod-start.sh"]
