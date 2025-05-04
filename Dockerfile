FROM python:3.11-bookworm
ARG PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
# Add parameter to control whether to use Tsinghua Ubuntu mirror
ARG USE_MIRROR_UBUNTU="true"
ARG USE_CUDA="true"
ARG DEFAULT_VENV=/opt/.uv.venv
ENV PYTHON_VERSION=3.11
WORKDIR /app
COPY . .
RUN if [ "$USE_MIRROR_UBUNTU" = "true" ]; then \
    sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends software-properties-common gnupg ca-certificates apt-transport-https \
    git \
    curl \
    wget \
    && if [ "$USE_CUDA" = "true" ]; then \
    wget https://developer.download.nvidia.cn/compute/cuda/12.6.3/local_installers/cuda-repo-debian12-12-6-local_12.6.3-560.35.05-1_amd64.deb && \
    dpkg -i cuda-repo-debian12-12-6-local_12.6.3-560.35.05-1_amd64.deb && \
    cp /var/cuda-repo-debian12-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    add-apt-repository contrib && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends cuda-toolkit-12-6 \
    && wget "https://developer.download.nvidia.cn/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.6.0.74_cuda12-archive.tar.xz" \
    && tar -Jxvf cudnn-linux-x86_64-9.6.0.74_cuda12-archive.tar.xz -C /usr/local --strip=1 \
    && rm -f cudnn-linux-x86_64-9.6.0.74_cuda12-archive.tar.xz;\
    fi \
    && python3.11 -m pip install --upgrade pipx \
    && pipx install -i $PIP_INDEX_URL uv --global \
    && uv venv --seed $DEFAULT_VENV \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV UV_LINK_MODE=copy \
    PIP_INDEX_URL=$PIP_INDEX_URL \
    VIRTUAL_ENV=$DEFAULT_VENV \
    UV_PROJECT_ENVIRONMENT=$DEFAULT_VENV \
    UV_PYTHON=$DEFAULT_VENV/bin/python3 \
    UV_INDEX=$PIP_INDEX_URL \
    UV_DEFAULT_INDEX=$PIP_INDEX_URL \
    MODELSCOPE_CACHE=/app/models

RUN pip config set global.index-url $PIP_INDEX_URL && \
    pip config set global.trusted-host $(echo "$PIP_INDEX_URL" | sed -E 's|^https?://([^/]+).*|\1|') && \
    . $DEFAULT_VENV/bin/activate  && \
    uv sync -v --active --all-packages --default-index $PIP_INDEX_URL --index-strategy unsafe-best-match --prerelease=allow --no-build-isolation && \
    echo "/app" >> /opt/.uv.venv/lib/python${PYTHON_VERSION}/site-packages/.pth