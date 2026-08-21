FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

WORKDIR /app

# 常用系统工具；Python 由 uv 安装正式版，避免 Ubuntu 22.04 的 3.11.0rc1
RUN apt-get update && apt-get install -y \
    curl \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 只使用 uv 管理的正式版 CPython 3.11，其中也包含 Triton JIT 需要的 Python.h。
ENV UV_MANAGED_PYTHON=1
RUN uv python install 3.11

# 先复制依赖定义
COPY pyproject.toml uv.lock ./

# 只装第三方依赖，利用 Docker cache
RUN uv sync --frozen --no-install-project

# 再复制项目源码
COPY . .

# 安装当前 myvllm package
RUN uv sync --frozen

CMD ["uv", "run", "python", "main.py"]
