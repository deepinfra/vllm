# keep your CUDA12-based base
FROM localhost:30500/vllm:v0.10.2

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VLLM_COMMIT=a0003b56b0b822c52bb0f3035c164370a802e6f5

# upgrade pip and install uv
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv

# optional: remove preinstalled vllm to avoid version shadowing/conflicts
# (safe even if not present)
RUN pip uninstall -y vllm || true

# install vLLM from the commit-specific wheels into the system site
RUN uv pip install --system vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}
