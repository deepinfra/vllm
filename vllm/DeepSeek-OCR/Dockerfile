# keep your CUDA12-based base
FROM localhost:30500/vllm:v0.10.2

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Set the vLLM commit hash
ENV VLLM_COMMIT=a0003b56b0b822c52bb0f3035c164370a802e6f5

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir uv

# Install vLLM from the commit-specific wheel index
RUN pip install vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}
