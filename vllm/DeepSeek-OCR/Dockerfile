FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    FORCE_CUDA=1

# Bare minimum tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git curl ca-certificates build-essential ninja-build \
 && rm -rf /var/lib/apt/lists/*

# Make the repo the workspace
WORKDIR /workspace
RUN git clone --depth=1 https://github.com/deepseek-ai/DeepSeek-OCR.git
WORKDIR /workspace/DeepSeek-OCR

# Download the exact vLLM 0.8.5 CU118 wheel
RUN curl -L -o /tmp/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl \
  https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# 1) Upgrade pip; 2) install the exact PyTorch trio from cu118 index
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install \
      torch==2.6.0 \
      torchvision==0.21.0 \
      torchaudio==2.6.0 \
      --index-url https://download.pytorch.org/whl/cu118

# Install the downloaded vLLM wheel
RUN python3 -m pip install /tmp/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# Project requirements
RUN python3 -m pip install -r requirements.txt

# FlashAttention exactly as requested
RUN python3 -m pip install flash-attn==2.7.3 --no-build-isolation

# Land inside the repo by default
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
