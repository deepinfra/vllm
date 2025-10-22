# keep your CUDA12-based base
FROM localhost:30500/vllm:v0.8.5

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY deepseek-ocr.patch /deepseek-ocr.patch
RUN patch -p1 -i /deepseek-ocr.patch -d /usr/local/lib/python3.12/dist-packages/

# Install Torch built for CUDA 12.1
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
      --index-url https://download.pytorch.org/whl/cu121

# Install vLLM 0.8.5 CUDA 12.1 wheel
RUN pip install --no-cache-dir \
      https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu121-cp38-abi3-manylinux1_x86_64.whl

# Then your OCR deps and flash-attn
RUN pip install --no-cache-dir -r /usr/local/lib/python3.12/dist-packages/vllm/DeepSeek-OCR/requirements.txt
RUN pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation
