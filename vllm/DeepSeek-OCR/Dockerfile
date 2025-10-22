FROM localhost:30500/vllm:v0.8.5

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# In case curl isn't present in the base image
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Your patch first, as you had it
COPY deepseek-ocr.patch /deepseek-ocr.patch
RUN patch -p1 -i /deepseek-ocr.patch -d /usr/local/lib/python3.12/dist-packages/

# --- INSERTED BLOCK: install Torch 2.6.0 (CU118) and vLLM 0.8.5 CU118 wheel ---
RUN set -eux; \
    pip install --no-cache-dir --upgrade pip; \
    pip install --no-cache-dir \
        torch==2.6.0 \
        torchvision==0.21.0 \
        torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu118; \
    curl -L -o /tmp/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl \
        https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl; \
    pip install --no-cache-dir /tmp/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl; \
    rm -f /tmp/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
# --- END INSERTED BLOCK ---

# Then do your existing requirements for DeepSeek-OCR under vLLM
RUN pip install --no-cache-dir -r /usr/local/lib/python3.12/dist-packages/vllm/DeepSeek-OCR/requirements.txt

# And flash-attn as you requested
RUN pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation
