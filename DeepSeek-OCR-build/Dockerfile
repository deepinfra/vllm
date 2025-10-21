FROM localhost:30500/vllm:v0.10.2

COPY deepseek-ocr.patch /deepseek-ocr.patch
RUN patch -p1 -i /deepseek-ocr.patch -d /usr/local/lib/python3.12/dist-packages/
