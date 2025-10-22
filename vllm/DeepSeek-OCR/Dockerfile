FROM localhost:30500/vllm:v0.10.2

COPY deepseek-ocr.patch /deepseek-ocr.patch
RUN patch -p1 -i /deepseek-ocr.patch -d /usr/local/lib/python3.12/dist-packages/


RUN pip install -r /usr/local/lib/python3.12/dist-packages/vllm/DeepSeek-OCR/requirements.txt
RUN pip install flash-attn==2.7.3 --no-build-isolation
