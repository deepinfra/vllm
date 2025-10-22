# keep your CUDA12-based base
FROM localhost:30500/vllm:v0.10.2

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# pin the exact vLLM commit you want to mirror
ARG VLLM_COMMIT=a0003b56b0b822c52bb0f3035c164370a802e6f5
ENV VLLM_COMMIT=${VLLM_COMMIT}

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip --root-user-action=ignore && \
    pip install --no-cache-dir uv "setuptools-scm[toml]"

# remove any preinstalled vllm to avoid shadowing
RUN pip uninstall -y vllm || true

# install the compiled wheel first (unchanged)
RUN uv pip install --system vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}

# then overlay exact Git commit as editable, choosing correct package dir
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    git clone --depth=1 https://github.com/vllm-project/vllm.git /opt/vllm-src && \
    cd /opt/vllm-src && \
    git fetch --depth=1 origin ${VLLM_COMMIT} && \
    git checkout ${VLLM_COMMIT} && \
    if [ -f /opt/vllm-src/python/pyproject.toml ]; then \
        PKG_DIR=/opt/vllm-src/python; \
    elif [ -f /opt/vllm-src/pyproject.toml ]; then \
        PKG_DIR=/opt/vllm-src; \
    else \
        echo "pyproject.toml not found in repo root or python/; vLLM changed layout again. Aborting." >&2; exit 1; \
    fi && \
    uv pip install --system -e "${PKG_DIR}" --no-deps

# sanity check: show a file that exists in the repo but not in the slim wheel
# (replace the path below with something you actually expect from that commit)
RUN python - <<'PY'
import importlib, pkgutil, vllm, pathlib
p = pathlib.Path(vllm.__file__).parent
print("vllm lives at:", p)
print("Sample tree:")
for child in sorted(p.iterdir())[:30]:
    print(" -", child.name)
PY
