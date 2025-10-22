# keep your CUDA12-based base
FROM localhost:30500/vllm:v0.10.2

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# pin the exact vLLM commit you want to mirror
ARG VLLM_COMMIT=a0003b56b0b822c52bb0f3035c164370a802e6f5
ENV VLLM_COMMIT=${VLLM_COMMIT}

# tooling
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv git+https://github.com/pypa/setuptools_scm[toml]  && \
    apt-get update && apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# remove any preinstalled vllm to avoid shadowing
RUN pip uninstall -y vllm || true

# 1) install the prebuilt wheel (gives you the right CUDA binaries)
RUN uv pip install --system "vllm" \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}

# 2) overlay the **exact Git commit** as editable, without touching deps
#    This places a .pth to the repo, so Python files come from the commit,
#    while the compiled extension from the wheel remains in use.
RUN git clone --depth=1 https://github.com/vllm-project/vllm.git /opt/vllm-src && \
    cd /opt/vllm-src && \
    git fetch --depth=1 origin ${VLLM_COMMIT} && \
    git checkout ${VLLM_COMMIT} && \
    # vLLM’s Python package lives under the 'python' subdir in recent commits
    uv pip install --system -e /opt/vllm-src/python --no-deps

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
