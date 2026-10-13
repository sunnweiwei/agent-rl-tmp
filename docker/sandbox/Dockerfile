# Sandbox worker: FastAPI server + nsjail + the rootfs agent commands run in.
# Build from the repository root:
#   docker build -f docker/sandbox/Dockerfile -t agent-sandbox .

# ============================================================
# Stage 1: rootfs — the filesystem nsjail chroots into.
# Add whatever your tasks need (compilers, libraries, CLIs).
# ============================================================
FROM python:3.12-slim AS rootfs

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy pandas pyarrow pytest sympy

# ============================================================
# Stage 2: build nsjail from source. Same base distro as the worker
# so shared library versions match at runtime.
# ============================================================
FROM python:3.12-slim AS nsjail-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf bison ca-certificates flex gcc g++ git \
    libnl-route-3-dev libnl-3-dev libprotobuf-dev libseccomp-dev \
    make pkg-config protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

ARG NSJAIL_VERSION=3.4
RUN git clone --branch ${NSJAIL_VERSION} --depth 1 https://github.com/google/nsjail.git /nsjail \
    && cd /nsjail && make -j$(nproc) && strip nsjail

# ============================================================
# Stage 3: sandbox worker
# ============================================================
FROM python:3.12-slim AS worker

RUN apt-get update && apt-get install -y --no-install-recommends \
    tini libprotobuf32 libnl-route-3-200 libnl-3-200 libseccomp2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=nsjail-builder /nsjail/nsjail /usr/local/bin/nsjail
COPY --from=rootfs / /opt/sandbox/rootfs/default/

COPY docker/sandbox/requirements-server.txt /tmp/requirements-server.txt
RUN pip install --no-cache-dir -r /tmp/requirements-server.txt && rm /tmp/requirements-server.txt

COPY sandbox_server/ /opt/sandbox/server/sandbox_server/
COPY docker/sandbox/nsjail.cfg /etc/sandbox/nsjail.cfg

RUN mkdir -p /var/sandbox/workspaces /opt/sandbox/tasks

ENV SANDBOX_ENABLED_PROFILES=nsjail \
    NSJAIL_EXECUTABLE=/usr/local/bin/nsjail \
    SANDBOX_PROFILE_ROOTFS=/opt/sandbox/rootfs/default \
    SANDBOX_NSJAIL_CONFIG=/etc/sandbox/nsjail.cfg \
    SANDBOX_TASK_ROOT=/opt/sandbox/tasks \
    SANDBOX_WORKSPACE_ROOT=/var/sandbox/workspaces \
    PYTHONPATH=/opt/sandbox/server

EXPOSE 8080

ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "uvicorn", "sandbox_server.server:app", "--host", "0.0.0.0", "--port", "8080"]
