# CPU only. GPU runtime is deliberately not selected or installed implicitly.
FROM node:24.22.0-bookworm-slim@sha256:91acba91b842bec0bf7ea6f5e09c644c7ccad72c4996ed9616d861d1c7cd60d4 AS frontend
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a AS uv
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never UV_TORCH_BACKEND=cpu \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
    VISIONINSPECT_STATE_DIR=/app/outputs/service
COPY requirements-agent-linux.txt /app/requirements-agent-linux.txt
RUN uv venv /app/.venv && uv pip sync /app/requirements-agent-linux.txt --python /app/.venv/bin/python --torch-backend cpu --require-hashes
COPY visioninspect /app/visioninspect
COPY configs /app/configs
COPY --from=frontend /web/dist /app/frontend/dist
RUN mkdir -p /app/outputs /app/weights /app/data /app/.cache /app/.tmp && \
    groupadd --gid 10001 visioninspect && useradd --uid 10001 --gid 10001 --no-create-home visioninspect && \
    chown -R 10001:10001 /app
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD ["/app/.venv/bin/python", "-c", "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4)"]
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "visioninspect.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
