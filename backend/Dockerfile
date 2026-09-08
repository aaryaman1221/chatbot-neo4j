# syntax=docker/dockerfile:1
# =============================================================================
# backend/Dockerfile — multi-stage build for the GraphRAG backend.
#   target: api     → lightweight FastAPI serving image (default)
#   target: ingest  → full pipeline image (tree-sitter, PyGithub, …) for
#                     backend_ingest.py and friends, run as one-off jobs.
#
# Build context is the REPO ROOT so that both backend/ and ingest/ are visible:
#   docker build -f backend/Dockerfile --target api  -t graphrag-api .
#   docker build -f backend/Dockerfile --target ingest -t graphrag-ingest .
# =============================================================================

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# ── API image — serving deps only, no build toolchain required ───────────────
FROM base AS api
COPY backend/requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
COPY backend/ ./backend/
RUN useradd --create-home --uid 10001 appuser
USER appuser
WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Ingest image — full pipeline dependencies ───────────────────────────────
FROM base AS ingest
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
COPY ingest/ ./ingest/
COPY backend_ingest.py backfill_embeddings.py enrich_semantic_edges.py \
     migrate_calls_edges.py patch_function_names.py ./
CMD ["python", "backend_ingest.py", "--help"]
