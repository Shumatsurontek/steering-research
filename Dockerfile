# ── Stage 1: Build frontend ──────────────────────────────────────────────────
FROM node:22-slim AS frontend-build

WORKDIR /app/web/frontend
COPY web/frontend/package.json web/frontend/package-lock.json* ./
RUN npm ci
COPY web/frontend/ ./
RUN npm run build


# ── Stage 2: Python runtime ─────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# System deps for torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps — install in layers for caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Web-specific deps
RUN pip install --no-cache-dir \
    fastapi==0.115.* \
    uvicorn[standard]==0.34.* \
    scikit-learn \
    sae-lens \
    transformer-lens

# Copy source
COPY src/ ./src/
COPY web/ ./web/

# Copy built frontend
COPY --from=frontend-build /app/web/frontend/dist ./web/frontend/dist

# Results directory (mount as volume for .pt files and SAE weights)
RUN mkdir -p /app/results
VOLUME /app/results

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Single process — uvicorn serves both API and static frontend
CMD ["uvicorn", "web.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning", "--no-access-log"]
