FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by chromadb / pymupdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persistent data dirs (will be on mounted volume)
RUN mkdir -p data/chroma_store

# Copy knowledge docs to a path outside the volume so they survive the mount
RUN mkdir -p /app/kb_docs && \
    cp -r data/documents/. /app/kb_docs/ 2>/dev/null || true

EXPOSE 8000

ENV COMPANY_NAME="ML-Solutions"

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
