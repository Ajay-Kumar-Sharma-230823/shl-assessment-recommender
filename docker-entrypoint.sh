#!/bin/bash
# ============================================================
# Docker Entrypoint Script
# ============================================================
set -e

echo "============================================"
echo "SHL Assessment Recommendation System"
echo "============================================"

# Run data pipeline if catalog doesn't exist
if [ ! -f "data/shl_catalog.json" ]; then
    echo "No catalog found. Running data pipeline..."
    python run_pipeline.py --all
fi

# Build vector index if not exists
if [ ! -d "vectorstore" ] || [ -z "$(ls -A vectorstore)" ]; then
    echo "No vector index found. Building index..."
    python run_pipeline.py --index
fi

echo "Starting FastAPI server..."
exec uvicorn app.main:app \
    --host "${APP_HOST:-0.0.0.0}" \
    --port "${APP_PORT:-8000}" \
    --workers 1 \
    --log-level "${LOG_LEVEL:-info}"
