# ============================================================
# SHL Assessment Recommendation System — Dockerfile
# ============================================================
FROM python:3.11-slim

# Metadata
LABEL maintainer="SHL Assessment Advisor"
LABEL description="Conversational SHL Assessment Recommendation System"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY scraper/ ./scraper/
COPY run_pipeline.py .
COPY .env .

# Create data and vectorstore directories
RUN mkdir -p data vectorstore

# Copy pre-built data if available (for faster startup)
# Comment out if running pipeline at startup
COPY data/ ./data/ 2>/dev/null || true
COPY vectorstore/ ./vectorstore/ 2>/dev/null || true

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Startup script
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["./docker-entrypoint.sh"]
