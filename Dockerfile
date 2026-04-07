FROM python:3.11-slim

# Security: create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install system dependencies and remove them after pip install
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove build-essential

# Copy the entire package with correct ownership
COPY --chown=appuser:appuser . /app/fake_news_investigator/

# Set Python path so the package is importable
ENV PYTHONPATH=/app

# Security: drop to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check — judges' automated validation pings this
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run server
CMD ["uvicorn", "fake_news_investigator.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
