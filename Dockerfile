FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire package
COPY . /app/fake_news_investigator/

# Set Python path so the package is importable
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "fake_news_investigator.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
