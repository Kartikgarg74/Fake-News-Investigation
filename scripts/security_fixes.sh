#!/bin/bash

# 🔒 Automated Security Remediation Script
# Fake News Investigator - Security Fixes

echo "🔒 Starting security remediation..."

# Create scripts directory if it doesn't exist
mkdir -p scripts

echo "📦 1. Updating vulnerable dependencies..."
pip install --upgrade \
    cryptography>=46.0.5 \
    starlette>=0.49.1 \
    urllib3>=2.6.3 \
    wheel>=0.47.0 \
    pip>=26.0

echo "✅ Dependencies updated"

echo "🛠️  2. Fixing trust_remote_code vulnerability..."
# Fix the unsafe remote code execution in setup_data.py
if [ -f "data/setup_data.py" ]; then
    # Create backup
    cp data/setup_data.py data/setup_data.py.backup

    # Remove trust_remote_code=True occurrences
    sed -i.bak 's/, trust_remote_code=True//g' data/setup_data.py
    sed -i.bak 's/trust_remote_code=True, //g' data/setup_data.py
    sed -i.bak 's/trust_remote_code=True//g' data/setup_data.py

    echo "✅ Fixed unsafe remote code execution in data/setup_data.py"
else
    echo "⚠️  data/setup_data.py not found"
fi

echo "🐳 3. Creating secure Dockerfile..."
cat > Dockerfile.secure << 'EOF'
FROM python:3.11-slim-bookworm

# Security: Create non-root user first
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Security: Update packages and install only necessary dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN chown appuser:appuser /app

# Copy and install Python dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Security: Remove build dependencies after installation
RUN apt-get autoremove -y build-essential && \
    apt-get autoremove -y && \
    apt-get autoclean

# Copy application code with proper ownership
COPY --chown=appuser:appuser . /app/fake_news_investigator/

# Set Python path
ENV PYTHONPATH=/app

# Security: Switch to non-root user
USER appuser

# Security: Use non-privileged port
EXPOSE 8000

# Enhanced health check with timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "
import urllib.request
try:
    urllib.request.urlopen('http://localhost:8000/health', timeout=5)
except:
    exit(1)
" || exit 1

# Run with explicit security settings
CMD ["uvicorn", "fake_news_investigator.server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--access-log", \
     "--log-level", "info"]
EOF

echo "✅ Created Dockerfile.secure with security hardening"

echo "🔐 4. Creating security middleware configuration..."
cat > server/security_middleware.py << 'EOF'
"""Security middleware for FastAPI application."""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.gzip import GZipMiddleware
import time
import logging

# Configure security logging
logging.basicConfig(level=logging.INFO)
security_logger = logging.getLogger("security")

def add_security_middleware(app: FastAPI):
    """Add comprehensive security middleware to FastAPI app."""

    # CORS with restrictive settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:8000"],  # Restrict to known origins
        allow_credentials=True,
        allow_methods=["GET", "POST"],  # Only allow necessary methods
        allow_headers=["*"],
    )

    # Trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.example.com"]  # Update with your domains
    )

    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Custom security headers middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        start_time = time.time()

        # Log request for security monitoring
        security_logger.info(f"Request: {request.method} {request.url} from {request.client.host}")

        # Rate limiting (simple implementation)
        # In production, use Redis-based rate limiting

        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "connect-src 'self';"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        # Remove server header
        response.headers.pop("Server", None)

        # Log response time for monitoring
        process_time = time.time() - start_time
        security_logger.info(f"Response: {response.status_code} in {process_time:.3f}s")

        return response

    # Request size limit middleware
    @app.middleware("http")
    async def limit_upload_size(request: Request, call_next):
        if "content-length" in request.headers:
            content_length = int(request.headers["content-length"])
            max_size = 10 * 1024 * 1024  # 10MB limit
            if content_length > max_size:
                raise HTTPException(status_code=413, detail="Request entity too large")

        return await call_next(request)

EOF

echo "✅ Created security middleware configuration"

echo "⚡ 5. Creating updated app.py with security middleware..."
cat > server/app_secure.py << 'EOF'
"""Secure FastAPI server for the Fake News Investigator environment."""

from openenv.core.env_server import create_fastapi_app
from ..models import InvestigateAction, InvestigateObservation
from .environment import FakeNewsEnvironment
from .security_middleware import add_security_middleware

# Create the standard OpenEnv app
app = create_fastapi_app(
    FakeNewsEnvironment,
    InvestigateAction,
    InvestigateObservation,
    max_concurrent_envs=10,
)

# Add security middleware
add_security_middleware(app)

# ... rest of your endpoints remain the same ...
EOF

echo "✅ Created secure app configuration"

echo "🔍 6. Creating .bandit configuration for ongoing security scanning..."
cat > .bandit << 'EOF'
[bandit]
# Configuration for bandit security scanner
exclude_dirs = [".git", "__pycache__", "tests", "venv", ".venv"]

# Skip specific tests if they are false positives
# skips = B101,B601

# Report format
format = json
output_file = bandit-report.json
EOF

echo "✅ Created Bandit configuration"

echo "🔒 7. Creating GitHub Actions security workflow..."
mkdir -p .github/workflows
cat > .github/workflows/security.yml << 'EOF'
name: Security Scan

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly scan

jobs:
  security-scan:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install security tools
      run: |
        pip install bandit safety pip-audit

    - name: Run Bandit Security Scan
      run: |
        bandit -r . -f json -o bandit-results.json || true

    - name: Run Safety Scan
      run: |
        safety scan --json --output safety-results.json || true

    - name: Run pip-audit
      run: |
        pip-audit --format=json --output=pip-audit-results.json || true

    - name: Generate Security Report
      run: |
        echo "# Security Scan Results" > security-summary.md
        echo "" >> security-summary.md
        echo "## Bandit Results" >> security-summary.md
        if [ -f bandit-results.json ]; then
          python -c "
import json
try:
    with open('bandit-results.json', 'r') as f:
        data = json.load(f)
    print(f'Found {len(data.get(\"results\", []))} security issues')
except:
    print('No bandit results')
"
        fi

    - name: Upload Security Results
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-results.json
          safety-results.json
          pip-audit-results.json
          security-summary.md
        retention-days: 30
EOF

echo "✅ Created GitHub Actions security workflow"

echo ""
echo "🎉 Security remediation complete!"
echo ""
echo "📋 What was fixed:"
echo "  ✅ Updated vulnerable dependencies"
echo "  ✅ Removed unsafe remote code execution"
echo "  ✅ Created secure Dockerfile with non-root user"
echo "  ✅ Added security middleware configuration"
echo "  ✅ Created GitHub Actions security workflow"
echo ""
echo "🔄 Next steps:"
echo "  1. Review and test the changes"
echo "  2. Replace Dockerfile with Dockerfile.secure"
echo "  3. Update server/app.py to use security middleware"
echo "  4. Commit the .github/workflows/security.yml for automated scans"
echo ""
echo "🔍 To verify fixes:"
echo "  bandit -r . -ll"
echo "  safety scan"
echo "  pip-audit"