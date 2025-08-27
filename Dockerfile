# ================================
# Production-Ready Multi-Stage Dockerfile for CodeSage MCP Server
# ================================

# -------------------------------
# Builder Stage: Compile dependencies and create virtual environment
# -------------------------------
FROM python:3.12-slim-bookworm AS builder

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# -------------------------------
# Runtime Stage: Lightweight production image
# -------------------------------
FROM python:3.12-slim-bookworm AS runtime

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    # Required for some Python packages
    libgomp1 \
    libatlas-base-dev \
    # Security: Remove package manager after use
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/*

# Create non-root user for security
RUN groupadd --gid 1000 codesage && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash codesage

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code with proper ownership
COPY --chown=codesage:codesage . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/.codesage && \
    mkdir -p /app/logs && \
    chown -R codesage:codesage /app

# Switch to non-root user
USER codesage

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/')" || exit 1

# Environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    UVICORN_WORKERS=1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

# Expose port
EXPOSE 8000

# Default command with production settings
CMD ["uvicorn", "codesage_mcp.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--access-log", \
     "--log-level", "info"]

# ================================
# Development Stage (Optional)
# ================================
FROM runtime AS development

# Switch back to root for development
USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    ruff \
    pre-commit

# Switch back to codesage user
USER codesage

# Development command
CMD ["uvicorn", "codesage_mcp.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--log-level", "debug"]
