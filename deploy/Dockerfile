# SAM-D Production Dockerfile
# Multi-stage build for minimized footprint

# --- Build Stage ---
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Compile C-Core Extensions
# We use --inplace for the builder, but we'll copy the built .so files to site-packages or keep PYTHONPATH
RUN python3 setup.py build_ext --inplace

# --- Runtime Stage ---
FROM python:3.11-slim

# Install runtime dependencies (minimal)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed python packages from builder
COPY --from=builder /root/.local /root/.local
# Copy project files (including built C extensions)
COPY --from=builder /app /app

# Set environment variables
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app/src/python:/app
ENV SAM_STRICT_LOCAL_ONLY=1
ENV SAM_AUTONOMOUS_ENABLED=1
ENV SAM_PROFILE=full
ENV PYTHONUNBUFFERED=1

# Expose Dashboard Port
EXPOSE 5005

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5005/api/health || exit 1

# Entry point
CMD ["python3", "src/python/complete_sam_unified.py", "--port", "5005"]
