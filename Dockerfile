# syntax=docker/dockerfile:1.6
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for building Python C extensions + Pillow runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    pkg-config \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python deps first (layer caching)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -U pip setuptools wheel \
 && python -m pip install -r requirements.txt

# Copy project
COPY . /app

# Build C extensions (in container)
RUN python setup.py build_ext --inplace

# Runtime dirs
RUN mkdir -p logs sam_data/backups

EXPOSE 5005

# Default runtime env (can override at docker run time)
ENV PYTHONPATH=src/python:. \
    SAM_PROFILE=full \
    SAM_AUTONOMOUS_ENABLED=1 \
    SAM_UNBOUNDED_MODE=1 \
    SAM_RESTART_ENABLED=1 \
    SAM_STRICT_LOCAL_ONLY=1 \
    SAM_HOT_RELOAD=0

COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "src/python/complete_sam_unified.py", "--port", "5005"]

