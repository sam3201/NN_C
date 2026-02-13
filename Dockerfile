FROM python:3.13-slim

# System deps for building Python C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for caching
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -U pip setuptools wheel \
 && python -m pip install -r requirements.txt

# Copy project
COPY . /app

# Build C extensions
RUN rm -rf build/ \
 && python setup.py build_ext --inplace

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=src/python:.

EXPOSE 5005

CMD ["python", "src/python/complete_sam_unified.py", "--port", "5005"]

