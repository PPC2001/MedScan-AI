FROM python:3.12-slim

# Install system dependencies (required for some OCR/PDF libraries and database adapters)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for high-performance dependency installation
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency definition files
COPY pyproject.toml uv.lock README.md ./

# Copy source code and scripts
COPY medscan ./medscan
COPY scripts ./scripts
COPY start.py ./

# Install package and dependencies to system Python environment
RUN uv pip install --system -e .

# Create uploads directory (used for temporary file processing)
RUN mkdir -p uploads

# Expose default HTTP port
EXPOSE 8000

# Start both FastAPI and Celery worker
CMD ["python", "start.py"]
