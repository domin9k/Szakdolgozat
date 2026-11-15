# DermAI Scan — Production & Cloud Deployment Dockerfile
# Updated: Python 3.11, Debian 12 (Bookworm) compatible, dynamic cloud PORT support
FROM python:3.11-slim

# System environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8000

# Install system dependencies
# Note: Debian 12 replaced 'libgl1-mesa-glx' with 'libgl1'
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install Python dependencies (cached if requirements.txt unchanged)
COPY webapp/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 2. Copy pre-trained Keras models (cached if models unchanged)
COPY mobilenet_skin_model.keras /app/mobilenet_skin_model.keras
COPY efficientnet_skin_model.keras /app/efficientnet_skin_model.keras

# 3. Copy web application source (FastAPI backend + PWA frontend)
COPY webapp/ /app/webapp/

WORKDIR /app/webapp

EXPOSE 8000

# Healthcheck supporting dynamic PORT
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

# Start uvicorn with support for both local ($PORT=8000) and cloud environments ($PORT provided by platform)
CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
