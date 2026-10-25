# Docker image untuk aplikasi Streamlit (app.py).
# Cocok untuk Hugging Face Spaces (Docker) yang memakai port 7860.
#
# Build:  docker build -t smart-qc-buah .
# Run:    docker run -p 7860:7860 smart-qc-buah
#
# Versi FastAPI lama ada di Dockerfile.fastapi.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY . .
RUN mkdir -p /app/data

ENV MODEL_PATH=/app/models/best.pt \
    DB_PATH=/app/data/detections.db \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 7860

CMD ["sh", "-c", "streamlit run app.py --server.address 0.0.0.0 --server.port ${PORT:-7860}"]
