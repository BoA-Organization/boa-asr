FROM nvcr.io/nvidia/pytorch:24.08-py3

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

ENV HF_HOME=/models/huggingface
ENV TRANSFORMERS_CACHE=/models/huggingface
ENV MODEL_ID=b1n1yam/shook-medium-amharic-2k

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY app.py .

RUN mkdir -p /models/huggingface

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=180s \
    --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD [
    "uvicorn",
    "app:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
    "--workers",
    "1"
]