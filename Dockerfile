FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    MODEL_ID=b1n1yam/shook-medium-amharic-2k

WORKDIR /app

# Install CUDA 13 PyTorch first.
RUN python -m pip install --upgrade pip && \
    python -m pip install \
        torch \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cu130

COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app.py .

RUN mkdir -p /models/huggingface && \
    python -c "import torch; print('Torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('Architectures:', torch.cuda.get_arch_list())"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]