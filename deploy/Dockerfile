# Search API image (CPU only). Build from the repo root:
#   docker compose -f deploy/docker-compose.yml build
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=4 \
    INDEX_DIR=/index/qwen3

WORKDIR /srv

COPY deploy/requirements-server.txt .
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-server.txt

COPY data.py build_index.py search.py rerank.py ./
COPY app ./app

# Model weights are downloaded on first start into the /hf volume (Qwen3 ~1.2 GB, once).
RUN useradd --create-home --uid 10001 app && mkdir -p /hf && chown app /hf
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
