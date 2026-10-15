FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System packages:
# - ffmpeg: audio extraction and keyframe sampling
# - tesseract-ocr + rus + eng: on-screen text and slide OCR
# - curl / ca-certificates: healthchecks and TLS
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-rus \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root runtime user and data directories
RUN groupadd -r ssmcp && useradd -r -g ssmcp -d /app -s /sbin/nologin ssmcp \
    && mkdir -p /data /data/whisper /data/uploads \
    && chown -R ssmcp:ssmcp /app /data

COPY pyproject.toml README.md LICENSE ./
COPY ssmcp ./ssmcp
COPY server.py ./server.py
COPY scripts ./scripts

RUN pip install --upgrade pip && pip install . \
    && chown -R ssmcp:ssmcp /app

USER ssmcp

ENV SSMCP_TRANSPORT=http \
    SSMCP_HOST=0.0.0.0 \
    SSMCP_PORT=8890 \
    SSMCP_WHISPER_MODEL=small \
    SSMCP_WHISPER_CACHE_DIR=/data/whisper \
    SSMCP_OCR_ENABLED=true \
    SSMCP_OCR_LANGUAGES=rus+eng \
    YT_QUOTA_STATE_PATH=/data/quota_state.json

EXPOSE 8890

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8890/health || exit 1

ENTRYPOINT ["ssmcp"]
