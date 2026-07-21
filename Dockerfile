FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OCXKCD_MODEL_ROOT=/models \
    PADDLE_PDX_CACHE_HOME=/models/paddlex

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY models /models

RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2" \
    && pip install ".[cpu,ml]" \
    && ocxkcd download-models --model-root /models --output /tmp/ocxkcd-download \
    && rm -rf /tmp/ocxkcd-download

VOLUME ["/input", "/output"]
ENTRYPOINT ["ocxkcd"]
CMD ["--help"]
