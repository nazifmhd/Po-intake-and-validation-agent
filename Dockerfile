FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/app/storage/models

# tesseract = local OCR for scanned POs (the vision LLM is the primary path; OCR text is extra evidence)
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# dependencies first, pinned, so editing code doesn't reinstall them
COPY pyproject.toml constraints.txt README.md ./
RUN mkdir app && touch app/__init__.py \
    && pip install ".[sheets]" -c constraints.txt \
    && pip uninstall -y po-agent && rm -rf app build *.egg-info
COPY app ./app

COPY review_ui ./review_ui
COPY data ./data
COPY scripts ./scripts
COPY workflows ./workflows
COPY tests ./tests

EXPOSE 8000 8501

CMD ["sh", "-c", "python -m app.cli kb build --if-missing && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
