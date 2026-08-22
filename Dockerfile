# Serving image for the GraphSAGE relational fraud detector.
#
# The service loads a precomputed serving bundle (graph tensors + calibrated
# scores + edge attention), so no training stack is needed at runtime — CPU
# torch is enough and the image stays small.
#
# The bundle and the feature parquet are NOT baked in; mount data/ at runtime:
#
#   docker build -t deepsentinel/graphsage .
#   docker run -p 8000:8000 -v "$PWD/data:/app/data:ro" deepsentinel/graphsage
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRAPHSAGE_API_HOST=0.0.0.0 \
    GRAPHSAGE_API_PORT=8000

WORKDIR /app

# CPU-only torch first (large, rarely changes) so it stays cached across builds.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
      torch-geometric pandas pyarrow scikit-learn "pydantic>=2" fastapi uvicorn

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
RUN pip install --no-cache-dir --no-deps -e .

# data/ is mounted at runtime (serving_bundle.pt is ~170 MB and changes per run).
COPY data/demo/ ./data/demo/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4)"

CMD ["python", "-m", "uvicorn", "graphsage.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
