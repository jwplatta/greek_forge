FROM python:3.10-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY src/ ./src/
COPY README.md ./
COPY models/ ./models/

# Validate models exist
# Note: Models must be trained locally before building Docker image
# Run 'make build-models' to train models
RUN if [ ! -d "./models/calls" ] || [ ! -d "./models/puts" ]; then \
      echo "Error: Models not found in ./models/calls or ./models/puts"; \
      echo "Please run 'make build-models' before building the Docker image."; \
      exit 1; \
    fi

FROM python:3.10-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY --from=builder /app/src ./src
COPY --from=builder /app/README.md ./

COPY --from=builder /app/models ./models

# Create non-root user for security
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app

USER apiuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

CMD ["uv", "run", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]