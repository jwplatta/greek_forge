.PHONY: test lint format build-call-model build-put-model build-models serve serve-dev clean help docker-build docker-run docker-stop docker-clean docker-deploy

# Docker configuration
DOCKER_IMAGE_NAME = greek-forge-api
DOCKER_TAG = latest
DOCKER_CONTAINER_NAME = greek-forge-api
DOCKER_PORT = 8000

help:
	@echo "Greek Forge - Available Make Targets"
	@echo "====================================="
	@echo ""
	@echo "Code Quality:"
	@echo "  make test          - Run all unit tests with pytest"
	@echo "  make lint          - Run ruff linting (with auto-fix)"
	@echo "  make format        - Run ruff formatting"
	@echo "  make check         - Run linting without auto-fix"
	@echo ""
	@echo "Model Training:"
	@echo "  make build-call-model  - Train and save CALL option model"
	@echo "  make build-put-model   - Train and save PUT option model"
	@echo "  make build-models      - Train and save both CALL and PUT models"
	@echo ""
	@echo "API Server:"
	@echo "  make serve         - Start FastAPI server (production mode)"
	@echo "  make serve-dev     - Start FastAPI server (development mode with auto-reload)"
	@echo ""
	@echo "Docker Deployment:"
	@echo "  make docker-build  - Build Docker image (requires models to be built first)"
	@echo "  make docker-run    - Run Docker container"
	@echo "  make docker-stop   - Stop running Docker container"
	@echo "  make docker-clean  - Remove Docker container and image"
	@echo "  make docker-deploy - Build models, create Docker image, and run container"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         - Remove Python cache files and artifacts"
	@echo "  make sync          - Sync uv dependencies"
	@echo ""

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check --fix src/ tests/

check:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

build-call-model:
	@echo "Training CALL option model..."
	uv run python -m src.utils.model_io --contract-type CALL

build-put-model:
	@echo "Training PUT option model..."
	uv run python -m src.utils.model_io --contract-type PUT

build-models: build-call-model build-put-model

serve:
	@echo "Starting Greek Forge API server..."
	uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000

serve-dev:
	@echo "Starting Greek Forge API server (development mode)..."
	uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

sync:
	uv sync

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

docker-build:
	@echo "Building Docker image: $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)"
	@if [ ! -d "./models/calls" ] || [ ! -d "./models/puts" ]; then \
		echo "Error: Models not found. Run 'make build-models' first."; \
		exit 1; \
	fi
	docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) .

docker-run:
	@echo "Starting Docker container: $(DOCKER_CONTAINER_NAME)"
	docker run -d \
		--name $(DOCKER_CONTAINER_NAME) \
		-p $(DOCKER_PORT):8000 \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)
	@echo "API available at http://localhost:$(DOCKER_PORT)"
	@echo "API docs at http://localhost:$(DOCKER_PORT)/docs"

docker-stop:
	@echo "Stopping Docker container: $(DOCKER_CONTAINER_NAME)"
	docker stop $(DOCKER_CONTAINER_NAME) 2>/dev/null || true
	docker rm $(DOCKER_CONTAINER_NAME) 2>/dev/null || true

docker-clean: docker-stop
	@echo "Removing Docker image: $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)"
	docker rmi $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) 2>/dev/null || true

docker-deploy: build-models docker-build docker-stop docker-run
	@echo "Docker deployment complete!"
	@echo "API is running at http://localhost:$(DOCKER_PORT)"
	@echo "View logs with: docker logs -f $(DOCKER_CONTAINER_NAME)"
	@echo "Stop with: make docker-stop"
