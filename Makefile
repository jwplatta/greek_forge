.PHONY: test lint format build-call-model build-put-model build-models clean help

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
	uv run python src/utils/model_io.py --contract-type CALL

build-put-model:
	@echo "Training PUT option model..."
	uv run python src/utils/model_io.py --contract-type PUT

build-models: build-call-model build-put-model

sync:
	uv sync

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
