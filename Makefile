# Makefile for fraud-mlops
# Run `make help` to see available commands.
# Tabs (not spaces!) are required for indentation in Makefiles.

.PHONY: help install install-dev data clean lint format test notebook check

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies via uv
	uv sync

install-dev:  ## Install runtime + dev dependencies
	uv sync --extra dev
	uv run pre-commit install

data:  ## Download PaySim dataset from Kaggle
	bash scripts/download_data.sh

notebook:  ## Launch JupyterLab
	uv run jupyter lab

lint:  ## Run linter without fixing
	uv run ruff check .
	uv run mypy src/

format:  ## Auto-format code
	uv run ruff format .
	uv run ruff check --fix .

test:  ## Run tests
	uv run pytest

check: lint test  ## Run all checks (lint + test)

clean:  ## Clean build artifacts and caches
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
