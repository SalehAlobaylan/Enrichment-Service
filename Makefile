.PHONY: dev run install install-dev test test-unit test-integration test-coverage lint typecheck format docker-build docker-build-reranker docker-up download-models dependency-check export-requirements ensure-venv ensure-runtime ensure-dev

PYTHON ?= python3
UV ?= uv
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python

ensure-venv:
	@test -x "$(VENV_PYTHON)" || $(UV) venv --python "$(PYTHON)" "$(VENV)"

ensure-runtime: ensure-venv
	$(UV) sync --frozen --no-dev --no-install-project

ensure-dev: ensure-venv
	$(UV) sync --frozen --all-extras --no-install-project

run: ensure-runtime
	$(VENV_PYTHON) -m uvicorn src.main:app --host 0.0.0.0 --port $${PORT:-5050}

dev: ensure-dev
	$(VENV_PYTHON) -m uvicorn src.main:app --host 0.0.0.0 --port $${PORT:-5050} --reload

install: ensure-venv
	$(UV) sync --frozen --no-dev --no-install-project

install-dev: ensure-venv
	$(UV) sync --frozen --all-extras --no-install-project

test: ensure-dev
	$(VENV_PYTHON) -m pytest tests/ -v

test-unit: ensure-dev
	$(VENV_PYTHON) -m pytest tests/unit/ -v

test-integration: ensure-dev
	$(VENV_PYTHON) -m pytest tests/integration/ -v

test-coverage: ensure-dev
	$(VENV_PYTHON) -m coverage run -m pytest tests/ -v
	$(VENV_PYTHON) -m coverage report -m
	$(VENV_PYTHON) -m coverage html

lint: ensure-dev
	$(VENV_PYTHON) -m ruff check src/ tests/

typecheck: ensure-dev
	$(VENV_PYTHON) -m pyright

format: ensure-dev
	$(VENV_PYTHON) -m ruff format src/ tests/

download-models: ensure-runtime
	$(VENV_PYTHON) scripts/download_models.py

dependency-check:
	$(UV) lock --check

export-requirements:
	@echo "Compatibility exports are intentionally not tracked; use:"
	@echo "  $(UV) export --frozen --no-dev --no-emit-project --format requirements-txt"

docker-build:
	docker build --target api -t enrichment-service .

docker-build-reranker:
	docker build --target reranker -t enrichment-service-reranker .

docker-up:
	docker compose up --build
