.PHONY: dev run install install-dev test test-unit test-integration test-coverage lint format docker-build docker-up download-models ensure-venv ensure-runtime ensure-dev

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

ensure-venv:
	@test -x "$(VENV_PYTHON)" || $(PYTHON) -m venv "$(VENV)"

ensure-runtime: ensure-venv
	@$(VENV_PYTHON) -c "import fastapi, uvicorn" >/dev/null 2>&1 || { \
		$(VENV_PYTHON) -m pip install --upgrade pip && \
		$(VENV_PIP) install -r requirements.txt && \
		$(VENV_PYTHON) -m playwright install chromium; \
	}

ensure-dev: ensure-venv
	@$(VENV_PYTHON) -c "import fastapi, uvicorn" >/dev/null 2>&1 || { \
		$(VENV_PYTHON) -m pip install --upgrade pip && \
		$(VENV_PIP) install -r requirements-dev.txt && \
		$(VENV_PYTHON) -m playwright install chromium; \
	}
	@$(VENV_PYTHON) -m pytest --version >/dev/null 2>&1
	@$(VENV_PYTHON) -m ruff --version >/dev/null 2>&1

run: ensure-runtime
	$(VENV_PYTHON) -m uvicorn src.main:app --host 0.0.0.0 --port $${PORT:-5050}

dev: ensure-dev
	$(VENV_PYTHON) -m uvicorn src.main:app --host 0.0.0.0 --port $${PORT:-5050} --reload

install: ensure-venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	$(VENV_PYTHON) -m playwright install chromium

install-dev: ensure-venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PIP) install -r requirements-dev.txt
	$(VENV_PYTHON) -m playwright install chromium

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

format: ensure-dev
	$(VENV_PYTHON) -m ruff format src/ tests/

download-models: ensure-runtime
	$(VENV_PYTHON) scripts/download_models.py

docker-build:
	docker build -t enrichment-service .

docker-up:
	docker compose up --build
