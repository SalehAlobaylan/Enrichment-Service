.PHONY: dev install install-dev test lint format docker-build docker-up download-models

dev:
	python -m uvicorn src.main:app --host 0.0.0.0 --port 5050 --reload

install:
	pip install -r requirements.txt
	playwright install chromium

install-dev:
	pip install -r requirements-dev.txt
	playwright install chromium

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-coverage:
	coverage run -m pytest tests/ -v
	coverage report -m
	coverage html

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

download-models:
	python scripts/download_models.py

docker-build:
	docker build -t enrichment-service .

docker-up:
	docker compose up --build
