PYTHON ?= python

.DEFAULT_GOAL := help
.PHONY: help install lint format test test-unit test-integration coverage up down data features train pipeline api

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-18s %s\n", $$1, $$2}'

install:  ## Install dev dependencies and git hooks
	$(PYTHON) -m pip install -r requirements-dev.txt
	pre-commit install

lint:  ## Lint and check formatting
	ruff check src tests
	ruff format --check src tests

format:  ## Auto-fix lint issues and format code
	ruff check --fix src tests
	ruff format src tests

test:  ## Run the full test suite (integration tests skip without PostgreSQL)
	$(PYTHON) -m pytest

test-unit:  ## Run unit tests only
	$(PYTHON) -m pytest tests/unit

test-integration:  ## Run integration tests (requires PostgreSQL)
	$(PYTHON) -m pytest -m integration

coverage:  ## Run tests with a coverage report (fails under 70%)
	$(PYTHON) -m pytest --cov --cov-report=term-missing --cov-fail-under=70

up:  ## Start PostgreSQL and the MLflow server
	docker compose up -d --wait postgres mlflow

down:  ## Stop all services
	docker compose down

data:  ## Load raw CSVs from data/raw/ into PostgreSQL
	$(PYTHON) -m src.data.load

features:  ## Build rfm_features.parquet and features.parquet
	$(PYTHON) -m src.features.sql_features
	$(PYTHON) -m src.features.build

train:  ## Train, evaluate and register the model in MLflow
	$(PYTHON) -m src.models.train

pipeline: data features train  ## Run the full offline pipeline

api:  ## Build and start the API container
	docker compose up -d --build --wait api
