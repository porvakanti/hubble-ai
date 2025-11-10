.PHONY: help install install-dev test lint format clean run-forecast run-ui

help:  ## Show this help message
	@echo "Hubble.AI - Treasury Cash-Flow Forecasting Platform"
	@echo "===================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package and dependencies
	pip install -e .

install-dev:  ## Install package with dev dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run test suite
	pytest tests/ -v --cov=cf_forecast --cov-report=term-missing --cov-report=html

test-fast:  ## Run fast tests only (skip slow/integration tests)
	pytest tests/ -v -m "not slow and not integration"

test-leakage:  ## Run data leakage tests only
	pytest tests/test_leakage.py -v

lint:  ## Run linters (ruff + mypy)
	ruff check src/ tests/
	mypy src/

format:  ## Format code with black
	black src/ tests/ app/

format-check:  ## Check code formatting
	black --check src/ tests/ app/

clean:  ## Clean build artifacts and cache files
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .coverage htmlcov/
	rm -rf .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

run-forecast:  ## Run full forecast pipeline (use ASOF=YYYY-MM-DD to set date)
	python -m cf_forecast.cli run --config configs/config.yml --asof $(or $(ASOF),2024-11-12)

run-ingest:  ## Ingest and validate data
	python -m cf_forecast.cli ingest --config configs/config.yml --asof $(or $(ASOF),2024-11-12)

run-features:  ## Generate feature set
	python -m cf_forecast.cli features --config configs/config.yml --asof $(or $(ASOF),2024-11-12)

run-train:  ## Train models
	python -m cf_forecast.cli train --config configs/config.yml --asof $(or $(ASOF),2024-11-12)

run-forecast-only:  ## Generate forecasts from trained models
	python -m cf_forecast.cli forecast --config configs/config.yml --asof $(or $(ASOF),2024-11-12)

run-reconcile:  ## Reconcile forecasts to hierarchy
	python -m cf_forecast.cli reconcile --config configs/config.yml --asof $(or $(ASOF),2024-11-12)

run-monitor:  ## Run monitoring checks
	python -m cf_forecast.cli monitor --config configs/config.yml --asof $(or $(ASOF),2024-11-12)

run-publish:  ## Publish forecasts (parquet + Excel)
	python -m cf_forecast.cli publish --config configs/config.yml --asof $(or $(ASOF),2024-11-12) --excel

run-ui:  ## Launch Streamlit UI
	streamlit run app/streamlit_app.py

docker-build:  ## Build Docker image
	docker build -t hubble-ai:latest .

docker-run:  ## Run Docker container
	docker run -p 8501:8501 -v $(PWD)/data:/app/data -v $(PWD)/configs:/app/configs hubble-ai:latest

# Development helpers
notebook:  ## Start Jupyter notebook server
	jupyter notebook notebooks/

check-data:  ## Quick data quality check
	python -c "from cf_forecast.validation import quick_check; quick_check('configs/config.yml')"

version:  ## Show version info
	python -c "import cf_forecast; print(cf_forecast.__version__)"
