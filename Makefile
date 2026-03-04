# Makefile for Financial Fraud Detection Platform

.PHONY: help install dev train api test docker clean

# Default target
help:
	@echo "Financial Fraud Detection Platform v2.0"
	@echo ""
	@echo "Available targets:"
	@echo "  install     - Install base dependencies"
	@echo "  dev         - Install development dependencies"
	@echo "  all         - Install all dependencies including deep learning"
	@echo "  train       - Run training pipeline"
	@echo "  api         - Start API server"
	@echo "  test        - Run tests"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code with black"
	@echo "  docker-build    - Build Docker image"
	@echo "  docker-up       - Start Docker containers"
	@echo "  docker-down     - Stop Docker containers"
	@echo "  clean       - Clean build artifacts"
	@echo "  docs        - Generate documentation"
	@echo ""

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

dev: install
	pip install pytest pytest-cov black isort flake8 mypy

all: install
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	pip install tensorflow
	pip install flwr opacus tenseal
	pip install kafka-python redis
	pip install torch-geometric dgl networkx

# Training
train:
	python pipelines/training_pipeline.py --config configs/config.yaml

train-verbose:
	python pipelines/training_pipeline.py --config configs/config.yaml --verbose

# API
api:
	python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

api-prod:
	python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html

test-fast:
	pytest tests/unit -v

test-integration:
	pytest tests/integration -v

# Code quality
lint:
	flake8 src/ api/ pipelines/ --max-line-length=100 --ignore=E501,W503

format:
	black src/ api/ pipelines/ --line-length 100
	isort src/ api/ pipelines/ --profile black --line-length 100

type-check:
	mypy src/ --ignore-missing-imports

# Docker
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f fraud-api

# Data
download-sample-data:
	@echo "Downloading sample datasets..."
	mkdir -p data/raw
	@echo "Sample data generation implemented in examples/quickstart.py"

# Documentation
docs:
	mkdir -p docs/html
	sphinx-apidoc -o docs/source src/
	cd docs && make html
	@echo "Documentation generated in docs/html/"

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ docs/html/ mlruns/
	rm -rf logs/*.log
	@echo "Cleanup complete!"

# Quick start demo
demo:
	python examples/quickstart.py
