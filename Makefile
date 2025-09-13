.PHONY: help build run test clean dev-up dev-down lint

# Default target
help:
	@echo "VGuard Model - Available commands:"
	@echo "  build      - Build Docker image"
	@echo "  run        - Run the model in Docker container"
	@echo "  test       - Run validation tests"
	@echo "  lint       - Run Python linting"
	@echo "  dev-up     - Start development environment"
	@echo "  dev-down   - Stop development environment"
	@echo "  clean      - Clean up Docker resources"

# Build Docker image
build:
	docker build -t vguard-model .

# Run the model
run: build
	docker run --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/logs:/app/logs \
		vguard-model

# Run tests
test:
	python tests/test_pipeline.py
	python -m py_compile pipeline.py

# Lint Python code
lint:
	@which black > /dev/null && black --check pipeline.py || echo "Install black for code formatting"
	@which flake8 > /dev/null && flake8 pipeline.py || echo "Install flake8 for linting"

# Development environment
dev-up:
	docker-compose up -d

dev-down:
	docker-compose down

# Clean up
clean:
	docker system prune -f
	docker rmi vguard-model 2>/dev/null || true

# Install dependencies locally
install:
	pip install -r requirements.txt

# Run locally without Docker
run-local: install
	python pipeline.py