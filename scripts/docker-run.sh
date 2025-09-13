#!/bin/bash

# Build and run VGuard Model Docker container
# Usage: ./scripts/docker-run.sh [command]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
IMAGE_NAME="vguard-model"
CONTAINER_NAME="vguard-model-app"
DEFAULT_COMMAND="python pipeline.py"

# Parse arguments
COMMAND="${1:-$DEFAULT_COMMAND}"

echo "Building Docker image..."
docker build -t "$IMAGE_NAME" "$PROJECT_DIR"

echo "Running Docker container..."
docker run --rm \
    --name "$CONTAINER_NAME" \
    -v "$PROJECT_DIR/data:/app/data" \
    -v "$PROJECT_DIR/models:/app/models" \
    -v "$PROJECT_DIR/logs:/app/logs" \
    "$IMAGE_NAME" \
    $COMMAND

echo "Container execution completed."