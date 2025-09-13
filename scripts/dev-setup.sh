#!/bin/bash

# Development setup with Docker Compose
# Usage: ./scripts/dev-setup.sh [up|down|logs]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

ACTION="${1:-up}"

cd "$PROJECT_DIR"

case "$ACTION" in
    "up")
        echo "Starting development environment..."
        docker-compose up -d
        echo "Development environment started!"
        echo "Access Jupyter notebook at: http://localhost:8888"
        ;;
    "down")
        echo "Stopping development environment..."
        docker-compose down
        echo "Development environment stopped."
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "notebook")
        echo "Starting development environment with notebook..."
        docker-compose --profile dev up -d
        echo "Jupyter notebook available at: http://localhost:8888"
        ;;
    *)
        echo "Usage: $0 [up|down|logs|notebook]"
        exit 1
        ;;
esac