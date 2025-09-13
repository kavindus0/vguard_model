# CI/CD Docker Pipeline Documentation

## Overview

This repository implements a comprehensive CI/CD pipeline that containerizes the VGuard Model (paddy disease classification ML model) using Docker and GitHub Actions.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Code Push     │───▶│  GitHub Actions │───▶│ Container Reg.  │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Docker Build   │
                       │  & Test         │
                       └─────────────────┘
```

## Pipeline Components

### 1. GitHub Actions Workflow (`.github/workflows/docker-ci-cd.yml`)

**Triggers:**
- Push to `main`/`master` branches
- Pull requests to `main`/`master`
- Manual workflow dispatch

**Jobs:**

#### Test Job
- Sets up Python 3.11 environment
- Installs dependencies from `requirements.txt`
- Validates Python syntax
- Runs pipeline validation tests
- Checks TensorFlow installation

#### Build-and-Push Job
- Builds Docker image using `Dockerfile`
- Pushes to GitHub Container Registry (`ghcr.io`)
- Uses Docker BuildKit for optimization
- Implements layer caching for faster builds
- Tags images with branch names and SHA

#### Security-Scan Job
- Runs Trivy vulnerability scanner
- Uploads results to GitHub Security tab
- Only runs on non-PR events

### 2. Dockerfile

**Base Image:** `python:3.11-slim`

**Features:**
- Multi-layer build for optimization
- Security best practices (non-root user coming in v2)
- Health checks included
- Volume mounts for data, models, and logs
- SSL certificate handling for pip installs

**Build Process:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pipeline.py README.md .
CMD ["python", "pipeline.py"]
```

### 3. Docker Compose

**Services:**
- `vguard-model`: Main application
- `notebook`: Jupyter development environment (dev profile)

**Features:**
- Volume mounting for persistent data
- Development mode with Jupyter
- Port exposure for web interfaces

### 4. Container Registry

**Registry:** GitHub Container Registry (ghcr.io)
**Image:** `ghcr.io/kavindus0/vguard_model`

**Tags:**
- `latest`: Latest from main branch
- `main-<sha>`: Specific commit from main
- `pr-<number>`: Pull request builds

## Usage Instructions

### For Developers

#### Local Development
```bash
# Clone repository
git clone https://github.com/kavindus0/vguard_model.git
cd vguard_model

# Start development environment
make dev-up
# or
./scripts/dev-setup.sh up

# Access Jupyter notebook
./scripts/dev-setup.sh notebook
# Then open http://localhost:8888
```

#### Building and Testing
```bash
# Build Docker image
make build

# Run tests
make test

# Run the model
make run

# Clean up
make clean
```

### For DevOps/Production

#### Running from Registry
```bash
# Pull and run latest image
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  ghcr.io/kavindus0/vguard_model:latest
```

#### Environment Variables
- `PYTHONUNBUFFERED=1`: Ensure immediate stdout/stderr
- `PYTHONDONTWRITEBYTECODE=1`: Prevent .pyc files

#### Volume Mounts
- `/app/data`: Dataset storage
- `/app/models`: Trained model storage
- `/app/logs`: Application logs

## Security Considerations

### Image Security
- Trivy vulnerability scanning on every build
- Results uploaded to GitHub Security tab
- Base image regularly updated

### Access Control
- GitHub Container Registry with proper permissions
- Secrets managed through GitHub Secrets
- No hardcoded credentials in code

### Network Security
- Minimal port exposure
- SSL/TLS for all external communications
- Trusted PyPI hosts configuration

## Monitoring and Maintenance

### Health Checks
- Container health check verifies TensorFlow import
- 30-second intervals with 3 retries
- 30-second timeout per check

### Logs
- Application logs available via Docker logs
- Structured logging recommended for production
- Log rotation handled by Docker daemon

### Updates
- Dependabot for dependency updates
- Automated security patches
- Regular base image updates

## Best Practices

### Development
1. Use `make` commands for consistency
2. Test locally before pushing
3. Use development profile for Jupyter
4. Mount volumes for persistent data

### Production
1. Use specific image tags, not `latest`
2. Implement proper log aggregation
3. Set resource limits for containers
4. Use orchestration tools (K8s, Docker Swarm) for scaling

### CI/CD
1. Keep Docker images small
2. Use multi-stage builds when necessary
3. Implement proper testing at each stage
4. Cache layers effectively

## Troubleshooting

### Common Issues

**Build Failures:**
- Check `requirements.txt` for compatibility
- Verify Python version compatibility
- Check SSL certificate issues

**Runtime Errors:**
- Ensure proper volume mounts
- Check environment variables
- Verify data file accessibility

**Performance Issues:**
- Monitor resource usage
- Implement caching strategies
- Optimize TensorFlow for container environment

### Debugging

```bash
# Interactive debugging
docker run -it --rm vguard-model bash

# Check logs
docker logs <container_id>

# Inspect image
docker inspect vguard-model:latest
```

## Future Enhancements

### Version 2.0 Roadmap
- [ ] Multi-stage Docker builds
- [ ] Non-root user implementation
- [ ] Kubernetes deployment manifests
- [ ] Advanced monitoring with Prometheus
- [ ] Model versioning and registry
- [ ] Automated testing with test datasets
- [ ] GPU support for training
- [ ] Distributed training capabilities

### Integration Opportunities
- [ ] MLflow for experiment tracking
- [ ] Weights & Biases integration
- [ ] Apache Airflow for pipeline orchestration
- [ ] MinIO for object storage
- [ ] Redis for caching and queuing