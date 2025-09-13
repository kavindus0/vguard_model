# VGuard Model

This repository contains a machine learning model that achieves 92% accuracy for paddy disease classification. The model uses EfficientNetB4 transfer learning with TensorFlow/Keras and includes complete CI/CD pipeline with Docker containerization.

## 🐳 Docker Usage

### Quick Start
```bash
# Build and run the model
./scripts/docker-run.sh

# Or using Docker directly
docker build -t vguard-model .
docker run --rm vguard-model
```

### Development Environment
```bash
# Start development environment
./scripts/dev-setup.sh up

# Start with Jupyter notebook
./scripts/dev-setup.sh notebook

# View logs
./scripts/dev-setup.sh logs

# Stop environment
./scripts/dev-setup.sh down
```

### Using Docker Compose
```bash
# Build and run the main application
docker-compose up

# Run with Jupyter notebook for development
docker-compose --profile dev up
```

## 🔄 CI/CD Pipeline

The repository includes a comprehensive GitHub Actions workflow that:

- **Tests**: Validates Python syntax and dependencies
- **Builds**: Creates optimized Docker images
- **Pushes**: Deploys to GitHub Container Registry
- **Scans**: Runs security vulnerability scans

### Container Registry

Images are automatically published to:
```
ghcr.io/kavindus0/vguard_model:latest
```

### Running from Registry
```bash
docker run --rm ghcr.io/kavindus0/vguard_model:latest
```

## 📋 Requirements

- Python 3.11+
- TensorFlow 2.12+
- Docker (for containerized deployment)
- Kaggle API credentials (for dataset access)

## 📚 Documentation

- [CI/CD Docker Pipeline Documentation](docs/CI-CD-DOCKER.md) - Comprehensive guide to the containerized deployment pipeline

## 🏗️ Local Development

### Without Docker
```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python pipeline.py
```

### With Docker
```bash
# Build image
docker build -t vguard-model .

# Run container with volume mounts
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  vguard-model
```

## Model Performance Screenshots

### Training Progress and Results

![Model Performance 1](Screenshot%202025-09-13%20at%2002-06-55%2098%20model%25%20-%20Colab.png)

![Model Performance 2](Screenshot%202025-09-13%20at%2002-07-16%2098%20model%25%20-%20Colab.png)

![Model Performance 3](Screenshot%202025-09-13%20at%2002-07-30%2098%20model%25%20-%20Colab.png)

![Model Performance 4](Screenshot%202025-09-13%20at%2002-08-30%2098%20model%25%20-%20Colab.png)

![Model Performance 5](Screenshot%202025-09-13%20at%2002-09-03%2098%20model%25%20-%20Colab.png)

![Model Performance 6](Screenshot%202025-09-13%20at%2002-09-14%2098%20model%25%20-%20Colab.png)

![Model Performance 7](Screenshot%202025-09-13%20at%2002-09-20%2098%20model%25%20-%20Colab.png)

## Key Achievements

- **92% Model Accuracy**: The model demonstrates high performance with 92% accuracy
- **Google Colab Implementation**: All training and evaluation conducted in Google Colab environment
- **Comprehensive Results**: Multiple screenshots showcase different aspects of model performance and metrics
- **Docker Containerization**: Complete containerized deployment with CI/CD pipeline
- **Automated Testing**: GitHub Actions workflow for continuous integration and deployment
