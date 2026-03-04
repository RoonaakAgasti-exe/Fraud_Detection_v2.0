# 🛡️ Financial Fraud Detection Platform v1.0

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub stars](https://img.shields.io/github/stars/danielandarge/financial-fraud-detection-model.svg?style=social&label=Star)](https://github.com/danielandarge/financial-fraud-detection-model)
[![CI/CD](https://github.com/danielandarge/financial-fraud-detection-model/workflows/CI/CD/badge.svg)](.github/workflows/ci-cd.yml)

**Advanced AI/ML fraud detection with deep learning, real-time streaming, and privacy-preserving capabilities.**

---

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Features

### 🧠 Deep Learning Models
- **TabNet**: Attention-based tabular learning
- **Transformers**: Sequential pattern recognition
- **Graph Neural Networks (GNNs)**: Transaction network analysis
- **Variational Autoencoders (VAEs)**: Anomaly detection

### 🔒 Privacy-Preserving ML
- **Federated Learning**: Decentralized model training
- **Differential Privacy**: Privacy guarantees with Opacus
- **Homomorphic Encryption**: Encrypted inference with TenSEAL

### ⚡ Real-time Processing
- **Kafka Streaming**: High-throughput event processing
- **WebSocket Alerts**: Real-time fraud notifications
- **Redis Cache**: Low-latency data access

### 📊 Monitoring & Explainability
- **Drift Detection**: Model performance monitoring with Evidently
- **SHAP/LIME**: Model interpretability
- **Prometheus + Grafana**: Metrics visualization

### 🚀 Production-Ready
- **FastAPI REST API**: High-performance endpoints
- **Docker & Kubernetes**: Containerized deployment
- **CI/CD Ready**: GitHub Actions workflows

---

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- pip or poetry
- Docker (optional, for containerized deployment)

### Clone the Repository
```bash
git clone https://github.com/danielandarge/financial-fraud-detection-model.git
cd financial-fraud-detection-model
```

### Install Dependencies

**Basic Installation:**
```bash
pip install -r requirements.txt
```

**Using pip extras:**
```bash
# Install all features
pip install -e ".[all]"

# Install specific features
pip install -e ".[deep-learning]"
pip install -e ".[streaming]"
pip install -e ".[privacy]"
pip install -e ".[api]"
pip install -e ".[monitoring]"
```

**Development Installation:**
```bash
pip install -e ".[all]"
pip install pytest pytest-cov black isort mypy
```

---

## 🚀 Quick Start

### 1. Setup Configuration
```bash
# Copy example environment file
cp .env.example .env

# Edit configuration in configs/config.yaml
```

### 2. Run Training Pipeline
```bash
python pipelines/training_pipeline.py
```

### 3. Start API Server
```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Deploy with Docker
```bash
docker-compose up -d
```

Access the API docs at: `http://localhost:8000/docs`

---

## 📁 Project Structure

```
financial-fraud-detection-model/
├── src/                      # Core source code
│   ├── data/                 # Data loaders and validators
│   ├── features/             # Feature engineering
│   ├── models/               # Model architectures
│   │   ├── classical/        # Traditional ML models
│   │   ├── deep_learning/    # Deep learning models
│   │   ├── anomaly/          # Anomaly detection
│   │   └── ensemble/         # Ensemble methods
│   ├── monitoring/           # Drift detection & monitoring
│   ├── explainability/       # SHAP, LIME, Captum
│   └── privacy/              # Federated learning & privacy
├── api/                      # FastAPI application
├── pipelines/                # Training pipelines
├── configs/                  # Configuration files
├── tests/                    # Test suites
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── performance/          # Performance tests
├── deploy/                   # Deployment configurations
│   ├── docker/               # Docker configurations
│   └── k8s/                  # Kubernetes manifests
├── examples/                 # Example notebooks & scripts
├── docs/                     # Documentation
├── .github/                  # GitHub workflows & templates
├── requirements.txt          # All dependencies
├── pyproject.toml           # Project metadata
└── README.md                # This file
```

---

## 💡 Usage

### Training Models

**Single Model Training:**
```python
from pipelines.training_pipeline import train_model

# Train TabNet model
model = train_model(model_type="tabnet", config_path="configs/config.yaml")
```

**Batch Training:**
```bash
python pipelines/training_pipeline.py --model tabnet --data data/train.csv
```

### API Endpoints

The FastAPI application provides the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/batch_predict` | POST | Batch predictions |
| `/explain` | POST | Prediction with explanations |
| `/metrics` | GET | Prometheus metrics |
| `/ws/alerts` | WebSocket | Real-time fraud alerts |

**Example API Request:**
```python
import requests

transaction_data = {
    "amount": 1500.00,
    "merchant_category": "electronics",
    "location": "US",
    "time_delta": 3600
}

response = requests.post(
    "http://localhost:8000/predict",
    json=transaction_data
)

print(response.json())
# Output: {"fraud_probability": 0.87, "is_fraud": true, "risk_level": "high"}
```

### Real-time Streaming

Start Kafka consumer for real-time fraud detection:
```bash
python streaming/kafka_consumer.py --topic transactions
```

---

## 🏗️ Model Architecture

### TabNet Implementation
Our implementation uses attention mechanisms for feature selection:

```python
from src.models.deep_learning.tabnet import TabNet

model = TabNet(
    n_d=8, n_a=8, n_steps=3,
    gamma=1.3, lambda_sparse=1e-4,
    input_dim=50, output_dim=1
)
```

### Graph Neural Networks
Transaction networks are modeled using PyTorch Geometric:

```python
from torch_geometric.nn import GCNConv

class FraudGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 1)
```

---

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

```yaml
model:
  type: tabnet
  params:
    n_d: 8
    n_a: 8
    n_steps: 3

training:
  batch_size: 1024
  epochs: 100
  learning_rate: 0.02
  early_stopping: true

data:
  path: data/processed/
  test_split: 0.2
  random_state: 42

privacy:
  federated: true
  differential_privacy:
    epsilon: 1.0
    delta: 1e-5
```

---

## 🧪 Testing

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

Run specific test types:
```bash
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
```

---

## 🚢 Deployment

### Docker Deployment

Build and run:
```bash
docker build -t fraud-detection:latest .
docker run -p 8000:8000 fraud-detection:latest
```

### Kubernetes Deployment

Deploy to Kubernetes:
```bash
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
```

### Environment Variables

Set these in your `.env` file:
```env
DATABASE_URL=postgresql://user:pass@localhost:5432/fraud_db
REDIS_HOST=localhost
REDIS_PORT=6379
KAFKA_BROKERS=localhost:9092
MODEL_PATH=models/latest/
LOG_LEVEL=INFO
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Start for Contributors
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style
We use `black` and `isort` for consistent code formatting:
```bash
black src/ api/ pipelines/
isort src/ api/ pipelines/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- TabNet paper: [Arik, S. Ö., & Pfister, T. (2021)](https://arxiv.org/abs/1908.07442)
- Federated Learning framework: [Flower Labs](https://flower.ai/)
- Differential Privacy: [Opacus by Meta](https://opacus.ai/)
- FastAPI: [Sebastián Ramírez](https://fastapi.tiangolo.com/)

---
 