# 🛡️ Financial Fraud Detection Platform v2.0

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

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
- **Transformers**: Sequential pattern recognition for transaction series
- **Graph Neural Networks (GNNs)**: Transaction network analysis using networkx and centrality
- **Variational Autoencoders (VAEs)**: Unsupervised anomaly detection based on reconstruction error

### 🔒 Privacy-Preserving ML
- **Federated Learning**: Decentralized model training with Flower
- **Differential Privacy**: Privacy guarantees with Opacus
- **Homomorphic Encryption**: Encrypted inference with TenSEAL

### ⚡ Real-time Processing
- **Kafka Streaming**: High-throughput event processing
- **WebSocket Alerts**: Real-time fraud notifications via FastAPI
- **Redis Cache**: Low-latency data access

### 📊 Monitoring & Explainability
- **Drift Detection**: Model performance and data drift monitoring with Evidently
- **SHAP Explanations**: Local model interpretability and feature importance
- **Prometheus + Grafana**: Metrics visualization

### 🚀 Production-Ready
- **FastAPI REST API**: High-performance endpoints for predictions and explanations
- **Docker & Kubernetes**: Containerized deployment
- **CI/CD Ready**: GitHub Actions workflows

---

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- pip or poetry
- Docker (optional)

### Clone the Repository
```bash
git clone https://github.com/danielandarge/financial-fraud-detection-model.git
cd financial-fraud-detection-model
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Generate Synthetic Data
```bash
python data/generate_data.py
```

### 2. Run Training Pipeline
```bash
python pipelines/training_pipeline.py --config configs/config.yaml
```

### 3. Start API Server
```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Access the API docs at: `http://localhost:8000/docs`

---

## 📁 Project Structure

```
financial-fraud-detection-model/
├── src/                      # Core source code
│   ├── data/                 # Data loaders and validators
│   ├── features/             # Graph and temporal feature engineering
│   ├── models/               # Model architectures
│   │   ├── deep_learning/    # TabNet, Transformers
│   │   └── anomaly/          # VAE, Isolation Forest
│   ├── monitoring/           # Drift detection & monitoring
│   ├── explainability/       # SHAP based explanations
│   └── privacy/              # Federated learning logic
├── api/                      # FastAPI application
├── pipelines/                # Training pipelines
├── configs/                  # Configuration YAML files
├── tests/                    # Unit and integration tests
├── data/                     # Raw and processed data
├── models/                   # Saved model artifacts
├── logs/                     # System and training logs
├── deploy/                   # Docker and K8s configurations
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
 