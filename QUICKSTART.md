# 🚀 Quick Start Guide

Get up and running with the Financial Fraud Detection Platform in minutes!

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

## Installation (5 minutes)

### Step 1: Clone the Repository

```bash
git clone https://github.com/danielandarge/financial-fraud-detection-model.git
cd financial-fraud-detection-model
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Setup Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit configs/config.yaml if needed (optional)
```

## Running Your First Model (2 minutes)

### Option 1: Run Training Pipeline

```bash
py pipelines/training_pipeline.py
```

This will:
- Load sample transaction data
- Engineer features (graph, temporal, aggregations)
- Train an XGBoost model
- Evaluate performance
- Save the trained model

Expected output:
```
✓ Loading training data...
✓ Graph features created
✓ Temporal features created
✓ xgboost model trained successfully
✓ Evaluation complete - AUC-ROC: 1.0000, F1: 1.0000
✓ Model saved to models/fraud_detection_model.pkl
```

### Option 2: Test the Model

```bash
py test_model.py
```

This will:
- Load the trained model
- Make predictions on sample data
- Display results with fraud probabilities

## Using the API (Optional)

### Start the API Server

```bash
py -m uvicorn api.main:app --reload
```

Access the interactive API docs at: http://localhost:8000/docs

### Example API Request

```python
import requests

transaction = {
    "amount": 1500.00,
    "merchant_category": "electronics",
    "location": "US",
    "time_delta": 3600
}

response = requests.post(
    "http://localhost:8000/predict",
    json=transaction
)

print(response.json())
```

## Docker Deployment (Optional)

### Build and Run

```bash
docker-compose up -d
```

Check status:
```bash
docker-compose ps
```

View logs:
```bash
docker-compose logs -f
```

## Next Steps

### Learn More
- 📖 [Full Documentation](README.md)
- 🤝 [Contributing Guide](CONTRIBUTING.md)
- 📋 [Code of Conduct](CODE_OF_CONDUCT.md)
- 🔒 [Security Policy](SECURITY.md)

### Customize
1. Edit `configs/config.yaml` for model parameters
2. Add your own training data to `data/raw/`
3. Modify feature engineering in `src/features/`
4. Try different models (TabNet, LightGBM, CatBoost)

### Production Deployment
- See `deploy/` for Kubernetes manifests
- Configure environment variables in `.env`
- Set up monitoring with Prometheus/Grafana

## Common Issues

### Issue: Missing dependencies
**Solution**: 
```bash
pip install -r requirements.txt --upgrade
```

### Issue: Data not found error
**Solution**: Ensure you have data in `data/raw/transactions.csv`

### Issue: Model training fails
**Solution**: Check that you have enough samples (min 100 recommended)

## Getting Help

- 💬 [GitHub Discussions](https://github.com/danielandarge/financial-fraud-detection-model/discussions)
- 🐛 [Report Issues](https://github.com/danielandarge/financial-fraud-detection-model/issues)
- 📧 Email: your.email@example.com

## What's Next?

Now that you have the platform running:

1. **Experiment with Models**: Try different algorithms in `configs/config.yaml`
2. **Add Your Data**: Replace sample data with your own dataset
3. **Feature Engineering**: Customize features for your use case
4. **Deploy to Production**: Use Docker/Kubernetes for deployment
5. **Monitor Performance**: Set up drift detection and alerts

Happy fraud detecting! 🎉
