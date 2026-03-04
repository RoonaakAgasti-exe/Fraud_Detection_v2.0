# Contributing to Financial Fraud Detection Platform

First off, thank you for considering contributing! It's people like you that make this project such a great tool for the community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community](#community)

---

## 📜 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to your.email@example.com.

---

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.9 or higher
- Git
- pip or poetry
- Docker (optional, for testing deployments)

### Fork and Clone

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/financial-fraud-detection-model.git
   cd financial-fraud-detection-model
   ```
3. **Add upstream** remote:
   ```bash
   git remote add upstream https://github.com/danielandarge/financial-fraud-detection-model.git
   ```
4. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
5. **Install dependencies**:
   ```bash
   pip install -e ".[all]"
   pip install pytest pytest-cov black isort mypy pre-commit
   ```
6. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

---

## 💡 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps to reproduce the problem**
* **Provide specific examples to demonstrate the steps**
* **Describe the behavior you observed and what behavior you expected**
* **Include error messages and stack traces if applicable**
* **Include system information (OS, Python version, etc.)**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title**
* **Provide a detailed description of the suggested enhancement**
* **Explain why this enhancement would be useful**
* **List some examples of how this enhancement would be used**
* **Specify which version of the project you're using**

### Your First Code Contribution

Unsure where to begin contributing? You can start by looking at these `good first issue` and `help wanted` issues:

* **Good first issues** are issues we think are good for beginners
* **Help wanted issues** are issues that need some help

### Pull Requests

* Fill in the required template
* Follow the coding standards
* Include tests when adding new features
* Update documentation accordingly
* Pass all CI/CD checks

---

## 🛠️ Development Setup

### Install Development Dependencies

```bash
pip install -e ".[all]"
pip install pytest pytest-cov black isort mypy pre-commit sphinx sphinx-rtd-theme
```

### Configure Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
pre-commit install
```

This will automatically run black, isort, flake8, and other checks before each commit.

### Set Up Development Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your local configuration
# Required environment variables:
# - DATABASE_URL
# - REDIS_HOST
# - KAFKA_BROKERS (optional for development)
```

---

## 🔄 Pull Request Process

### Before Submitting a PR

1. **Update your branch** with the latest main:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   git checkout your-branch
   git rebase main
   ```

2. **Run tests** to ensure everything passes:
   ```bash
   pytest --cov=src --cov-report=html
   ```

3. **Format your code**:
   ```bash
   black src/ api/ pipelines/ tests/
   isort src/ api/ pipelines/ tests/
   ```

4. **Check for type errors** (if using mypy):
   ```bash
   mypy src/ api/ pipelines/
   ```

5. **Build documentation** (if you made docs changes):
   ```bash
   cd docs
   make html
   ```

### PR Checklist

When submitting a PR, please ensure:

- [ ] Tests pass locally
- [ ] Code is formatted with black
- [ ] Imports are sorted with isort
- [ ] No mypy errors (if using type hints)
- [ ] Documentation is updated
- [ ] Commit messages follow conventions
- [ ] Branch is up to date with main

### Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types include:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning (white-space, formatting, etc)
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

Examples:
```
feat(models): add TabNet implementation
fix(api): resolve null pointer in prediction endpoint
docs(readme): update installation instructions
refactor(features): improve feature extraction pipeline
```

---

## 📏 Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

* **Line length**: Maximum 100 characters
* **Indentation**: 4 spaces (no tabs)
* **Encoding**: UTF-8
* **Imports**: Use isort to sort imports

Example:
```python
# Standard library imports
import os
from typing import List, Dict

# Third-party imports
import numpy as np
import pandas as pd
from fastapi import FastAPI

# Local imports
from src.models.tabnet import TabNet
from src.data.loaders import DataLoader
```

### Type Hints

Use type hints wherever possible:

```python
from typing import List, Dict, Optional, Union
import numpy as np

def predict_fraud(
    model: TabNet,
    data: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Union[int, float]]:
    """
    Predict fraud probability for given transaction data.
    
    Args:
        model: Trained TabNet model
        data: Input features array
        threshold: Classification threshold
        
    Returns:
        Dictionary containing prediction and probability
    """
    prediction = model.predict(data)
    return {
        "is_fraud": int(prediction > threshold),
        "probability": float(prediction)
    }
```

### Documentation

Write docstrings for all public functions and classes using Google style:

```python
class FraudDetector:
    """
    Main fraud detection class combining multiple models.
    
    Attributes:
        models: List of trained models for ensemble prediction
        config: Configuration dictionary
    """
    
    def detect(self, transaction_data: dict) -> dict:
        """
        Detect potential fraud in transaction.
        
        Args:
            transaction_data: Dictionary containing transaction details
            
        Returns:
            Dictionary with fraud probability and risk assessment
            
        Raises:
            ValidationError: If transaction data is invalid
        """
        pass
```

---

## 🧪 Testing

### Running Tests

Run all tests:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/unit/test_models.py
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html --cov-report=term-missing
```

### Writing Tests

Write tests for all new features and bug fixes:

```python
import pytest
from src.models.tabnet import TabNet

class TestTabNet:
    """Tests for TabNet model."""
    
    @pytest.fixture
    def model(self):
        """Create a TabNet model for testing."""
        return TabNet(n_d=8, n_a=8, input_dim=10, output_dim=1)
    
    def test_model_initialization(self, model):
        """Test model initializes correctly."""
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_prediction_shape(self, model, sample_data):
        """Test prediction output shape."""
        prediction = model.predict(sample_data)
        assert prediction.shape == (sample_data.shape[0], 1)
```

### Test Coverage

We aim for at least 80% code coverage. Check coverage:
```bash
coverage report --show-missing
```

---

## 📚 Documentation

### Building Documentation

```bash
cd docs
make html
```

View documentation locally:
```bash
open _build/html/index.html  # macOS
start _build/html/index.html  # Windows
xdg-open _build/html/index.html  # Linux
```

### Documentation Guidelines

* Use clear and concise language
* Include code examples where appropriate
* Keep documentation up to date with code changes
* Use proper markdown formatting
* Add type hints to function signatures

---

## 🌟 Community

### Communication

* **GitHub Issues**: For bug reports and feature requests
* **GitHub Discussions**: For questions and general discussions
* **Email**: your.email@example.com for sensitive matters

### Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Annual contributor highlights

### Questions?

Feel free to open an issue with the "question" label if you need help understanding the codebase or contribution process.

---

Thank you for contributing to Financial Fraud Detection Platform! 🎉

Your contributions make open-source projects like this possible and help the entire community fight financial fraud more effectively.
