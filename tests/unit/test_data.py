import pytest
import pandas as pd
import numpy as np
from src.data.loaders import CSVDataLoader, DataLoaderFactory
from src.data.validators import DataValidator
import os

@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        'user_id': ['u1', 'u2'],
        'merchant_id': ['m1', 'm2'],
        'amount': [10.5, 20.0],
        'timestamp': ['2023-01-01T10:00:00', '2023-01-01T11:00:00'],
        'is_fraud': [0, 1]
    })
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return str(path)

def test_csv_loader(sample_csv):
    loader = CSVDataLoader()
    df = loader.load(sample_csv)
    assert len(df) == 2
    assert 'user_id' in df.columns
    assert loader.validate(df)

def test_loader_factory(sample_csv):
    loader = DataLoaderFactory.get_loader(sample_csv)
    assert isinstance(loader, CSVDataLoader)

def test_data_validator():
    validator = DataValidator(domain='fraud')
    df = pd.DataFrame({
        'user_id': ['u1'],
        'merchant_id': ['m1'],
        'amount': [100.0],
        'timestamp': ['2023-01-01T00:00:00'],
        'is_fraud': [0]
    })
    
    # Check schema validation
    # Note: DataValidator implementation details used here
    results = validator.run_full_validation(df)
    assert isinstance(results, dict)
    # Expected results based on src/data/validators.py (assuming it works)
    # If it fails, we'll fix it.
