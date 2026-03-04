"""
Quick Start Example - Fraud Detection Platform v2.0

This script demonstrates the complete workflow from data loading to model training
and API deployment.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_data
from src.data.validators import DataValidator
from src.features.graph_features import create_graph_features
from pipelines.training_pipeline import TrainingPipeline


def generate_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic fraud detection dataset for testing"""
    
    np.random.seed(42)
    
    # Generate features
    data = {
        'transaction_id': [f'tx_{i}' for i in range(n_samples)],
        'user_id': np.random.randint(1, 1000, n_samples),
        'merchant_id': np.random.randint(1, 500, n_samples),
        'amount': np.random.exponential(scale=100, size=n_samples).clip(1, 10000),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'is_fraud': np.random.binomial(n=1, p=0.02, size=n_samples)  # 2% fraud rate
    }
    
    df = pd.DataFrame(data)
    
    # Add some patterns
    fraud_mask = df['is_fraud'] == 1
    df.loc[fraud_mask, 'amount'] *= np.random.uniform(2, 5, fraud_mask.sum())
    
    return df


def main():
    """Main quick start workflow"""
    
    print("=" * 60)
    print("Financial Fraud Detection Platform v2.0 - Quick Start")
    print("=" * 60)
    
    # Step 1: Generate or load data
    print("\n[1/5] Loading data...")
    data_path = Path('data/raw/sample_transactions.csv')
    
    if not data_path.exists():
        print("Generating sample dataset...")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df = generate_sample_data(10000)
        df.to_csv(data_path, index=False)
        print(f"✓ Sample data saved to {data_path}")
    else:
        df = load_data(str(data_path))
        print(f"✓ Loaded {len(df)} transactions from {data_path}")
    
    # Step 2: Validate data
    print("\n[2/5] Validating data...")
    validator = DataValidator(domain='fraud')
    
    expected_columns = ['user_id', 'merchant_id', 'amount', 'timestamp', 'is_fraud']
    is_valid = validator.validate_schema(df, expected_columns)
    
    if is_valid:
        print("✓ Schema validation passed")
    
    # Step 3: Feature engineering
    print("\n[3/5] Engineering features...")
    
    try:
        df_featured = create_graph_features(
            df,
            user_col='user_id',
            merchant_col='merchant_id',
            amount_col='amount'
        )
        print(f"✓ Created {len(df_featured.columns)} features (including graph features)")
    except Exception as e:
        print(f"⚠ Graph features skipped: {e}")
        df_featured = df.copy()
    
    # Step 4: Train model
    print("\n[4/5] Training model...")
    
    pipeline = TrainingPipeline('configs/config.yaml')
    results = pipeline.run()
    
    if results.get('status') == 'success':
        print("✓ Training completed successfully!")
        print(f"  - AUC-ROC: {results['metrics']['auc_roc']:.4f}")
        print(f"  - F1-Score: {results['metrics']['f1_score']:.4f}")
        print(f"  - Time: {results['elapsed_time_seconds']:.2f}s")
    else:
        print(f"✗ Training failed: {results.get('error', 'Unknown error')}")
    
    # Step 5: Summary
    print("\n[5/5] Next Steps")
    print("=" * 60)
    print("""
Your fraud detection platform is ready! Here's what you can do next:

1. Start the API server:
   $ python -m uvicorn api.main:app --reload --port 8000

2. Test predictions:
   $ curl -X POST http://localhost:8000/predict \\
     -H "Content-Type: application/json" \\
     -d '{"user_id": "123", "merchant_id": "456", "amount": 100.50}'

3. View API documentation:
   Open http://localhost:8000/docs in your browser

4. Monitor model performance:
   Check logs/monitoring_report.json for drift analysis

5. Deploy with Docker:
   $ docker-compose up -d

For more examples, see the examples/ directory.
    """)
    print("=" * 60)


if __name__ == "__main__":
    main()
