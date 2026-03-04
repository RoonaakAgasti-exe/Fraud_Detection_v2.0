"""
Quick test script to verify the fraud detection model works
"""
import sys
from pathlib import Path
import pandas as pd
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loaders import load_data
from src.features.graph_features import create_graph_features

def test_model_prediction():
    """Test loading model and making predictions"""
    
    print("=" * 60)
    print("Testing Fraud Detection Model")
    print("=" * 60)
    
    # Load the trained model
    model_path = Path("models/fraud_detection_model.pkl")
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Please run the training pipeline first: py pipelines/training_pipeline.py")
        return
    
    print(f"✓ Loading model from {model_path}")
    model = joblib.load(model_path)
    print("✓ Model loaded successfully")
    
    # Load test data
    data_path = Path("data/raw/transactions.csv")
    if not data_path.exists():
        print(f"❌ Data not found at {data_path}")
        return
    
    print(f"\n✓ Loading test data from {data_path}")
    df = load_data(str(data_path))
    print(f"✓ Loaded {len(df)} samples")
    
    # Create features (simplified version)
    print("\n✓ Creating features...")
    
    # Add timestamp if not present
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.Timestamp.now()
    
    # Add merchant_id if not present  
    if 'merchant_id' not in df.columns:
        df['merchant_id'] = 'merch_001'
    
    try:
        df_featured = create_graph_features(df)
        print("✓ Graph features created")
    except Exception as e:
        print(f"⚠ Graph features skipped: {e}")
        df_featured = df.copy()
    
    # Add temporal features
    if 'timestamp' in df_featured.columns:
        df_featured['hour'] = pd.to_datetime(df_featured['timestamp']).dt.hour
        df_featured['day_of_week'] = pd.to_datetime(df_featured['timestamp']).dt.dayofweek
        df_featured['is_weekend'] = (df_featured['day_of_week'] >= 5).astype(int)
        print("✓ Temporal features created")
    
    # Add aggregation features
    if 'user_id' in df_featured.columns and 'amount' in df_featured.columns:
        user_stats = df_featured.groupby('user_id')['amount'].agg(['mean', 'std', 'max', 'min', 'count'])
        user_stats.columns = [f'user_{col}' for col in user_stats.columns]
        df_featured = df_featured.merge(user_stats.reset_index(), on='user_id', how='left')
        print("✓ Aggregation features created")
    
    # Prepare features for prediction
    exclude_cols = ['user_id', 'merchant_id', 'timestamp', 'transaction_id', 'is_fraud']
    feature_cols = [col for col in df_featured.columns 
                   if col != 'is_fraud' and col not in exclude_cols]
    
    X = df_featured[feature_cols].fillna(0)
    
    # Encode categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            X[col] = X[col].astype('category').cat.codes
        X = X.replace(-1, 0)
    
    print(f"✓ Prepared {len(feature_cols)} features")
    
    # Make predictions
    print("\n✓ Making predictions...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else predictions
    
    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    
    for idx in range(min(5, len(predictions))):
        actual = df_featured['is_fraud'].iloc[idx] if 'is_fraud' in df_featured.columns else 'N/A'
        print(f"\nTransaction {idx + 1}:")
        print(f"  Amount: ${df_featured['amount'].iloc[idx]:.2f}")
        print(f"  Category: {df_featured['merchant_category'].iloc[idx]}")
        print(f"  Location: {df_featured['location'].iloc[idx]}")
        print(f"  Predicted: {'FRAUD ⚠️' if predictions[idx] == 1 else 'Legitimate ✓'}")
        print(f"  Probability: {probabilities[idx]:.2%}")
        print(f"  Actual: {'FRAUD' if actual == 1 else 'Legitimate'}")
    
    print("\n" + "=" * 60)
    print(f"Total transactions analyzed: {len(predictions)}")
    print(f"Fraud detected: {sum(predictions)}")
    print(f"Legitimate: {len(predictions) - sum(predictions)}")
    print("=" * 60)
    print("\n✅ Model test completed successfully!")
    
    return predictions, probabilities

if __name__ == "__main__":
    test_model_prediction()
