"""
End-to-End Training Pipeline for Fraud Detection Models

Orchestrates data loading, preprocessing, feature engineering, model training,
and MLflow tracking in a unified pipeline.
"""

import sys
from pathlib import Path
import argparse
from typing import Dict, Any, Optional, List
from loguru import logger
import pandas as pd
import numpy as np
import joblib
import yaml
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_data
from src.data.validators import DataValidator
from src.features.graph_features import create_graph_features
from src.monitoring.drift_detector import create_monitor


class TrainingPipeline:
    """Complete training pipeline for fraud detection models"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize pipeline with configuration"""
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.pipeline_config = self.config.get('pipeline', {})
        self.model_config = self.config.get('model', {})
        self.data_config = self.config.get('data', {})
        
        # Initialize components
        self.validator = DataValidator(domain='fraud')
        self.monitor = create_monitor()
        
        # Tracking
        self.run_id = None
        self.metrics = {}
        
    def load_and_validate_data(self) -> pd.DataFrame:
        """Load and validate training data"""
        
        logger.info("Loading training data...")
        
        # Load data
        data_path = self.data_config.get('path', 'data/raw/train.csv')
        df = load_data(data_path)
        
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
        
        # Validate
        expected_columns = self.data_config.get('expected_columns', [])
        if expected_columns:
            self.validator.validate_schema(df, expected_columns)
        
        # Run full validation
        validation_results = self.validator.run_full_validation(df)
        
        logger.success(f"Validation completed: {sum(validation_results.values())}/{len(validation_results)} passed")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from raw data"""
        
        logger.info("Engineering features...")
        
        # Graph-based features
        if self.config.get('features', {}).get('graph', True):
            try:
                df = create_graph_features(
                    df,
                    user_col=self.data_config.get('user_col', 'user_id'),
                    merchant_col=self.data_config.get('merchant_col', 'merchant_id'),
                    amount_col=self.data_config.get('amount_col', 'amount')
                )
                logger.success("Graph features created")
            except Exception as e:
                logger.warning(f"Graph feature creation failed: {e}")
        
        # Temporal features
        if self.config.get('features', {}).get('temporal', True):
            time_col = self.data_config.get('time_col', 'timestamp')
            if time_col in df.columns:
                df['hour'] = pd.to_datetime(df[time_col]).dt.hour
                df['day_of_week'] = pd.to_datetime(df[time_col]).dt.dayofweek
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
                logger.success("Temporal features created")
        
        # Aggregation features
        if self.config.get('features', {}).get('aggregations', True):
            user_col = self.data_config.get('user_col', 'user_id')
            amount_col = self.data_config.get('amount_col', 'amount')
            
            if user_col in df.columns and amount_col in df.columns:
                # User-level aggregations
                user_stats = df.groupby(user_col)[amount_col].agg(['mean', 'std', 'max', 'min', 'count'])
                user_stats.columns = [f'user_{col}' for col in user_stats.columns]
                
                df = df.merge(user_stats.reset_index(), on=user_col, how='left')
                logger.success("Aggregation features created")
        
        return df
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Train selected model"""
        
        logger.info("Training model...")
        
        model_type = self.model_config.get('type', 'xgboost')
        
        if model_type == 'xgboost':
            import xgboost as xgb
            
            params = self.model_config.get('xgboost_params', {})
            model = xgb.XGBClassifier(**params)
            
        elif model_type == 'lightgbm':
            import lightgbm as lgb
            
            params = self.model_config.get('lightgbm_params', {})
            model = lgb.LGBMClassifier(**params)
            
        elif model_type == 'catboost':
            import catboost as cb
            
            params = self.model_config.get('catboost_params', {})
            model = cb.CatBoostClassifier(**params, verbose=0)
            
        elif model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            
            params = self.model_config.get('rf_params', {})
            model = RandomForestClassifier(**params)
            
        elif model_type == 'tabnet':
            from src.models.deep_learning.tabnet import TabNetClassifier
            import torch
            
            params = self.model_config.get('tabnet_params', {})
            model = TabNetClassifier(input_dim=X.shape[1], **params)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X.values)
            y_tensor = torch.LongTensor(y.values)
            
            model.fit(X_tensor, y_tensor, epochs=params.get('epochs', 50))
            
            return model
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train classical models
        model.fit(X, y)
        
        logger.success(f"{model_type} model trained successfully")
        
        return model
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model performance"""
        
        logger.info("Evaluating model...")
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, classification_report
        )
        
        # Predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = y_pred
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        }
        
        # Log to monitor
        self.monitor.evaluate_performance(y_test.values, y_pred, y_pred_proba)
        
        # Print detailed report
        logger.info("\n" + classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud']))
        
        self.metrics = metrics
        
        logger.success(f"Evaluation complete - AUC-ROC: {metrics['auc_roc']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def save_artifacts(self, model: Any, feature_engineer: Any = None):
        """Save model and related artifacts"""
        
        logger.info("Saving artifacts...")
        
        # Create directories
        models_dir = Path(self.config.get('artifacts', {}).get('models_dir', 'models'))
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = models_dir / self.config.get('artifacts', {}).get('model_name', 'fraud_detection_model.pkl')
        joblib.dump(model, model_path)
        logger.success(f"Model saved to {model_path}")
        
        # Save feature engineer if provided
        if feature_engineer:
            fe_path = models_dir / 'feature_engineer.pkl'
            joblib.dump(feature_engineer, fe_path)
            logger.success(f"Feature engineer saved to {fe_path}")
        
        # Save metrics
        metrics_path = models_dir / 'training_metrics.json'
        import json
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        logger.success(f"Metrics saved to {metrics_path}")
        
        # Save monitoring report
        report_path = Path('logs') / f'monitoring_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_path.parent.mkdir(exist_ok=True)
        self.monitor.save_report(str(report_path))
    
    def run(self) -> Dict[str, Any]:
        """Execute complete training pipeline"""
        
        logger.info("=" * 60)
        logger.info("Starting Fraud Detection Training Pipeline v2.0")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Load and validate data
            df = self.load_and_validate_data()
            
            # Step 2: Feature engineering
            df_featured = self.engineer_features(df)
            
            # Step 3: Prepare features and target
            target_col = self.data_config.get('target_col', 'is_fraud')
            exclude_cols = self.data_config.get('exclude_cols', ['user_id', 'merchant_id', 'timestamp'])
            
            feature_cols = [col for col in df_featured.columns 
                          if col != target_col and col not in exclude_cols]
            
            X = df_featured[feature_cols].fillna(0)
            
            # Encode categorical columns
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                logger.info(f"Encoding categorical columns: {list(categorical_cols)}")
                for col in categorical_cols:
                    X[col] = X[col].astype('category').cat.codes
                X = X.replace(-1, 0)  # Replace unknown categories with 0
            
            y = df_featured[target_col]
            
            logger.info(f"Prepared {len(feature_cols)} features for training")
            
            # Step 4: Split data
            from sklearn.model_selection import train_test_split
            
            test_size = self.pipeline_config.get('test_size', 0.2)
            random_state = self.pipeline_config.get('random_state', 42)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
            
            # Step 5: Train model
            model = self.train_model(X_train, y_train)
            
            # Step 6: Evaluate model
            metrics = self.evaluate_model(model, X_test, y_test)
            
            # Step 7: Save artifacts
            self.save_artifacts(model)
            
            # Summary
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            summary = {
                'status': 'success',
                'run_id': self.run_id or datetime.now().strftime('%Y%m%d_%H%M%S'),
                'metrics': metrics,
                'elapsed_time_seconds': elapsed_time,
                'num_train_samples': len(X_train),
                'num_test_samples': len(X_test),
                'num_features': len(feature_cols),
                'model_type': self.model_config.get('type', 'unknown')
            }
            
            logger.info("=" * 60)
            logger.info("Training Pipeline Completed Successfully!")
            logger.info(f"Total time: {elapsed_time:.2f} seconds")
            logger.info(f"Final AUC-ROC: {metrics['auc_roc']:.4f}")
            logger.info(f"Final F1-Score: {metrics['f1_score']:.4f}")
            logger.info("=" * 60)
            
            return summary
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e),
                'elapsed_time': (datetime.now() - start_time).total_seconds()
            }


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if args.verbose else "INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
    )
    
    # Run pipeline
    pipeline = TrainingPipeline(config_path=args.config)
    results = pipeline.run()
    
    # Exit with appropriate code
    sys.exit(0 if results.get('status') == 'success' else 1)


if __name__ == "__main__":
    main()
