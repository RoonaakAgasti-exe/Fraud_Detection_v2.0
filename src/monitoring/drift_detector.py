"""
Model Drift Detection and Performance Monitoring

Detect data drift, concept drift, and model performance degradation
using statistical tests and metrics tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime


class DriftDetector:
    """Detect various types of drift in fraud detection models"""
    
    def __init__(self, reference_data: pd.DataFrame = None):
        self.reference_data = reference_data
        self.drift_results = {}
        
    def set_reference(self, data: pd.DataFrame):
        """Set reference dataset for drift comparison"""
        self.reference_data = data
        logger.info("Reference data set for drift detection")
    
    def population_stability_index(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change
        """
        
        # Create bins based on expected distribution
        bin_edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
        bin_edges[-1] += 1e-6  # Ensure max value is included
        
        # Bin the data
        expected_bins = np.digitize(expected, bin_edges[:-1]) - 1
        actual_bins = np.digitize(actual, bin_edges[:-1]) - 1
        
        # Calculate percentages
        expected_pct = np.bincount(expected_bins, minlength=bins) / len(expected)
        actual_pct = np.bincount(actual_bins, minlength=bins) / len(actual)
        
        # Avoid division by zero
        expected_pct = np.clip(expected_pct, 1e-6, 1)
        actual_pct = np.clip(actual_pct, 1e-6, 1)
        
        # Calculate PSI
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        
        return psi
    
    def kolmogorov_smirnov_test(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, bool]:
        """
        Two-sample Kolmogorov-Smirnov test
        
        Returns:
            statistic: KS statistic
            has_drift: True if distributions are significantly different
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        
        # Significant at 0.05 level
        has_drift = p_value < 0.05
        
        return statistic, has_drift
    
    def wasserstein_distance(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> float:
        """Calculate Earth Mover's Distance (Wasserstein distance)"""
        return stats.wasserstein_distance(reference, current)
    
    def jensen_shannon_divergence(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 50
    ) -> float:
        """Calculate Jensen-Shannon divergence between distributions"""
        
        # Create histograms
        hist_ref, bin_edges = np.histogram(reference, bins=bins, density=True)
        hist_cur, _ = np.histogram(current, bins=bin_edges, density=True)
        
        # Normalize
        hist_ref = hist_ref / hist_ref.sum()
        hist_cur = hist_cur / hist_cur.sum()
        
        # Calculate JS divergence
        js_div = stats.entropy(hist_ref, (hist_ref + hist_cur) / 2) + \
                 stats.entropy(hist_cur, (hist_ref + hist_cur) / 2)
        
        return js_div / 2  # Normalize to [0, 1]
    
    def detect_feature_drift(
        self,
        current_data: pd.DataFrame,
        threshold_psi: float = 0.1,
        threshold_ks: float = 0.05
    ) -> Dict[str, Dict]:
        """
        Detect drift for all features
        
        Returns:
            Dictionary with drift information for each feature
        """
        
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")
        
        drift_info = {}
        
        common_cols = set(self.reference_data.columns) & set(current_data.columns)
        
        for col in common_cols:
            ref_col = self.reference_data[col].dropna().values
            cur_col = current_data[col].dropna().values
            
            if len(ref_col) == 0 or len(cur_col) == 0:
                continue
            
            # Skip non-numeric columns
            if not np.issubdtype(ref_col.dtype, np.number):
                continue
            
            # Calculate metrics
            psi = self.population_stability_index(ref_col, cur_col)
            ks_stat, ks_drift = self.kolmogorov_smirnov_test(ref_col, cur_col)
            ws_dist = self.wasserstein_distance(ref_col, cur_col)
            js_div = self.jensen_shannon_divergence(ref_col, cur_col)
            
            # Determine if drift occurred
            has_drift = psi > threshold_psi or ks_drift
            
            drift_info[col] = {
                'psi': psi,
                'ks_statistic': ks_stat,
                'wasserstein_distance': ws_dist,
                'js_divergence': js_div,
                'has_drift': has_drift,
                'severity': self._categorize_drift(psi)
            }
        
        self.drift_results['feature_drift'] = drift_info
        
        # Summary
        drifted_features = [k for k, v in drift_info.items() if v['has_drift']]
        logger.info(f"Drift detected in {len(drifted_features)}/{len(common_cols)} features: {drifted_features}")
        
        return drift_info
    
    def _categorize_drift(self, psi: float) -> str:
        """Categorize drift severity based on PSI"""
        if psi < 0.1:
            return "none"
        elif psi < 0.2:
            return "moderate"
        else:
            return "significant"
    
    def detect_concept_drift(
        self,
        predictions_old: np.ndarray,
        predictions_new: np.ndarray,
        labels_new: np.ndarray,
        window_size: int = 100
    ) -> Dict:
        """
        Detect concept drift using prediction accuracy over time
        
        Args:
            predictions_old: Historical predictions
            predictions_new: Recent predictions
            labels_new: True labels for recent predictions
            window_size: Window size for rolling accuracy
        """
        
        # Calculate rolling accuracy
        accuracies = []
        
        for i in range(0, len(predictions_new), window_size):
            end_idx = min(i + window_size, len(predictions_new))
            acc = (predictions_new[i:end_idx] == labels_new[i:end_idx]).mean()
            accuracies.append(acc)
        
        # Compare with historical accuracy
        old_acc = (predictions_old == labels_new[:len(predictions_old)]).mean()
        new_acc = np.mean(accuracies[-3:])  # Recent accuracy
        
        accuracy_drop = old_acc - new_acc
        
        has_concept_drift = accuracy_drop > 0.05  # 5% drop threshold
        
        concept_drift_info = {
            'historical_accuracy': old_acc,
            'recent_accuracy': new_acc,
            'accuracy_drop': accuracy_drop,
            'has_concept_drift': has_concept_drift,
            'rolling_accuracies': accuracies
        }
        
        self.drift_results['concept_drift'] = concept_drift_info
        
        if has_concept_drift:
            logger.warning(f"Concept drift detected! Accuracy dropped from {old_acc:.4f} to {new_acc:.4f}")
        
        return concept_drift_info


class ModelMonitor:
    """Continuous monitoring of model performance and data quality"""
    
    def __init__(self, model_name: str = "fraud_detection"):
        self.model_name = model_name
        self.metrics_history = []
        self.alerts = []
        self.drift_detector = DriftDetector()
        
    def log_predictions(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        features: pd.DataFrame,
        timestamp: Optional[datetime] = None
    ):
        """Log predictions for later analysis"""
        
        log_entry = {
            'timestamp': timestamp or datetime.now(),
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'feature_stats': features.describe().to_dict()
        }
        
        self.metrics_history.append(log_entry)
        
        logger.debug(f"Logged {len(predictions)} predictions")
    
    def evaluate_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict:
        """Comprehensive model evaluation"""
        
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'timestamp': datetime.now(),
            'accuracy': report['accuracy'],
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall'],
            'f1_macro': report['macro avg']['f1-score'],
            'precision_fraud': report.get('1', {}).get('precision', 0),
            'recall_fraud': report.get('1', {}).get('recall', 0),
            'f1_fraud': report.get('1', {}).get('f1-score', 0),
            'confusion_matrix': cm.tolist(),
            'total_samples': len(y_true),
            'fraud_count': int(y_true.sum())
        }
        
        if y_proba is not None:
            # Add AUC-ROC if probabilities available
            from sklearn.metrics import roc_auc_score
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
            except:
                pass
        
        self.metrics_history.append(metrics)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        logger.info(f"Model evaluation - Accuracy: {metrics['accuracy']:.4f}, "
                   f"F1 (Fraud): {metrics['f1_fraud']:.4f}")
        
        return metrics
    
    def _check_alerts(self, metrics: Dict):
        """Generate alerts for concerning metrics"""
        
        alert_conditions = [
            (metrics['accuracy'] < 0.9, "Low overall accuracy"),
            (metrics['recall_fraud'] < 0.8, "Low fraud recall - missing fraud cases"),
            (metrics['precision_fraud'] < 0.7, "Low fraud precision - many false positives"),
            (metrics['f1_fraud'] < 0.75, "Low F1 score for fraud class")
        ]
        
        for condition, message in alert_conditions:
            if condition:
                alert = {
                    'timestamp': datetime.now(),
                    'severity': 'high' if 'recall' in message.lower() else 'medium',
                    'message': message,
                    'metric_value': None
                }
                self.alerts.append(alert)
                logger.warning(f"ALERT [{alert['severity'].upper()}]: {message}")
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """Get summary of all logged metrics"""
        
        if not self.metrics_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.metrics_history)
        return df
    
    def save_report(self, filepath: str):
        """Save monitoring report to file"""
        
        report = {
            'model_name': self.model_name,
            'generated_at': datetime.now().isoformat(),
            'total_evaluations': len(self.metrics_history),
            'alerts': [
                {
                    'timestamp': str(a['timestamp']),
                    'severity': a['severity'],
                    'message': a['message']
                }
                for a in self.alerts
            ],
            'latest_metrics': self.metrics_history[-1] if self.metrics_history else None,
            'metrics_history': self.metrics_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Monitoring report saved to {filepath}")
    
    def generate_dashboard_data(self) -> Dict:
        """Generate data for monitoring dashboard"""
        
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame(self.metrics_history)
        
        dashboard = {
            'performance_trends': {
                'accuracy': df[['timestamp', 'accuracy']].to_dict('records'),
                'f1_fraud': df[['timestamp', 'f1_fraud']].to_dict('records'),
                'recall_fraud': df[['timestamp', 'recall_fraud']].to_dict('records')
            },
            'alerts_summary': {
                'total': len(self.alerts),
                'high_severity': sum(1 for a in self.alerts if a['severity'] == 'high'),
                'medium_severity': sum(1 for a in self.alerts if a['severity'] == 'medium')
            },
            'data_drift': self.drift_detector.drift_results if hasattr(self.drift_detector, 'drift_results') else {}
        }
        
        return dashboard


def create_monitor(model_name: str = "fraud_detection") -> ModelMonitor:
    """Create a model monitor instance"""
    return ModelMonitor(model_name=model_name)
