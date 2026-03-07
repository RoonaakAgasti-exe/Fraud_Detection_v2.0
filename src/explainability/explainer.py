import shap
import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Any, List, Optional

class FraudExplainer:
    """Provides SHAP-based explanations for fraud predictions"""
    
    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
    def _initialize_explainer(self, background_data: Optional[pd.DataFrame] = None):
        """Initialize the SHAP explainer"""
        if self.explainer is not None:
            return
            
        logger.info("Initializing SHAP explainer...")
        
        # Check model type to select appropriate explainer
        model_type = type(self.model).__name__.lower()
        
        if 'xgboost' in model_type or 'lgbm' in model_type or 'catboost' in model_type or 'randomforest' in model_type:
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # Fallback for generic models
            if background_data is not None:
                self.explainer = shap.KernelExplainer(self.model.predict_proba, background_data)
            else:
                logger.warning("No background data provided for KernelExplainer. Explanations might be slow or inaccurate.")
                
    def explain(self, df: pd.DataFrame, top_n: int = 5) -> Dict[str, Any]:
        """Generate explanations for a single or multiple samples"""
        self._initialize_explainer()
        
        if self.explainer is None:
            return {"error": "Explainer not initialized"}
            
        try:
            shap_values = self.explainer.shap_values(df)
            
            # Handle multi-class/binary classification outputs
            if isinstance(shap_values, list):
                # Usually [class 0, class 1] for binary
                val = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                val = shap_values
                
            results = []
            for i in range(len(df)):
                row_val = val[i]
                abs_val = np.abs(row_val)
                top_indices = np.argsort(abs_val)[-top_n:][::-1]
                
                explanation = {
                    "top_features": [
                        {
                            "feature": self.feature_names[j],
                            "importance": float(abs_val[j]),
                            "contribution": float(row_val[j])
                        }
                        for j in top_indices
                    ],
                    "base_value": float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, (list, np.ndarray)) 
                                       else self.explainer.expected_value)
                }
                results.append(explanation)
                
            return results[0] if len(results) == 1 else results
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            return {"error": str(e)}

def create_explainer(model: Any, feature_names: List[str]) -> FraudExplainer:
    return FraudExplainer(model, feature_names)
