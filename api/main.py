"""
FastAPI Application for Fraud Detection Service

RESTful API with real-time predictions, batch processing,
and model explainability endpoints.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
from loguru import logger
import asyncio
from contextlib import asynccontextmanager
import uvicorn


# Models for request/response
class TransactionInput(BaseModel):
    """Single transaction input"""
    amount: float = Field(..., description="Transaction amount")
    user_id: str = Field(..., description="User identifier")
    merchant_id: str = Field(..., description="Merchant identifier")
    timestamp: str = Field(..., description="Transaction timestamp")
    
    # Additional features can be added dynamically
    class Config:
        extra = "allow"


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    transactions: List[Dict[str, Any]]
    include_explanation: bool = False


class PredictionResponse(BaseModel):
    """Prediction response"""
    transaction_id: Optional[str] = None
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    explanation: Optional[Dict] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    last_updated: datetime


# Global variables
model = None
feature_engineer = None
monitor = None
prediction_queue = asyncio.Queue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Loading fraud detection models...")
    await load_models()
    logger.info("Models loaded successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down fraud detection service")


app = FastAPI(
    title="Financial Fraud Detection API",
    description="Advanced AI/ML-based fraud detection with real-time predictions and explainability",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def load_models():
    """Load trained models and feature engineers"""
    global model, feature_engineer, monitor
    
    try:
        # Load main model
        model_path = Path("models/fraud_detection_model.pkl")
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("Main model loaded")
        else:
            logger.warning("No model found at models/fraud_detection_model.pkl")
        
        # Load feature engineer
        fe_path = Path("models/feature_engineer.pkl")
        if fe_path.exists():
            with open(fe_path, 'rb') as f:
                feature_engineer = pickle.load(f)
            logger.info("Feature engineer loaded")
        
        # Initialize monitor
        from src.monitoring.drift_detector import create_monitor
        monitor = create_monitor()
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


def preprocess_transaction(transaction: Dict) -> pd.DataFrame:
    """Preprocess single transaction for prediction"""
    df = pd.DataFrame([transaction])
    
    # Apply feature engineering if available
    if feature_engineer is not None:
        df = feature_engineer.transform(df)
    
    return df


def determine_risk_level(probability: float) -> str:
    """Determine risk level based on fraud probability"""
    if probability >= 0.8:
        return "CRITICAL"
    elif probability >= 0.6:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    elif probability >= 0.2:
        return "LOW"
    else:
        return "VERY_LOW"


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Financial Fraud Detection API",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.now()
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    health_status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_engineer_loaded": feature_engineer is not None,
        "monitor_initialized": monitor is not None,
        "timestamp": datetime.now()
    }
    
    return health_status


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionInput):
    """
    Predict whether a single transaction is fraudulent
    
    Returns:
        PredictionResponse with fraud prediction and risk assessment
    """
    
    try:
        # Preprocess
        df = preprocess_transaction(transaction.dict())
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[0, 1]
            prediction = (proba > 0.5)
        else:
            prediction = model.predict(df)[0]
            proba = prediction
        
        # Log to monitor
        if monitor is not None:
            monitor.log_predictions(
                predictions=np.array([int(prediction)]),
                probabilities=np.array([proba]),
                features=df
            )
        
        return PredictionResponse(
            transaction_id=transaction.user_id,
            is_fraud=bool(prediction),
            fraud_probability=float(proba),
            risk_level=determine_risk_level(proba)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction endpoint for multiple transactions
    
    Args:
        request: BatchPredictionRequest with list of transactions
        
    Returns:
        List of PredictionResponse objects
    """
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.transactions)
        
        # Apply feature engineering
        if feature_engineer is not None:
            df_featured = feature_engineer.transform(df)
        else:
            df_featured = df
        
        # Make predictions
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(df_featured)[:, 1]
            predictions = (probas > 0.5).astype(int)
        else:
            predictions = model.predict(df_featured)
            probas = predictions
        
        # Build responses
        responses = []
        for i in range(len(predictions)):
            response = PredictionResponse(
                transaction_id=request.transactions[i].get('user_id', f"tx_{i}"),
                is_fraud=bool(predictions[i]),
                fraud_probability=float(probas[i]),
                risk_level=determine_risk_level(probas[i])
            )
            
            # Add explanation if requested
            if request.include_explanation:
                # TODO: Implement SHAP/LIME explanations
                response.explanation = {
                    "status": "not_implemented",
                    "message": "Explanation generation coming soon"
                }
            
            responses.append(response)
        
        # Log batch predictions
        if monitor is not None:
            monitor.log_predictions(
                predictions=predictions,
                probabilities=probas,
                features=df_featured
            )
        
        return responses
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
async def explain_prediction(transaction: TransactionInput):
    """
    Generate SHAP/LIME explanations for a prediction
    
    Returns:
        Explanation with feature importance and contribution
    """
    
    try:
        # Preprocess
        df = preprocess_transaction(transaction.dict())
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[0, 1]
        else:
            proba = model.predict(df)[0]
        
        # Generate SHAP explanation
        shap_explanation = None
        try:
            import shap
            
            # Create explainer
            if hasattr(model, 'feature_importances_'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(df)
                
                # Get top features
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Class 1
                
                feature_names = df.columns.tolist()
                abs_shap = np.abs(shap_values[0])
                top_indices = np.argsort(abs_shap)[-5:][::-1]
                
                shap_explanation = {
                    "top_features": [
                        {
                            "feature": feature_names[i],
                            "importance": float(abs_shap[i]),
                            "contribution": float(shap_values[0, i])
                        }
                        for i in top_indices
                    ],
                    "base_value": float(explainer.expected_value[1] if isinstance(explainer.expected_value, list) 
                                       else explainer.expected_value)
                }
        except ImportError:
            logger.warning("SHAP not installed")
        
        return {
            "prediction": bool(proba > 0.5),
            "probability": float(proba),
            "shap_explanation": shap_explanation,
            "features": transaction.dict()
        }
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=ModelMetrics)
async def get_metrics():
    """Get current model performance metrics"""
    
    if monitor is None or not monitor.metrics_history:
        raise HTTPException(
            status_code=404,
            detail="No metrics available. Model hasn't been evaluated yet."
        )
    
    latest_metrics = monitor.metrics_history[-1]
    
    return ModelMetrics(
        accuracy=latest_metrics.get('accuracy', 0.0),
        precision=latest_metrics.get('precision_fraud', 0.0),
        recall=latest_metrics.get('recall_fraud', 0.0),
        f1_score=latest_metrics.get('f1_fraud', 0.0),
        auc_roc=latest_metrics.get('auc_roc'),
        last_updated=latest_metrics.get('timestamp', datetime.now())
    )


@app.get("/drift-report")
async def get_drift_report():
    """Get data drift analysis report"""
    
    if monitor is None:
        raise HTTPException(status_code=404, detail="Monitor not initialized")
    
    if not monitor.drift_detector.reference_data is not None:
        raise HTTPException(
            status_code=400,
            detail="Reference data not set for drift detection"
        )
    
    return {
        "drift_results": monitor.drift_detector.drift_results,
        "alerts": monitor.alerts,
        "generated_at": datetime.now()
    }


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time fraud alerts"""
    
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive transaction data
            data = await websocket.receive_text()
            transaction = json.loads(data)
            
            # Process transaction
            df = preprocess_transaction(transaction)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(df)[0, 1]
            else:
                proba = model.predict(df)[0]
            
            # Send prediction
            response = {
                "is_fraud": bool(proba > 0.5),
                "probability": float(proba),
                "risk_level": determine_risk_level(proba),
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket.send_json(response)
            
            # Send alert for high-risk transactions
            if proba > 0.7:
                alert = {
                    "type": "FRAUD_ALERT",
                    "severity": "HIGH" if proba > 0.9 else "MEDIUM",
                    "transaction": transaction,
                    "probability": float(proba)
                }
                await websocket.send_json(alert)
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


@app.post("/admin/set-reference-data")
async def set_reference_data(data: List[Dict[str, Any]]):
    """Set reference data for drift detection (admin endpoint)"""
    
    if monitor is None:
        raise HTTPException(status_code=404, detail="Monitor not initialized")
    
    df = pd.DataFrame(data)
    monitor.drift_detector.set_reference(df)
    
    return {
        "status": "success",
        "message": f"Reference data set with {len(df)} samples",
        "features": list(df.columns)
    }


@app.on_event("startup")
async def startup_event():
    """Additional startup tasks"""
    logger.info("Starting Fraud Detection API v2.0")
    
    # Create directories if they don't exist
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Saving monitoring data...")
    
    if monitor is not None:
        try:
            monitor.save_report("logs/monitoring_report.json")
        except Exception as e:
            logger.error(f"Error saving monitoring data: {e}")


def main():
    """Run the API server"""
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
