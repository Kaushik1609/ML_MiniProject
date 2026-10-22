"""
FraudGuard AI - FastAPI Backend Server
========================================
Serves the Machine Learning models, live prediction inference, sample transactions,
and static frontend assets for the Credit Card Fraud Detection web application.
"""

import os
import sys
import joblib
import numpy as np
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Ensure src can import local modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.logistic_regression_scratch import LogisticRegressionScratch

app = FastAPI(
    title="FraudGuard AI - Credit Card Fraud Detection API",
    description="Live scoring API comparing custom NumPy Logistic Regression, Scikit-Learn LR, and Random Forest.",
    version="1.0.0",
)

# Load serialized artifacts
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fraud_models.joblib")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model artifacts not found at {MODEL_PATH}. Run training first.")

artifacts = joblib.load(MODEL_PATH)
scaler = artifacts["scaler"]
feature_names = artifacts["feature_names"]
scratch_weights = artifacts["scratch_lr"]["weights"]
scratch_bias = artifacts["scratch_lr"]["bias"]
sk_lr = artifacts["sk_lr"]
rf = artifacts["rf"]
fraud_samples = artifacts["fraud_samples"]
legit_samples = artifacts["legit_samples"]


class PredictionRequest(BaseModel):
    model_type: str = Field(default="scratch_lr", description="scratch_lr | sk_lr | rf")
    threshold: float = Field(default=0.95, ge=0.01, le=0.99)
    features: Dict[str, float]


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "models_available": ["scratch_lr", "sk_lr", "rf"],
        "num_features": len(feature_names),
        "default_threshold": 0.95,
    }


@app.get("/api/samples")
def get_sample_transactions():
    """Return authentic test-set samples for quick 1-click UI testing."""
    return {
        "fraud_samples": fraud_samples,
        "legit_samples": legit_samples,
    }


@app.get("/api/metrics")
def get_benchmark_metrics():
    """Return precomputed test set benchmark comparison metrics."""
    return {
        "benchmarks": [
            {
                "model": "Majority Baseline",
                "accuracy": 0.9983,
                "precision": 0.0000,
                "recall": 0.0000,
                "f1_score": 0.0000,
                "roc_auc": 0.5000,
                "pr_auc": 0.0017,
                "tp": 0,
                "fp": 0,
                "fn": 98,
                "type": "baseline",
            },
            {
                "model": "Logistic Regression (Scratch, Default)",
                "threshold": 0.50,
                "accuracy": 0.9781,
                "precision": 0.0676,
                "recall": 0.9184,
                "f1_score": 0.1259,
                "roc_auc": 0.9752,
                "pr_auc": 0.7138,
                "tp": 90,
                "fp": 1242,
                "fn": 8,
                "type": "scratch",
            },
            {
                "model": "Scikit-Learn Logistic Regression",
                "threshold": 0.50,
                "accuracy": 0.9755,
                "precision": 0.0610,
                "recall": 0.9184,
                "f1_score": 0.1144,
                "roc_auc": 0.9721,
                "pr_auc": 0.7190,
                "tp": 90,
                "fp": 1386,
                "fn": 8,
                "type": "sklearn",
            },
            {
                "model": "Logistic Regression (Scratch, Tuned)",
                "threshold": 0.95,
                "accuracy": 0.9981,
                "precision": 0.4775,
                "recall": 0.8673,
                "f1_score": 0.6159,
                "roc_auc": 0.9752,
                "pr_auc": 0.7138,
                "tp": 85,
                "fp": 93,
                "fn": 13,
                "type": "scratch_tuned",
            },
            {
                "model": "Random Forest Classifier",
                "threshold": 0.50,
                "accuracy": 0.9994,
                "precision": 0.8404,
                "recall": 0.8061,
                "f1_score": 0.8229,
                "roc_auc": 0.9732,
                "pr_auc": 0.8313,
                "tp": 79,
                "fp": 15,
                "fn": 19,
                "type": "ensemble",
            },
        ],
        "test_size": 56962,
        "total_frauds": 98,
        "total_legit": 56864,
    }


@app.post("/api/predict")
def predict_fraud(payload: PredictionRequest):
    """
    Score a single transaction with the requested model and threshold.
    """
    # Build ordered vector
    vector = []
    for feat in feature_names:
        vector.append(float(payload.features.get(feat, 0.0)))

    x_raw = np.array(vector, dtype=np.float64).reshape(1, -1)
    # Apply fitted standard scaler
    x_scaled = scaler.transform(x_raw)

    model_type = payload.model_type
    th = payload.threshold

    if model_type == "scratch_lr":
        # Pure NumPy dot product + Sigmoid
        z = np.dot(x_scaled, scratch_weights) + scratch_bias
        z_clipped = np.clip(z, -250.0, 250.0)
        p1 = float(1.0 / (1.0 + np.exp(-z_clipped))[0])
        model_name = "Custom Logistic Regression (Scratch NumPy)"
    elif model_type == "sk_lr":
        p1 = float(sk_lr.predict_proba(x_scaled)[0, 1])
        model_name = "Scikit-Learn Logistic Regression (Balanced)"
    elif model_type == "rf":
        p1 = float(rf.predict_proba(x_scaled)[0, 1])
        model_name = "Random Forest Classifier (Balanced)"
    else:
        raise HTTPException(status_code=400, detail=f"Invalid model_type: {model_type}")

    is_fraud = bool(p1 >= th)

    # Determine risk category
    if p1 >= 0.85:
        risk_level = "CRITICAL"
        risk_color = "#ef4444"
    elif p1 >= 0.50:
        risk_level = "HIGH"
        risk_color = "#f97316"
    elif p1 >= 0.20:
        risk_level = "ELEVATED"
        risk_color = "#eab308"
    else:
        risk_level = "LOW"
        risk_color = "#10b981"

    # Identify top feature contributions (using scratch weights * scaled features)
    feature_contributions = []
    prod = (x_scaled[0] * scratch_weights)
    sorted_indices = np.argsort(np.abs(prod))[::-1]
    for idx in sorted_indices[:6]:
        feat_name = feature_names[idx]
        val = float(payload.features.get(feat_name, 0.0))
        impact_val = float(prod[idx])
        direction = "Increases Fraud Risk" if impact_val > 0 else "Supports Legitimacy"
        feature_contributions.append({
            "feature": feat_name,
            "raw_value": val,
            "impact_score": round(impact_val, 3),
            "direction": direction,
        })

    return {
        "model": model_name,
        "fraud_probability": round(p1, 5),
        "fraud_probability_percent": round(p1 * 100, 2),
        "threshold": th,
        "is_fraud": is_fraud,
        "status": "FRAUD SUSPECTED" if is_fraud else "TRANSACTION APPROVED",
        "risk_level": risk_level,
        "risk_color": risk_color,
        "decision_rule": f"P(Fraud) = {p1*100:.2f}% {'≥' if is_fraud else '<'} Threshold {th*100:.1f}%",
        "top_feature_contributions": feature_contributions,
    }


# Mount outputs directory for visual charts
outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
if os.path.exists(outputs_dir):
    app.mount("/outputs", StaticFiles(directory=outputs_dir), name="outputs")

# Mount frontend directory for HTML, CSS, JS
frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
def serve_index():
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"message": "Frontend index.html is being prepared."})
