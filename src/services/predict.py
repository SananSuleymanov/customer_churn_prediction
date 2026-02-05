from joblib import load
from pathlib import Path
import pandas as pd
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "churn_model_pipeline.joblib"

@lru_cache(maxsize=1)
def get_model():
    """Load and cache the model"""
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        model = load(MODEL_PATH)
        logger.info("Model loaded successfully")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def predict_churn(payload: dict):
    df = pd.DataFrame([payload])
    model = get_model()
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    
    churn_label = "Yes" if pred == 1 else "No"
    return {"churn": churn_label, "probability": round(float(prob), 2)}
