from fastapi import FastAPI
from src.api import churn, health
from src.core.logging import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Churn prediction API",
    description="Prediction of cstomer churn using Logistic regression",
    version="1.0"
)

@app.on_event("startup")
def startup_event():
    try:
        logger.info("Application startup completed")
    except Exception as e:
        logger.critical(f"Startup failed: {e}")
        raise

app.include_router(health.router)
app.include_router(churn.router, prefix="/api", tags=["Churn"])