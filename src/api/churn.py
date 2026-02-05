from fastapi import APIRouter
from src.schemas.customer import CustomerInput, PredictionOutput
from src.services.predict import predict_churn

router = APIRouter()

@router.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    """
    Inference for predicting churn for signle customer
    """
    return predict_churn(customer.dict())
