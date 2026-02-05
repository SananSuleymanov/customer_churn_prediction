from pydantic import BaseModel


#Input parameters which should be same as used in training
class CustomerInput(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    OnlineSecurity: str
    TechSupport: str
    PaperlessBilling: str

class PredictionOutput(BaseModel):
    churn: str
    probability: float
