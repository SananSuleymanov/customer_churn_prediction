# Churn Prediction API

A FastAPI-based machine learning service that predicts customer churn using a Logistic Regression model.

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
```

### 2. Activate Virtual Environment

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

## API Endpoints

### Health Check

**Endpoint:** `GET /health`

**Description:** Check if the API is running

**Example:**
```bash
curl http://127.0.0.1:8000/health
```

### Predict Churn

**Endpoint:** `POST /api/predict`

**Description:** Predict customer churn probability

**Request Body:**
| Field | Type | Description |
|-------|------|-------------|
| `tenure` | int | Number of months the customer has stayed |
| `MonthlyCharges` | float | Monthly charges amount |
| `TotalCharges` | float | Total charges amount |
| `Contract` | str | Contract type (e.g., "Month-to-month", "One year", "Two year") |
| `InternetService` | str | Internet service type (e.g., "Fiber optic", "DSL", "No") |
| `OnlineSecurity` | str | Online security status (e.g., "Yes", "No", "No internet service") |
| `TechSupport` | str | Tech support status (e.g., "Yes", "No", "No internet service") |
| `PaperlessBilling` | str | Paperless billing status (e.g., "Yes", "No") |

**Response:**
| Field | Type | Description |
|-------|------|-------------|
| `churn` | str | Predicted churn ("Yes" or "No") |
| `probability` | float | Probability of churn (0.0 to 1.0) |

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 75.35,
    "TotalCharges": 903.50,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "TechSupport": "No",
    "PaperlessBilling": "Yes"
  }'
```

**Example Response:**
```json
{
  "churn": "Yes",
  "probability": 0.64
}
```

## Model Training

The machine learning model was trained using Google Colab. The training notebook is available at `train.ipynb`.

## Project Structure

```
churn_prediction/
├── main.py                          # FastAPI application entry point
├── requirements.txt                 # Python dependencies
├── train.ipynb                      # Model training notebook (Google Colab)
└── src/
    ├── api/
    │   ├── churn.py                # Churn prediction endpoint
    │   └── health.py               # Health check endpoint
    ├── core/
    │   └── logging.py              # Logging configuration
    ├── models/
    │   └── churn_model_pipeline.joblib  # Trained model
    ├── schemas/
    │   └── customer.py             # Pydantic schemas
    └── services/
        └── predict.py              # Prediction service
```


