from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load("churn_model.pkl")

app = FastAPI(
    title="Telco Customer Churn Prediction",
    description="API for predicting customer churn",
    version="1.0",
)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
async def predict_churn(data: CustomerData):
    data_dict = data.dict()
    df = pd.DataFrame([data_dict])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {"prediction": int(prediction), "churn_probability": float(probability)}


@app.get("/")
async def root():
    return {"message": "Prediction API is running!"}
