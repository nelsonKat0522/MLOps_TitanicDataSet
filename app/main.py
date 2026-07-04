from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os

app = FastAPI(title="Titanic Readmission Risk API")

MODEL_PATH = "models/model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Please run src/train.py first.")

model = joblib.load(MODEL_PATH)


class PatientInput(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Age: float = Field(..., ge=0, le=120)
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)
    Fare: float = Field(..., ge=0)
    Sex_male: int = Field(..., ge=0, le=1)
    Embarked_Q: int = Field(..., ge=0, le=1)
    Embarked_S: int = Field(..., ge=0, le=1)


@app.get("/")
def home():
    return {"message": "Readmission Risk Prediction API is running"}


@app.post("/predict")
def predict(data: PatientInput, x_api_key: str = Header(default=None)):
    # Simple API key security
    if x_api_key != "my-secret-key":
        raise HTTPException(status_code=401, detail="Invalid API key")

    input_df = pd.DataFrame([data.dict()])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "risk_probability": float(probability)
    }
