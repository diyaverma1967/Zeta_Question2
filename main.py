from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, os
import pandas as pd
import numpy as np
import tensorflow as tf

PREPROCESSOR_PATH = "output/preprocessor.joblib"
MODEL_PATH        = "output/final_model.h5"

if not os.path.isfile(PREPROCESSOR_PATH):
    raise RuntimeError(f"Missing preprocessor: {PREPROCESSOR_PATH}")
preprocessor = joblib.load(PREPROCESSOR_PATH)

if not os.path.isfile(MODEL_PATH):
    raise RuntimeError(f"Missing model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)


class LoanRequest(BaseModel):
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Gender: str
    Married: str
    Dependents: float
    Education: str
    Self_Employed: str
    Credit_History: float
    Property_Area: str

def generate_recommendation(score: float) -> str:
    if score >= 75:
        return "Excellent credit likelihood. You should qualify for premium loans at great rates."
    elif score >= 50:
        return "Good credit likelihood. Standard loan offers are likely available to you."
    elif score >= 25:
        return "Fair credit likelihood. Consider improving your on‐time payments and reducing debts."
    else:
        return "Low credit likelihood. Focus on reducing missed payments and boosting stable income."


app = FastAPI(
    title="Loan Eligibility with AI",
    description="Returns a 0–100 credit‐likelihood score and a recommendation.",
    version="1.0"
)


@app.post("/loan/eligibility")
def check_eligibility(request: LoanRequest):
    data = request.dict()
    data["TotalIncome"] = data["ApplicantIncome"] + data["CoapplicantIncome"]

    df = pd.DataFrame([data])

    try:
        X_pp = preprocessor.transform(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {e}")

    prob = model.predict(X_pp).reshape(-1)[0]
    score = round(float(prob) * 100, 2)

    recommendation = generate_recommendation(score)

    return {
        "score": score,
        "recommendation": recommendation
    }