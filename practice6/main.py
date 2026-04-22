# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("model.joblib")

app = FastAPI()

# Input schema
class InputData(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "ML API is running"}

@app.post("/predict")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    
    return {"prediction": int(prediction[0])}