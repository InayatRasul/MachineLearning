# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import mlflow.pyfunc


# Load model
model = mlflow.pyfunc.load_model("models:/iris-model/1")

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