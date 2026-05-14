# ML FastAPI + Docker + MLflow + Streamlit Project

mlflow.ui

uvicorn main:app --reload

streamlit run app.py
## 📌 Overview

This project demonstrates a complete **end-to-end Machine Learning pipeline**, starting from model training and ending with user interaction through a web interface.

The system includes:

* Machine Learning model training
* Experiment tracking using MLflow
* Model deployment via FastAPI
* User interface using Streamlit
* (Optional) Containerization using Docker

The goal is to show how a trained model can be **tracked, versioned, deployed, and used by end users**.

---

## ⚙️ System Architecture

The pipeline follows this structure:

Train Model → Log with MLflow → Register Model → Serve via API → Use via Frontend

So:

* If the model is trained → it is logged in MLflow
* If logged → it can be versioned and reused
* If versioned → API loads a stable model
* If API works → frontend can interact with it

---

## 📁 Project Structure

```
ml-fastapi-docker/
│
├── train.py           # Model training + MLflow logging + registration
├── main.py            # FastAPI backend (serves predictions)
├── app.py             # Streamlit frontend (user interface)
├── requirements.txt   # Project dependencies
├── Dockerfile         # Container setup
├── mlflow.db          # MLflow database (created after running)
└── README.md          # Project documentation
```

---

## 🧠 Step 1 — Model Training

The model is trained using the Iris dataset and a Random Forest classifier.

During training:

* Data is split into training and testing sets
* Model is trained
* Predictions are evaluated

### Metrics logged:

* Accuracy
* F1-score

---

## 📊 Step 2 — MLflow Integration

MLflow is used for experiment tracking and model lifecycle management.

### Experiment Tracking:

* Logs hyperparameters (e.g., number of trees)
* Logs evaluation metrics
* Logs trained model as artifact

### Model Registry:

* Model is registered under a name (e.g., `iris-model`)
* Each training run creates a new version

This allows:

* reproducibility
* version control
* comparison of experiments

---

## 🚀 Step 3 — FastAPI Backend

The FastAPI application provides two endpoints:

### 1. Root endpoint

```
GET /
```

Returns a message confirming that the API is running.

---

### 2. Prediction endpoint

```
POST /predict
```

Accepts input features:

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Returns prediction:

```json
{
  "prediction": 0,
  "class_name": "Setosa"
}
```

The API loads the model from MLflow Model Registry.

---

## 🌐 Step 4 — Streamlit Frontend

A simple user interface is built using Streamlit.

Features:

* Input fields for 4 numerical features
* Button to send request to API
* Displays predicted class

This allows non-technical users to interact with the model.

---

## 🐳 Step 5 — Docker (Optional)

Docker is used to containerize the application.

### Purpose:

* Ensure reproducibility
* Run project on any machine
* Package dependencies and runtime

---

## ▶️ How to Run the Project

### 1. Install dependencies

```
pip install -r requirements.txt
```

---

### 2. Train the model

```
python train.py
```

---

### 3. Start MLflow UI

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

```
http://127.0.0.1:5000
```

---

### 4. Run FastAPI server

```
uvicorn main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

### 5. Run Streamlit frontend

```
streamlit run app.py
```

---

## 🧪 Testing the API

Example test input:

```json
{
  "features": [6.0, 2.9, 4.5, 1.5]
}
```

Expected output:

```
Versicolor
```

---
