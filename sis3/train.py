# train.py
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Start MLflow experiment
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("iris-classifier")

with mlflow.start_run():
    # Load data
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Hyperparameters
    n_estimators = 100

    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Log to MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Register model to Model Registry
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, "iris-model")

    print("Logged and registered model to MLflow!")