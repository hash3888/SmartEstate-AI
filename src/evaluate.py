import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocessing import preprocess_data


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


print("\nLoading test data...")

# Load preprocessed train-test split
X_train, X_test, y_train, y_test, preprocessor = preprocess_data()

# Load trained log Random Forest model
model = joblib.load(
    MODELS_DIR / "random_forest_log.pkl"
)

print("\nGenerating predictions...")

# Predict log prices
predictions_log = model.predict(X_test)

# Convert predictions back to rupees
predictions = np.expm1(predictions_log)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

# Calculate Mean Absolute Percentage Error
mape = np.mean(
    np.abs((y_test - predictions) / y_test)
) * 100


print("\n===================================")
print("       MODEL EVALUATION REPORT")
print("===================================")

print(f"MAE  : ₹{mae:,.2f}")
print(f"RMSE : ₹{rmse:,.2f}")
print(f"R2   : {r2:.4f}")
print(f"MAPE : {mape:.2f}%")

print("===================================")


# Save evaluation results
results = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2 Score", "MAPE"],
    "Value": [mae, rmse, r2, mape]
})

results.to_csv(
    MODELS_DIR / "evaluation_results.csv",
    index=False
)

print("\nEvaluation results saved successfully!")