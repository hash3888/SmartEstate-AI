import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import preprocess_data


print("\nLoading dataset...")

X_train, X_test, y_train, y_test, preprocessor = preprocess_data()

# Log transform target
y_train_log = np.log1p(y_train)

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=2
)

print("\nTraining Random Forest with Log Transformation...")

model.fit(X_train, y_train_log)

# Predict in log scale
predictions_log = model.predict(X_test)

# Convert predictions back to rupees
predictions = np.expm1(predictions_log)

# Evaluation
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\n===== LOG TRANSFORMATION RESULTS =====")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))
import os
import joblib
from pathlib import Path

models_dir = Path(__file__).resolve().parent.parent / "models"
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / "random_forest_log.pkl"

joblib.dump(
    model,
    model_path
)

print(f"\nLog Random Forest model saved successfully: {model_path}")
import os
import joblib
from pathlib import Path

models_dir = Path(__file__).resolve().parent.parent / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# Save trained model
joblib.dump(
    model,
    models_dir / "random_forest_log.pkl"
)

# Save fitted preprocessor
joblib.dump(
    preprocessor,
    models_dir / "preprocessor_log.pkl"
)

print("\nModel and Preprocessor saved successfully!")