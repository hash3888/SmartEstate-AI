import os
import joblib
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from preprocessing import preprocess_data


# Load processed data
print("\nLoading dataset...")

X_train, X_test, y_train, y_test, preprocessor = preprocess_data()

# Create models directory
os.makedirs("models", exist_ok=True)


# Define models
models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=2
    ),

    "XGBoost": XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        n_jobs=2,
        objective="reg:squarederror"
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42,
        n_jobs=2,
        verbosity=-1
    )
}


# Model file paths
model_paths = {
    "Linear Regression": "models/linear_regression.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl",
    "LightGBM": "models/lightgbm.pkl"
}


# Store evaluation results
results = []


# Train and evaluate models
for name, model in models.items():

    print(f"\nTraining {name}...")

    start_time = time.time()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(y_test, predictions)

    elapsed_time = time.time() - start_time

    print(f"\n--- {name} RESULTS ---")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))
    print("Training Time:", round(elapsed_time, 2), "seconds")

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2,
        "Training Time": elapsed_time
    })

    # Save trained model
    joblib.dump(model, model_paths[name])

    print(name, "saved successfully!")


# Save preprocessor
joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)


# Save model comparison
results_df = __import__("pandas").DataFrame(results)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)


print("\n========== FINAL MODEL COMPARISON ==========")

print(results_df.to_string(index=False))

print("\nBest Model:", results_df.iloc[0]["Model"])

print("\nAll models trained successfully!")
print("Models saved in the models directory.")