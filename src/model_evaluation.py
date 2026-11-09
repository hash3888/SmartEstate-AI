import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports" / "model_evaluation"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD SAVED MODEL
# ==========================================

print("\nLoading saved Random Forest model...")

model = joblib.load(
    MODELS_DIR / "random_forest_log.pkl"
)

preprocessor = joblib.load(
    MODELS_DIR / "preprocessor_log.pkl"
)

print("Model loaded successfully!")


# ==========================================
# LOAD CLEANED DATASET
# ==========================================

print("\nLoading cleaned dataset...")

df = pd.read_csv(
    DATA_DIR / "cleaned_house_prices.csv"
)

numerical_features = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "balcony",
    "parking_count",
    "floor_number",
    "total_floors"
]

categorical_features = [
    "location",
    "Furnishing",
    "Transaction",
    "facing",
    "Ownership",
    "overlooking"
]

X = df[numerical_features + categorical_features]
y = df["price"]

print("Dataset shape:", df.shape)


# ==========================================
# RECREATE ORIGINAL TRAIN-TEST SPLIT
# ==========================================

print("\nRecreating GroupShuffleSplit...")

groups = pd.util.hash_pandas_object(
    X,
    index=False
)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# ==========================================
# PREPROCESS DATA
# ==========================================

print("\nTransforming features...")

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

if X_train_processed.shape[1] != model.n_features_in_:
    raise ValueError("Model and preprocessor feature mismatch!")

print("Feature count:", X_train_processed.shape[1])


# ==========================================
# PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

train_predictions_log = model.predict(X_train_processed)
test_predictions_log = model.predict(X_test_processed)

# Convert log predictions to original rupees

train_predictions = np.maximum(
    np.expm1(train_predictions_log),
    0
)

test_predictions = np.maximum(
    np.expm1(test_predictions_log),
    0
)


# ==========================================
# EVALUATION FUNCTION
# ==========================================

def calculate_metrics(actual, predicted):

    return {
        "MAE": mean_absolute_error(actual, predicted),

        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),

        "R2 Score": r2_score(actual, predicted),

        "MAPE (%)": mean_absolute_percentage_error(
            actual,
            predicted
        ) * 100
    }


# ==========================================
# TRAINING AND TESTING METRICS
# ==========================================

train_metrics = calculate_metrics(
    y_train,
    train_predictions
)

test_metrics = calculate_metrics(
    y_test,
    test_predictions
)

results = pd.DataFrame(
    [train_metrics, test_metrics],
    index=["Training", "Testing"]
)

print("\n========== MODEL PERFORMANCE ==========")

print(results.round(4).to_string())


# ==========================================
# OVERFITTING ANALYSIS
# ==========================================

r2_gap = train_metrics["R2 Score"] - test_metrics["R2 Score"]

print("\n========== OVERFITTING ANALYSIS ==========")

print("Training R2:", round(train_metrics["R2 Score"], 4))
print("Testing R2:", round(test_metrics["R2 Score"], 4))
print("R2 Gap:", round(r2_gap, 4))

if r2_gap > 0.15:
    print("Warning: Possible overfitting detected.")

elif r2_gap > 0.08:
    print("Moderate train-test performance gap.")

else:
    print("No large R2 gap detected.")


# ==========================================
# SAVE METRICS REPORT
# ==========================================

results.to_csv(
    REPORT_DIR / "training_testing_metrics.csv"
)

print("\nMetrics CSV saved!")


# ==========================================
# SAVE PREDICTIONS
# ==========================================

predictions_df = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": test_predictions,
    "Absolute Error": np.abs(
        y_test.values - test_predictions
    )
})

predictions_df.to_csv(
    REPORT_DIR / "test_predictions.csv",
    index=False
)


# ==========================================
# ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    test_predictions,
    alpha=0.3
)

min_price = min(y_test.min(), test_predictions.min())
max_price = max(y_test.max(), test_predictions.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--"
)

plt.xlabel("Actual House Price (₹)")
plt.ylabel("Predicted House Price (₹)")

plt.title("Actual vs Predicted House Prices")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ==========================================
# RESIDUAL ERROR GRAPH
# ==========================================

residuals = y_test.values - test_predictions

plt.figure(figsize=(9, 6))

plt.scatter(
    test_predictions,
    residuals,
    alpha=0.3
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted House Price (₹)")
plt.ylabel("Residual Error (₹)")

plt.title("Residual Error Analysis")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "residual_analysis.png",
    dpi=300
)

plt.close()


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n======================================")
print("MODEL EVALUATION COMPLETED!")
print("======================================")

print("Reports saved at:")
print(REPORT_DIR)

print("\nGenerated files:")

for file in REPORT_DIR.iterdir():
    print("-", file.name)