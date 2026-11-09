import joblib
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Load trained model and preprocessor
model = joblib.load(MODELS_DIR / "random_forest_log.pkl")
preprocessor = joblib.load(MODELS_DIR / "preprocessor_log.pkl")

# Get feature names
feature_names = preprocessor.get_feature_names_out()

# Get feature importance
importance = model.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

# Sort by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Display top 20 features
print("\n===== TOP 20 IMPORTANT FEATURES =====")
print(importance_df.head(20).to_string(index=False))

# Save results
output_path = MODELS_DIR / "feature_importance.csv"

importance_df.to_csv(output_path, index=False)

print("\nFeature importance saved successfully!")
print(output_path)
import matplotlib.pyplot as plt

# Top 15 features visualization
top_features = importance_df.head(15).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Top 15 House Price Prediction Features")

plt.tight_layout()

plt.savefig(
    MODELS_DIR / "feature_importance.png",
    dpi=300
)

plt.show()

print("Feature importance graph saved successfully!")