import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from preprocessing import preprocess_data

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

print("Loading trained model...")

# Load trained model and preprocessor
model = joblib.load(MODELS_DIR / "random_forest_log.pkl")
preprocessor = joblib.load(MODELS_DIR / "preprocessor_log.pkl")

print("Preparing test data...")

# Get processed test data
_, X_test, _, _, _ = preprocess_data()

# Use only 100 rows to keep SHAP computation manageable
X_sample = X_test[:100].toarray()

# Get feature names
feature_names = preprocessor.get_feature_names_out()

print("Calculating SHAP values...")

# Create SHAP Tree Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

print("Generating SHAP summary plot...")

# Generate summary plot
shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=feature_names,
    show=False,
    max_display=15
)

plt.tight_layout()
plt.savefig(
    MODELS_DIR / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nSHAP ANALYSIS COMPLETED!")
print("Graph saved at:", MODELS_DIR / "shap_summary.png")