import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Load trained model and preprocessor
model = joblib.load(MODELS_DIR / "random_forest_log.pkl")
preprocessor = joblib.load(MODELS_DIR / "preprocessor_log.pkl")


def predict_house_price(
    area_sqft,
    bedrooms,
    bathrooms,
    balcony,
    parking_count,
    floor_number,
    total_floors,
    location,
    furnishing,
    transaction,
    facing,
    ownership,
    overlooking
):

    # Prepare input data
    input_data = pd.DataFrame([{
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "balcony": balcony,
        "parking_count": parking_count,
        "floor_number": floor_number,
        "total_floors": total_floors,
        "location": location,
        "Furnishing": furnishing,
        "Transaction": transaction,
        "facing": facing,
        "Ownership": ownership,
        "overlooking": overlooking
    }])

    # Apply preprocessing
    processed_data = preprocessor.transform(input_data)

    # Predict log price
    prediction_log = model.predict(processed_data)

    # Convert to original price
    predicted_price = np.expm1(prediction_log[0])

    return predicted_price


if __name__ == "__main__":

    price = predict_house_price(
        area_sqft=1200,
        bedrooms=3,
        bathrooms=2,
        balcony=1,
        parking_count=1,
        floor_number=3,
        total_floors=10,
        location="bangalore",
        furnishing="Semi-Furnished",
        transaction="New Property",
        facing="East",
        ownership="Freehold",
        overlooking="Garden/Park"
    )

    print("\n========== HOUSE PRICE PREDICTION ==========")

    print("Predicted House Price: ₹", round(price, 2))