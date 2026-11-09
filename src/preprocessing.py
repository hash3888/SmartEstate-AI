import re
from sklearn.model_selection import GroupShuffleSplit
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# Convert Lac and Crore into rupees
def convert_price(value):

    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()

    match = re.search(
        r"([\d,]+(?:\.\d+)?)\s*(lac|lakh|cr|crore)",
        text
    )

    if not match:
        return np.nan

    number = float(match.group(1).replace(",", ""))
    unit = match.group(2)

    if unit in ["lac", "lakh"]:
        return number * 100000

    return number * 10000000


# Extract numerical values from area
def extract_number(value):

    if pd.isna(value):
        return np.nan

    match = re.search(
        r"[\d,]+(?:\.\d+)?",
        str(value)
    )

    if match:
        return float(match.group().replace(",", ""))

    return np.nan


# Extract floor information
def extract_floor(value):

    if pd.isna(value):
        return np.nan, np.nan

    text = str(value).lower().strip()

    total_match = re.search(r"out\s*of\s*(\d+)", text)

    total_floors = (
        float(total_match.group(1))
        if total_match else np.nan
    )

    if "ground" in text:
        floor_number = 0

    else:
        floor_match = re.search(r"^\s*(\d+)", text)

        floor_number = (
            float(floor_match.group(1))
            if floor_match else np.nan
        )

    return floor_number, total_floors


def preprocess_data():

    # Load dataset
    df = pd.read_csv("data/raw/house_prices.csv")

    print("Original dataset:", df.shape)

    # Convert target price
    df["price"] = df["Amount(in rupees)"].apply(convert_price)

    # Extract area
    carpet_area = df["Carpet Area"].apply(extract_number)
    super_area = df["Super Area"].apply(extract_number)

    df["area_sqft"] = carpet_area.fillna(super_area)

    # Extract bedrooms from title
    df["bedrooms"] = (
        df["Title"]
        .str.extract(r"(\d+)\s*BHK", flags=re.IGNORECASE)[0]
    )

    df["bedrooms"] = pd.to_numeric(
        df["bedrooms"], errors="coerce"
    )

    # Bathroom and balcony
    df["bathrooms"] = df["Bathroom"].apply(extract_number)
    df["balcony"] = df["Balcony"].apply(extract_number)

    # Parking
    df["parking_count"] = df["Car Parking"].apply(extract_number)

    # Floor
    floor_data = df["Floor"].apply(extract_floor)

    df["floor_number"] = floor_data.apply(lambda x: x[0])
    df["total_floors"] = floor_data.apply(lambda x: x[1])

    # Remove records without valid target price or area
    df = df[
        (df["price"].notna()) &
        (df["price"] > 0) &
        (df["price"] <= 200000000) &
        (df["area_sqft"].notna()) &
        (df["area_sqft"] > 0)
    ].copy()

    # Clean categorical columns
    categorical_features = [
        "location",
        "Furnishing",
        "Transaction",
        "facing",
        "Ownership",
        "overlooking"
    ]

    for col in categorical_features:

        df[col] = df[col].apply(
            lambda value: value.strip()
            if isinstance(value, str)
            else np.nan
        )

        df[col] = df[col].replace("", np.nan)

    # Numerical features
    numerical_features = [
        "area_sqft",
        "bedrooms",
        "bathrooms",
        "balcony",
        "parking_count",
        "floor_number",
        "total_floors"
    ]

    # Final input and target
    X = df[numerical_features + categorical_features]
    y = df["price"]

    # Save cleaned dataset
    os.makedirs("data/processed", exist_ok=True)

    df[X.columns.tolist() + ["price"]].to_csv(
        "data/processed/cleaned_house_prices.csv",
        index=False
    )

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing pipelines
    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    # Group-based Train-Test Split
    groups = pd.util.hash_pandas_object(X, index=False)

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

    # Fit only on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    # Results
    print("\nPREPROCESSING COMPLETED!")

    print("Cleaned dataset:", df.shape)
    print("Training data:", X_train_processed.shape)
    print("Testing data:", X_test_processed.shape)

    print("\nMissing target values:", y.isnull().sum())

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )


if __name__ == "__main__":
    preprocess_data()