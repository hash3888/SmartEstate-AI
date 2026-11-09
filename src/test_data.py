import pandas as pd

# Load new dataset
df = pd.read_csv("data/raw/house_prices.csv")

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DATASET INFO ---")
df.info()
print("\n--- IMPORTANT COLUMN SAMPLES ---")

columns = [
    "Amount(in rupees)",
    "Price (in rupees)",
    "location",
    "Carpet Area",
    "Super Area",
    "Bathroom",
    "Floor",
    "Furnishing",
    "Car Parking"
]

for col in columns:
    print(f"\n{col}:")
    print(df[col].dropna().head(10).tolist())
    print("\n--- RANDOM PROPERTY PRICE SAMPLES ---")

print(
    df[["Amount(in rupees)", "Price (in rupees)", "Carpet Area", "Super Area"]]
    .sample(15, random_state=42)
    .to_string(index=False)
)

print("\n--- PRICE FORMAT COUNTS ---")

print(
    df["Amount(in rupees)"]
    .astype(str)
    .str.extract(r"(Lac|Cr)", expand=False)
    .value_counts(dropna=False)
)

print("\n--- UNIQUE FURNISHING VALUES ---")

print(df["Furnishing"].value_counts(dropna=False))

print("\n--- UNIQUE STATUS VALUES ---")

print(df["Status"].value_counts(dropna=False))
print("\n--- UNUSUAL PRICE FORMATS ---")

price_text = df["Amount(in rupees)"].astype(str).str.strip()

unusual_prices = df[
    ~price_text.str.contains(
        r"Lac|Cr",
        case=False,
        na=False
    )
]

print("Unusual price records:", len(unusual_prices))

print("\nExamples:")
print(
    unusual_prices["Amount(in rupees)"]
    .value_counts()
    .head(30)
)