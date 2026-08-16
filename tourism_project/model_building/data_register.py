
from pathlib import Path
import pandas as pd


# Project root
PROJECT_ROOT = Path("tourism_project")

# Dataset path
DATA_PATH = PROJECT_ROOT / "data" / "tourism.csv"


# Check dataset exists
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )


# Load dataset
df = pd.read_csv(DATA_PATH)


# Expected columns
expected_columns = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]


# Check expected columns
missing_columns = [
    column for column in expected_columns
    if column not in df.columns
]


if missing_columns:
    raise ValueError(
        f"Missing expected columns: {missing_columns}"
    )


# Print summary
print("Dataset registered successfully.")
print(f"Dataset path: {DATA_PATH}")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nColumns:")
print(list(df.columns))

print("\nTarget distribution:")
print(df["ProdTaken"].value_counts())
