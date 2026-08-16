
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# Project paths
PROJECT_ROOT = Path("tourism_project")

DATA_PATH = PROJECT_ROOT / "data" / "tourism.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"


# Load dataset
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# Remove unnecessary column
if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])
    print("Removed CustomerID")


# Separate features and target
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Create train and test dataframes
train_df = X_train.copy()
train_df["ProdTaken"] = y_train

test_df = X_test.copy()
test_df["ProdTaken"] = y_test


# Create output directory
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Save CSV files
train_path = OUTPUT_DIR / "train.csv"
test_path = OUTPUT_DIR / "test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)


print("\nFiles created successfully:")
print("Train:", train_path)
print("Test :", test_path)

print("\nTrain shape:", train_df.shape)
print("Test shape :", test_df.shape)
