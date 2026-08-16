

# Path to dataset inside the GitHub repository
DATA_PATH = Path( "tourism_project/data/tourism.csv")

# Expected columns
EXPECTED_COLUMNS = [
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
    "MonthlyIncome",
]


def register_dataset():
    """Read, validate, and summarize the tourism dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Check expected columns
    missing_columns = [
        column for column in EXPECTED_COLUMNS if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {', '.join(missing_columns)}"
        )

    print("Dataset registered successfully!")
    print(f"File: {DATA_PATH}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print("\nColumns:")
    print(", ".join(df.columns))


if __name__ == "__main__":
    register_dataset()
