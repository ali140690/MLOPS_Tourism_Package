
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Paths
PROJECT_ROOT = Path("tourism_project")

TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "splits" / "test.csv"

DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
MODEL_PATH = DEPLOYMENT_DIR / "tourism_model.pkl"


# Check files
print("Training file:", TRAIN_PATH)
print("Testing file:", TEST_PATH)

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")

if not TEST_PATH.exists():
    raise FileNotFoundError(f"Testing file not found: {TEST_PATH}")


# Load data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


# Features and target
X_train = train_df.drop(columns=["ProdTaken"])
y_train = train_df["ProdTaken"]

X_test = test_df.drop(columns=["ProdTaken"])
y_test = test_df["ProdTaken"]


# Identify categorical and numerical columns
categorical_columns = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)


# Model pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# Hyperparameter grid
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}


# MLflow
mlflow.set_experiment("Tourism Package Prediction")

with mlflow.start_run():

    print("\nStarting hyperparameter tuning...")

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("\nBest parameters:")
    print(grid_search.best_params_)


    # Log parameters
    mlflow.log_params(grid_search.best_params_)

    # Log CV score
    mlflow.log_metric(
        "best_cv_f1",
        grid_search.best_score_
    )


    # Predictions
    y_pred = best_model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    y_probability = best_model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )


    # Log metrics
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1", f1)
    mlflow.log_metric("test_roc_auc", roc_auc)


    # Create deployment directory
    DEPLOYMENT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # Save model
    joblib.dump(
        best_model,
        MODEL_PATH
    )


    # Log model to MLflow
    mlflow.sklearn.log_model(
        best_model,
        "tourism_model"
    )


    # Results
    print("\n" + "=" * 50)
    print("MODEL TRAINING COMPLETE")
    print("=" * 50)

    print("\nAccuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nModel saved at:")
    print(MODEL_PATH)
