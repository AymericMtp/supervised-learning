import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
)
from lightgbm import LGBMClassifier


# ================================
# 1. Load and prepare the dataset
# ================================

DATA_PATH = "income_boosted.csv"
TARGET_COL = "income"

def load_data(path=DATA_PATH):
    """
    Loads dataset and separates features/target.
    """
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def build_preprocessor(X):
    """
    Creates a ColumnTransformer that:
    - scales numeric features
    - one-hot encodes categorical features
    """
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    return preprocessor


def build_model():
    """
    Builds the LightGBM model with the hyperparameters that gave the best performance.
    """
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="binary",
    )
    return model


# ================================
# 2. Main execution (with MLflow)
# ================================

def main():

    # Create or load experiment
    mlflow.set_experiment("Income Prediction")

    with mlflow.start_run():

        # Load data
        X, y = load_data()

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Build preprocessing
        preprocessor = build_preprocessor(X)

        # Build LightGBM model
        model = build_model()

        # Create pipeline (preprocessing → model)
        clf = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        # Log model configuration to MLflow
        mlflow.log_param("model_type", "LightGBM")
        for key, value in model.get_params().items():
            mlflow.log_param(f"lgbm_{key}", value)

        # Train model
        clf.fit(X_train, y_train)

        # Predictions
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        prec_w = precision_score(y_test, y_pred, average="weighted")
        rec_w = recall_score(y_test, y_pred, average="weighted")
        f1_w = f1_score(y_test, y_pred, average="weighted")
        roc_auc = roc_auc_score(y_test, y_proba)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("balanced_accuracy", bal_acc)
        mlflow.log_metric("precision_weighted", prec_w)
        mlflow.log_metric("recall_weighted", rec_w)
        mlflow.log_metric("f1_weighted", f1_w)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log the entire sklearn pipeline (preprocessing + model)
        mlflow.sklearn.log_model(clf, "lightgbm_income_model")

        joblib.dump(clf, "final_lightgbm_pipeline.joblib")
        print("\nModel pipeline saved to final_lightgbm_pipeline.joblib")


        # Print results
        print("\n========== FINAL MODEL: LightGBM ==========")
        print(f"Accuracy             : {acc:.4f}")
        print(f"Balanced Accuracy    : {bal_acc:.4f}")
        print(f"Precision (weighted) : {prec_w:.4f}")
        print(f"Recall (weighted)    : {rec_w:.4f}")
        print(f"F1-score (weighted)  : {f1_w:.4f}")
        print(f"ROC AUC              : {roc_auc:.4f}")


if __name__ == "__main__":
    main()
