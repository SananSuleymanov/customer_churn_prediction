import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from joblib import dump


DATA_PATH = "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "src/models/churn_model_pipeline.joblib"
FIGURES_PATH = "reports/figures"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def preprocess_data(df: pd.DataFrame):
    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    # Drop ID column
    df = df.drop(columns=["customerID"])

    # Target encoding
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])

    # Feature types
    categorical_cols = X.select_dtypes(include="object").columns
    numerical_cols = X.select_dtypes(exclude="object").columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    return X, y, preprocessor


def plot_churn_distribution(df: pd.DataFrame):
    os.makedirs(FIGURES_PATH, exist_ok=True)

    ax = df["Churn"].value_counts().plot(
        kind="bar",
        color=["#B0B0B0", "#404040"]
    )
    ax.set_title("Churn Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "churn_distribution.png"))
    plt.close()


def train_model(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\nModel Evaluation")
    print("-" * 30)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(model, path)


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Plotting churn distribution...")
    plot_churn_distribution(df)

    print("Preprocessing data...")
    X, y, preprocessor = preprocess_data(df)

    print("Training model...")
    model, X_test, y_test = train_model(X, y, preprocessor)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("Saving model...")
    save_model(model, MODEL_PATH)

    print("Training completed successfully.")


if __name__ == "__main__":
    main()
