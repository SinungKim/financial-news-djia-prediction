"""
02_baseline_model.py

Purpose:
    Train a baseline DJIA direction prediction model using
    TF-IDF features and Logistic Regression.

Input:
    outputs/tables/processed_news.csv

Output:
    outputs/tables/baseline_classification_report.csv
    outputs/tables/baseline_confusion_matrix.csv
"""

from pathlib import Path
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


INPUT_PATH = Path("outputs/tables/processed_news.csv")
OUTPUT_DIR = Path("outputs/tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "baseline_classification_report.csv"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "baseline_confusion_matrix.csv"


def load_processed_data(path: Path) -> pd.DataFrame:
    """Load processed news data."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}\n"
            "Run src/01_data_preprocessing.py first."
        )

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def time_based_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """Split dataset by chronological order to prevent data leakage."""
    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df


def train_baseline_model(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Train TF-IDF + Logistic Regression baseline model."""
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english"
    )

    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    y_train = train_df["Label"].astype(int)
    y_test = test_df["Label"].astype(int)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_test, y_pred


def save_evaluation(y_test, y_pred) -> None:
    """Save classification report and confusion matrix."""
    report_dict = classification_report(
        y_test,
        y_pred,
        digits=4,
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(REPORT_PATH, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"]
    )
    cm_df.to_csv(CONFUSION_MATRIX_PATH, encoding="utf-8")

    print("[SAVE] Classification report:", REPORT_PATH)
    print("[SAVE] Confusion matrix:", CONFUSION_MATRIX_PATH)


def main():
    df = load_processed_data(INPUT_PATH)
    train_df, test_df = time_based_split(df)

    print("=== Time-based Split ===")
    print(f"Train: {train_df['Date'].min().date()} ~ {train_df['Date'].max().date()}")
    print(f"Test : {test_df['Date'].min().date()} ~ {test_df['Date'].max().date()}")

    y_test, y_pred = train_baseline_model(train_df, test_df)
    save_evaluation(y_test, y_pred)


if __name__ == "__main__":
    main()
