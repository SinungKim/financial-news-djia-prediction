"""
01_data_preprocessing.py

Purpose:
    Load the DJIA financial news dataset, validate schema,
    combine daily Top1~Top25 headlines into a single text field,
    clean text, and save processed data.

Input:
    data/Combined_News_DJIA.csv

Output:
    outputs/tables/processed_news.csv
"""

from pathlib import Path
import re
import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMBINED_PATH = DATA_DIR / "Combined_News_DJIA.csv"
OUTPUT_PATH = OUTPUT_DIR / "processed_news.csv"


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the raw DJIA news dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Please place Combined_News_DJIA.csv under the data/ directory."
        )

    df = pd.read_csv(path, encoding="utf-8", engine="python")
    df.columns = [col.strip() for col in df.columns]
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate required columns."""
    required_cols = {"Date", "Label"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def get_news_columns(df: pd.DataFrame) -> list[str]:
    """Detect Top1~Top25 headline columns."""
    candidates = []

    for col in df.columns:
        col_lower = col.strip().lower()
        match = re.match(r"top(\d+)$", col_lower)
        if match:
            candidates.append((int(match.group(1)), col))

    if not candidates:
        raise ValueError("No Top1~Top25 headline columns found.")

    return [col for _, col in sorted(candidates)]


def combine_headlines(row: pd.Series, news_cols: list[str]) -> str:
    """Combine multiple headline columns into one daily text string."""
    texts = []

    for col in news_cols:
        value = row[col]
        if pd.notna(value):
            text = str(value).strip()
            if text:
                texts.append(text)

    return " ".join(texts)


def clean_text(text: str) -> str:
    """Basic text cleaning for financial news headlines."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s\.\,\!\?\-\'\"]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess() -> pd.DataFrame:
    """Run full preprocessing pipeline."""
    df = load_dataset(COMBINED_PATH)
    validate_dataset(df)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    news_cols = get_news_columns(df)

    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Detected headline columns: {len(news_cols)}")
    print("[INFO] Label distribution:")
    print(df["Label"].value_counts(dropna=False))

    df["raw_text"] = df.apply(lambda row: combine_headlines(row, news_cols), axis=1)
    df["text"] = df["raw_text"].apply(clean_text)
    df["word_count"] = df["text"].str.split().str.len()

    processed = df[["Date", "Label", "text", "word_count"]].copy()

    empty_mask = processed["text"].isna() | (processed["text"].str.len() == 0)
    if empty_mask.any():
        print(f"[WARNING] Empty text rows removed: {empty_mask.sum()}")
        processed = processed.loc[~empty_mask].copy()

    processed.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"[SAVE] Processed dataset saved to: {OUTPUT_PATH}")

    return processed


if __name__ == "__main__":
    preprocess()
