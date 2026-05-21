"""
03_eda_analysis.py

Purpose:
    Perform exploratory data analysis for DJIA financial news data.
    This script visualizes label distribution, text length,
    repeated-word ratio, word frequency, and VADER sentiment scores.

Input:
    outputs/tables/processed_news.csv

Output:
    outputs/figures/*.png
    outputs/tables/vader_sentiment_scores.csv
"""

from pathlib import Path
from collections import Counter
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


INPUT_PATH = Path("outputs/tables/processed_news.csv")
FIGURE_DIR = Path("outputs/figures")
TABLE_DIR = Path("outputs/tables")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
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


def simple_tokenize(text: str) -> list[str]:
    """Simple English tokenizer."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def plot_label_distribution(df: pd.DataFrame) -> None:
    """Plot label distribution."""
    plt.figure()
    df["Label"].value_counts().sort_index().plot(kind="bar")
    plt.title("Label Distribution (0=Down, 1=Up)")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "label_distribution.png", dpi=160)
    plt.close()


def plot_monthly_up_ratio(df: pd.DataFrame) -> None:
    """Plot monthly proportion of up days."""
    monthly_up_ratio = df.set_index("Date")["Label"].resample("ME").mean()

    plt.figure()
    monthly_up_ratio.plot()
    plt.title("Monthly Proportion of Up Days (DJIA)")
    plt.ylabel("Proportion of Label == 1")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "monthly_up_ratio.png", dpi=160)
    plt.close()


def plot_text_length_distribution(df: pd.DataFrame) -> None:
    """Plot distribution of daily news text length."""
    df["word_count"] = df["text"].str.split().str.len()

    plt.figure()
    df["word_count"].hist(bins=30)
    plt.title("Distribution of Daily News Length")
    plt.xlabel("Word Count per Day")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "text_length_distribution.png", dpi=160)
    plt.close()


def compute_repeated_word_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Compute repeated-word ratio per day."""
    records = []

    for _, row in df.iterrows():
        tokens = simple_tokenize(str(row["text"]))
        total = len(tokens)
        unique = len(set(tokens))
        repeated = total - unique
        ratio = repeated / total if total > 0 else np.nan

        records.append({
            "Date": row["Date"],
            "Label": row["Label"],
            "total_words": total,
            "unique_words": unique,
            "repeated_words": repeated,
            "repeated_word_ratio": ratio
        })

    return pd.DataFrame(records)


def plot_repeated_word_ratio(rep_df: pd.DataFrame) -> None:
    """Plot repeated-word ratio by label."""
    plt.figure()

    bins = np.linspace(0, np.nanmax(rep_df["repeated_word_ratio"]), 30)

    for label in [0, 1]:
        subset = rep_df.loc[rep_df["Label"] == label, "repeated_word_ratio"].dropna()
        plt.hist(subset, bins=bins, alpha=0.5, label=f"Label {label}")

    plt.title("Distribution of Repeated-Word Ratio by Label")
    plt.xlabel("Repeated-Word Ratio")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "repeated_word_ratio_by_label.png", dpi=160)
    plt.close()


def plot_top_word_frequency(df: pd.DataFrame, top_n: int = 20) -> None:
    """Plot top-N word frequency."""
    all_tokens = []

    for text in df["text"]:
        all_tokens.extend(simple_tokenize(str(text)))

    freq = Counter(all_tokens)
    top_words = freq.most_common(top_n)

    words = [word for word, _ in top_words]
    counts = [count for _, count in top_words]

    plt.figure(figsize=(10, 5))
    plt.bar(words, counts)
    plt.title(f"Top-{top_n} Word Frequency")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "top_word_frequency.png", dpi=160)
    plt.close()


def compute_vader_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Compute VADER sentiment scores."""
    nltk.download("vader_lexicon", quiet=True)
    analyzer = SentimentIntensityAnalyzer()

    sentiment_df = df[["Date", "Label", "text"]].copy()
    scores = sentiment_df["text"].apply(
        lambda text: pd.Series(analyzer.polarity_scores(str(text)))
    )

    sentiment_df = pd.concat([sentiment_df, scores], axis=1)
    sentiment_df.to_csv(TABLE_DIR / "vader_sentiment_scores.csv", index=False, encoding="utf-8")

    return sentiment_df


def plot_vader_sentiment(sentiment_df: pd.DataFrame) -> None:
    """Plot VADER sentiment trends and label-level averages."""
    monthly_sentiment = sentiment_df.set_index("Date")[["pos", "neg"]].resample("ME").mean()

    plt.figure()
    monthly_sentiment["pos"].plot(label="Positive")
    monthly_sentiment["neg"].plot(label="Negative")
    plt.title("Monthly Average Sentiment (VADER)")
    plt.ylabel("Score")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "monthly_vader_sentiment.png", dpi=160)
    plt.close()

    grouped = sentiment_df.groupby("Label")[["pos", "neg"]].mean()

    plt.figure()
    x = np.arange(len(grouped.index))
    width = 0.35

    plt.bar(x - width / 2, grouped["pos"].values, width=width, label="Positive")
    plt.bar(x + width / 2, grouped["neg"].values, width=width, label="Negative")

    plt.xticks(x, [f"Label {int(label)}" for label in grouped.index])
    plt.ylabel("Mean VADER Score")
    plt.title("Average VADER Sentiment by Label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "vader_sentiment_by_label.png", dpi=160)
    plt.close()


def main():
    df = load_data(INPUT_PATH)

    print("[INFO] Running EDA...")
    plot_label_distribution(df)
    plot_monthly_up_ratio(df)
    plot_text_length_distribution(df)

    repeated_df = compute_repeated_word_ratio(df)
    repeated_df.to_csv(TABLE_DIR / "repeated_word_ratio.csv", index=False, encoding="utf-8")
    plot_repeated_word_ratio(repeated_df)

    plot_top_word_frequency(df)

    sentiment_df = compute_vader_sentiment(df)
    plot_vader_sentiment(sentiment_df)

    print("[DONE] EDA figures and tables saved.")


if __name__ == "__main__":
    main()
