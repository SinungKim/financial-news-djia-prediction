"""
04_finbert_sentiment_analysis.py

Purpose:
    Perform FinBERT-based sentiment analysis on financial news headlines.
    This script supports both:
        1. Raw DJIA news dataset with Top1~Top25 headline columns
        2. Processed dataset with a single text column

Input:
    data/Combined_News_DJIA.csv
    or
    outputs/tables/processed_news.csv

Output:
    outputs/tables/headline_finbert_scores.csv
    outputs/tables/daily_finbert_aggregates.csv
    outputs/figures/monthly_finbert_sentiment_mean.png
    outputs/figures/finbert_mean_by_label.png
    outputs/figures/finbert_pred_distribution_by_label.png
    outputs/figures/finbert_pos_vs_neg_scatter.png
"""

from pathlib import Path
import re
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


INPUT_PATH = Path("outputs/tables/processed_news.csv")
TABLE_DIR = Path("outputs/tables")
FIGURE_DIR = Path("outputs/figures")

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "yiyanghkust/finbert-tone"


def load_input(path: Path) -> pd.DataFrame:
    """Load input CSV and normalize Date column."""
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Run src/01_data_preprocessing.py first or place input CSV under the expected path."
        )

    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]

    if "Date" not in df.columns:
        date_candidates = [col for col in df.columns if "date" in col.lower()]
        if not date_candidates:
            raise ValueError("No Date column found.")
        df = df.rename(columns={date_candidates[0]: "Date"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    return df


def detect_schema(df: pd.DataFrame) -> Tuple[str, List[str]]:
    """
    Detect input schema.

    Returns:
        ("topk", [Top1, Top2, ...]) if Top columns exist
        ("text", ["text"]) if processed text column exists
    """
    top_cols = []

    for col in df.columns:
        col_lower = col.strip().lower()
        match = re.match(r"top(\d+)$", col_lower)
        if match:
            top_cols.append((int(match.group(1)), col))

    if top_cols:
        return "topk", [col for _, col in sorted(top_cols)]

    text_cols = [col for col in df.columns if col.lower() == "text"]
    if text_cols:
        return "text", [text_cols[0]]

    raise ValueError("Cannot detect schema. Need Top1~Top25 columns or a text column.")


def flatten_to_headlines(df: pd.DataFrame, mode: str, cols: List[str]) -> pd.DataFrame:
    """Convert input data into headline-level long format."""
    has_label = "Label" in df.columns
    records = []

    if mode == "topk":
        for _, row in df.iterrows():
            date_value = row["Date"]
            label_value = int(row["Label"]) if has_label and pd.notna(row["Label"]) else None

            for col in cols:
                value = row[col]
                if pd.notna(value):
                    headline = str(value).strip()
                    if headline:
                        records.append((date_value, label_value, headline))

    elif mode == "text":
        text_col = cols[0]

        for _, row in df.iterrows():
            value = row[text_col]
            if pd.notna(value):
                text = str(value).strip()
                if text:
                    label_value = int(row["Label"]) if has_label and pd.notna(row["Label"]) else None
                    records.append((row["Date"], label_value, text))

    headline_df = pd.DataFrame(records, columns=["Date", "Label", "headline"])
    return headline_df


def build_finbert_pipeline(model_name: str = MODEL_NAME):
    """Build FinBERT sentiment analysis pipeline."""
    device = 0 if torch.cuda.is_available() else -1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    sentiment_pipeline = pipeline(
        task="sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,
        truncation=True,
        max_length=512
    )

    return sentiment_pipeline


def run_finbert(sentiment_pipeline, texts: List[str], batch_size: int = 16) -> pd.DataFrame:
    """Run FinBERT inference and return probability scores."""
    results = []

    for start_idx in tqdm(range(0, len(texts), batch_size), desc="FinBERT inference"):
        batch = texts[start_idx:start_idx + batch_size]
        outputs = sentiment_pipeline(
            batch,
            truncation=True,
            max_length=512
        )

        for output in outputs:
            prob = {item["label"].lower(): float(item["score"]) for item in output}

            positive = prob.get("positive", np.nan)
            neutral = prob.get("neutral", np.nan)
            negative = prob.get("negative", np.nan)

            label_pred = max(prob, key=prob.get) if prob else None

            results.append({
                "positive": positive,
                "neutral": neutral,
                "negative": negative,
                "label_pred": label_pred
            })

    return pd.DataFrame(results)


def aggregate_daily(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate headline-level FinBERT scores into daily metrics."""
    grouped = scored_df.groupby("Date")

    daily = grouped.agg(
        pos_mean=("positive", "mean"),
        pos_median=("positive", "median"),
        neu_mean=("neutral", "mean"),
        neu_median=("neutral", "median"),
        neg_mean=("negative", "mean"),
        neg_median=("negative", "median"),
        headline_count=("headline", "count")
    ).reset_index()

    pred_dist = (
        scored_df
        .groupby(["Date", "label_pred"])
        .size()
        .groupby(level=0)
        .apply(lambda x: x / x.sum())
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["positive", "neutral", "negative"]:
        if col not in pred_dist.columns:
            pred_dist[col] = 0.0

    pred_dist = pred_dist.rename(columns={
        "positive": "pred_pos_ratio",
        "neutral": "pred_neu_ratio",
        "negative": "pred_neg_ratio"
    })

    daily = daily.merge(
        pred_dist[["Date", "pred_pos_ratio", "pred_neu_ratio", "pred_neg_ratio"]],
        on="Date",
        how="left"
    )

    label_df = scored_df[["Date", "Label"]].drop_duplicates()
    if "Label" in scored_df.columns:
        daily = daily.merge(label_df, on="Date", how="left")

    return daily


def plot_monthly_sentiment(daily_df: pd.DataFrame) -> None:
    """Plot monthly average FinBERT sentiment."""
    monthly = daily_df.set_index("Date")[["pos_mean", "neg_mean", "neu_mean"]].resample("ME").mean()

    plt.figure()
    monthly["pos_mean"].plot(label="Positive")
    monthly["neg_mean"].plot(label="Negative")
    monthly["neu_mean"].plot(label="Neutral")
    plt.title("Monthly Average FinBERT Sentiment")
    plt.xlabel("Date")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "monthly_finbert_sentiment_mean.png", dpi=160)
    plt.close()


def plot_sentiment_by_label(daily_df: pd.DataFrame) -> None:
    """Plot average FinBERT sentiment by actual DJIA direction label."""
    if "Label" not in daily_df.columns:
        return

    grouped = daily_df.groupby("Label")[["pos_mean", "neg_mean", "neu_mean"]].mean()

    plt.figure()
    x = np.arange(len(grouped.index))
    width = 0.25

    plt.bar(x - width, grouped["pos_mean"].values, width=width, label="Positive")
    plt.bar(x, grouped["neg_mean"].values, width=width, label="Negative")
    plt.bar(x + width, grouped["neu_mean"].values, width=width, label="Neutral")

    plt.xticks(x, [f"Label {int(label)}" for label in grouped.index])
    plt.ylabel("Mean Probability")
    plt.title("FinBERT Mean Sentiment by DJIA Direction Label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "finbert_mean_by_label.png", dpi=160)
    plt.close()


def plot_prediction_distribution_by_label(daily_df: pd.DataFrame) -> None:
    """Plot predicted sentiment label distribution by actual DJIA direction label."""
    if "Label" not in daily_df.columns:
        return

    grouped = daily_df.groupby("Label")[
        ["pred_pos_ratio", "pred_neg_ratio", "pred_neu_ratio"]
    ].mean()

    plt.figure()
    x = np.arange(len(grouped.index))
    width = 0.25

    plt.bar(x - width, grouped["pred_pos_ratio"].values, width=width, label="Pred Positive")
    plt.bar(x, grouped["pred_neg_ratio"].values, width=width, label="Pred Negative")
    plt.bar(x + width, grouped["pred_neu_ratio"].values, width=width, label="Pred Neutral")

    plt.xticks(x, [f"Label {int(label)}" for label in grouped.index])
    plt.ylabel("Average Daily Ratio")
    plt.title("FinBERT Predicted Sentiment Distribution by Label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "finbert_pred_distribution_by_label.png", dpi=160)
    plt.close()


def plot_pos_neg_scatter(daily_df: pd.DataFrame) -> None:
    """Plot daily positive vs negative sentiment scores."""
    plt.figure()

    if "Label" in daily_df.columns:
        colors = daily_df["Label"].map({0: "tab:red", 1: "tab:blue"}).fillna("gray")
        plt.scatter(daily_df["pos_mean"], daily_df["neg_mean"], c=colors, alpha=0.6)
    else:
        plt.scatter(daily_df["pos_mean"], daily_df["neg_mean"], alpha=0.6)

    plt.xlabel("Daily Positive Mean")
    plt.ylabel("Daily Negative Mean")
    plt.title("FinBERT Positive vs Negative Sentiment")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "finbert_pos_vs_neg_scatter.png", dpi=160)
    plt.close()


def main():
    df = load_input(INPUT_PATH)
    mode, cols = detect_schema(df)

    print(f"[INFO] Input shape: {df.shape}")
    print(f"[INFO] Detected schema: {mode}")
    print(f"[INFO] Text columns: {cols}")

    headline_df = flatten_to_headlines(df, mode, cols)
    print(f"[INFO] Headline-level rows: {len(headline_df)}")

    sentiment_pipeline = build_finbert_pipeline()
    score_df = run_finbert(sentiment_pipeline, headline_df["headline"].tolist())

    scored_df = pd.concat(
        [headline_df.reset_index(drop=True), score_df.reset_index(drop=True)],
        axis=1
    )

    scored_path = TABLE_DIR / "headline_finbert_scores.csv"
    scored_df.to_csv(scored_path, index=False, encoding="utf-8")

    daily_df = aggregate_daily(scored_df)
    daily_path = TABLE_DIR / "daily_finbert_aggregates.csv"
    daily_df.to_csv(daily_path, index=False, encoding="utf-8")

    plot_monthly_sentiment(daily_df)
    plot_sentiment_by_label(daily_df)
    plot_prediction_distribution_by_label(daily_df)
    plot_pos_neg_scatter(daily_df)

    print("[SAVE]", scored_path)
    print("[SAVE]", daily_path)
    print("[DONE] FinBERT sentiment analysis completed.")


if __name__ == "__main__":
    main()
