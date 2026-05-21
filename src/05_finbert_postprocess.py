"""
05_finbert_postprocess.py

Purpose:
    Generate daily FinBERT aggregates and visualizations
    from an already-created headline_finbert_scores.csv file.

Input:
    outputs/tables/headline_finbert_scores.csv

Output:
    outputs/tables/daily_finbert_aggregates.csv
    outputs/figures/monthly_finbert_sentiment_mean.png
    outputs/figures/finbert_mean_by_label.png
    outputs/figures/finbert_pred_distribution_by_label.png
    outputs/figures/finbert_pos_vs_neg_scatter.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TABLE_DIR = Path("outputs/tables")
FIGURE_DIR = Path("outputs/figures")

SCORED_PATH = TABLE_DIR / "headline_finbert_scores.csv"
DAILY_PATH = TABLE_DIR / "daily_finbert_aggregates.csv"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_scored_data(path: Path) -> pd.DataFrame:
    """Load headline-level FinBERT scores."""
    if not path.exists():
        raise FileNotFoundError(
            f"FinBERT score file not found: {path}\n"
            "Run src/04_finbert_sentiment_analysis.py first."
        )

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def aggregate_daily(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate headline-level FinBERT scores into daily metrics."""

    daily = (
        scored_df
        .groupby("Date")
        .agg(
            pos_mean=("positive", "mean"),
            pos_median=("positive", "median"),
            neu_mean=("neutral", "mean"),
            neu_median=("neutral", "median"),
            neg_mean=("negative", "mean"),
            neg_median=("negative", "median"),
            headline_count=("headline", "count")
        )
        .reset_index()
    )

    pred_dist = (
        scored_df
        .groupby(["Date", "label_pred"])
        .size()
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

    if "Label" in scored_df.columns:
        label_df = scored_df[["Date", "Label"]].drop_duplicates()
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
    scored_df = load_scored_data(SCORED_PATH)

    print(f"[INFO] Loaded FinBERT scores: {scored_df.shape}")

    daily_df = aggregate_daily(scored_df)
    daily_df.to_csv(DAILY_PATH, index=False, encoding="utf-8")

    plot_monthly_sentiment(daily_df)
    plot_sentiment_by_label(daily_df)
    plot_prediction_distribution_by_label(daily_df)
    plot_pos_neg_scatter(daily_df)

    print(f"[SAVE] {DAILY_PATH}")
    print("[SAVE] FinBERT figures saved to outputs/figures")
    print("[DONE] FinBERT postprocessing completed.")


if __name__ == "__main__":
    main()
