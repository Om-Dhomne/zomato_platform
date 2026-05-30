"""
=============================================================================
ZOMATO PRODUCT INTELLIGENCE PLATFORM
Phase 2: Sentiment Analysis Pipeline — DistilBERT
=============================================================================

Architecture:
  DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
    → per-review positive/negative logits
    → mapped to positive / negative / neutral (confidence threshold)
    → aggregated per restaurant
    → charts saved to output/charts/

Install:
    pip install transformers torch pandas numpy matplotlib seaborn tqdm

Run:
    python zomato_phase2_sentiment.py

Inputs:
    output/zomato_cleaned.csv   (Phase 1 output)

Outputs:
    output/zomato_sentiment.csv
    output/charts/sentiment_distribution.png
    output/charts/sentiment_by_rest_type.png
    output/charts/sentiment_vs_rating.png
    output/charts/top_negative_restaurants.png
    output/charts/review_score_scatter.png
=============================================================================
"""

import ast
import json
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("zomato.sentiment")

# =============================================================================
# CHART STYLE
# =============================================================================

PALETTE = {
    "positive": "#1D9E75",   # teal
    "neutral":  "#888780",   # grey
    "negative": "#D85A30",   # coral
    "bg":       "#FAFAF8",
    "text":     "#2C2C2A",
    "grid":     "#E8E6E0",
}

def apply_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["bg"],
        "axes.edgecolor":    PALETTE["grid"],
        "axes.labelcolor":   PALETTE["text"],
        "axes.titleweight":  "500",
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "xtick.color":       PALETTE["text"],
        "ytick.color":       PALETTE["text"],
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "grid.color":        PALETTE["grid"],
        "grid.linewidth":    0.6,
        "text.color":        PALETTE["text"],
        "font.family":       "DejaVu Sans",
        "figure.dpi":        150,
        "savefig.bbox":      "tight",
        "savefig.facecolor": PALETTE["bg"],
    })

# =============================================================================
# STEP 1: LOAD CLEANED DATA
# =============================================================================

def load_cleaned(path: str) -> pd.DataFrame:
    log.info(f"Loading cleaned data from {path}")
    df = pd.read_csv(path, dtype=str)

    # review_texts was saved as a string representation of a list
    if "review_texts" in df.columns:
        def safe_parse(val):
            if pd.isna(val) or str(val).strip() in ["", "[]", "nan"]:
                return []
            try:
                return ast.literal_eval(str(val))
            except Exception:
                return []
        df["review_texts"] = df["review_texts"].apply(safe_parse)
    else:
        df["review_texts"] = [[] for _ in range(len(df))]

    # Restore numeric columns
    for col in ["rate", "votes", "approx_cost", "review_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info(f"Loaded {len(df):,} rows")
    return df


# =============================================================================
# STEP 2: DISTILBERT SENTIMENT ANALYSER
# =============================================================================

class DistilBERTSentimentAnalyser:
    """
    Wraps HuggingFace pipeline for DistilBERT SST-2.

    Model: distilbert-base-uncased-finetuned-sst-2-english
      - 40% smaller than BERT-base, 97% of BERT performance on GLUE
      - Pre-trained on SST-2 (Stanford Sentiment Treebank): movie reviews
      - Works well on short restaurant review sentences
      - Output labels: POSITIVE / NEGATIVE (binary)

    Neutral detection:
      SST-2 is binary (no neutral class). We recover neutral by thresholding
      confidence: if max(P_pos, P_neg) < NEUTRAL_THRESHOLD, we label it
      neutral. Default threshold = 0.75.

      WHY 0.75: empirically, reviews like "The food was okay" score
      ~0.60–0.73 confidence on either class — below the threshold.
      Strongly positive reviews like "Absolutely loved it!" score 0.97+.

    Batching:
      Processes reviews in batches of BATCH_SIZE for GPU/CPU efficiency.
      Max token length = 128 (covers 99% of short restaurant reviews;
      DistilBERT max is 512 but longer = slower with no accuracy gain here).
    """

    NEUTRAL_THRESHOLD = 0.75
    BATCH_SIZE        = 32
    MAX_LENGTH        = 128
    MODEL_NAME        = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self):
        log.info(f"Loading model: {self.MODEL_NAME}")
        try:
            from transformers import pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1
            device_label = "GPU" if device == 0 else "CPU"
            log.info(f"Running inference on: {device_label}")
            self._pipe = pipeline(
                "sentiment-analysis",
                model=self.MODEL_NAME,
                device=device,
                truncation=True,
                max_length=self.MAX_LENGTH,
            )
        except ImportError as e:
            raise ImportError(
                "Install transformers and torch:\n"
                "  pip install transformers torch"
            ) from e
        log.info("Model loaded.")

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Returns list of dicts: {label, score, sentiment}
          label     — raw model output: "POSITIVE" or "NEGATIVE"
          score     — confidence [0, 1]
          sentiment — mapped: "positive" / "negative" / "neutral"
        """
        if not texts:
            return []

        # Clean texts: remove newlines, strip, truncate very long strings
        cleaned = [str(t).replace("\n", " ").strip()[:512] for t in texts]

        results = []
        for i in range(0, len(cleaned), self.BATCH_SIZE):
            batch = cleaned[i : i + self.BATCH_SIZE]
            batch_results = self._pipe(batch)
            results.extend(batch_results)

        output = []
        for r in results:
            label = r["label"]          # "POSITIVE" or "NEGATIVE"
            score = float(r["score"])   # confidence of the predicted label

            if score < self.NEUTRAL_THRESHOLD:
                sentiment = "neutral"
            elif label == "POSITIVE":
                sentiment = "positive"
            else:
                sentiment = "negative"

            output.append({
                "label":     label,
                "score":     round(score, 4),
                "sentiment": sentiment,
            })
        return output


# =============================================================================
# STEP 3: RUN PER-REVIEW INFERENCE
# =============================================================================

def run_review_level_sentiment(
    df: pd.DataFrame,
    analyser: DistilBERTSentimentAnalyser,
) -> pd.DataFrame:
    """
    Explodes df (one row per restaurant) into one row per review,
    runs DistilBERT on all reviews in batches, then re-aggregates.

    Why explode-then-aggregate rather than per-restaurant?
      Per-restaurant: we'd concatenate all review texts into one long
      string, losing per-sentence signal and blowing the token limit.
      Explode: each review gets its own inference → accurate per-review
      label → meaningful aggregation statistics.

    Returns df with new columns per restaurant:
      sentiment_positive_pct  — % of reviews labelled positive
      sentiment_negative_pct  — % of reviews labelled negative
      sentiment_neutral_pct   — % of reviews labelled neutral
      dominant_sentiment      — majority class
      avg_sentiment_score     — mean confidence-weighted signed score
                                 (+1 = fully positive, -1 = fully negative)
      sentiment_volatility    — std of signed scores (high = mixed reviews)
    """
    log.info("Exploding reviews to per-review rows...")

    # Explode: one row per review text
    review_df = df[["name", "url", "review_texts"]].copy()
    review_df = review_df.explode("review_texts").dropna(subset=["review_texts"])
    review_df = review_df[review_df["review_texts"].str.strip() != ""]
    review_df = review_df.reset_index(drop=True)
    review_df = review_df.rename(columns={"review_texts": "review_text"})

    log.info(f"Running DistilBERT on {len(review_df):,} reviews...")

    all_texts = review_df["review_text"].tolist()
    predictions = []
    for i in tqdm(range(0, len(all_texts), analyser.BATCH_SIZE),
                  desc="Sentiment inference", unit="batch"):
        batch = all_texts[i : i + analyser.BATCH_SIZE]
        predictions.extend(analyser.predict_batch(batch))

    review_df["sentiment"]       = [p["sentiment"] for p in predictions]
    review_df["confidence"]      = [p["score"]     for p in predictions]
    review_df["raw_label"]       = [p["label"]     for p in predictions]

    # Signed score: positive → +confidence, negative → -confidence, neutral → 0
    def signed_score(row):
        if row["sentiment"] == "positive":
            return row["confidence"]
        elif row["sentiment"] == "negative":
            return -row["confidence"]
        return 0.0

    review_df["signed_score"] = review_df.apply(signed_score, axis=1)

    # Aggregate back to restaurant level
    log.info("Aggregating to restaurant level...")

    agg = review_df.groupby("url").agg(
        total_reviews       = ("sentiment", "count"),
        positive_count      = ("sentiment", lambda x: (x == "positive").sum()),
        negative_count      = ("sentiment", lambda x: (x == "negative").sum()),
        neutral_count       = ("sentiment", lambda x: (x == "neutral").sum()),
        avg_signed_score    = ("signed_score", "mean"),
        sentiment_volatility= ("signed_score", "std"),
    ).reset_index()

    agg["sentiment_volatility"] = agg["sentiment_volatility"].fillna(0)

    for col in ["positive", "negative", "neutral"]:
        agg[f"sentiment_{col}_pct"] = (
            agg[f"{col}_count"] / agg["total_reviews"] * 100
        ).round(1)

    agg["dominant_sentiment"] = agg[
        ["positive_count", "negative_count", "neutral_count"]
    ].idxmax(axis=1).str.replace("_count", "")

    agg["avg_signed_score"] = agg["avg_signed_score"].round(4)

    # Merge back to main df
    df = df.merge(agg, on="url", how="left")

    # Restaurants with no reviews get "neutral" defaults
    fill_map = {
        "total_reviews":          0,
        "positive_count":         0,
        "negative_count":         0,
        "neutral_count":          0,
        "sentiment_positive_pct": 0.0,
        "sentiment_negative_pct": 0.0,
        "sentiment_neutral_pct":  0.0,
        "avg_signed_score":       0.0,
        "sentiment_volatility":   0.0,
        "dominant_sentiment":     "neutral",
    }
    for col, default in fill_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    n_positive = (df["dominant_sentiment"] == "positive").sum()
    n_negative = (df["dominant_sentiment"] == "negative").sum()
    n_neutral  = (df["dominant_sentiment"] == "neutral").sum()
    log.info(f"Dominant sentiment: +{n_positive} pos / -{n_negative} neg / ~{n_neutral} neutral")

    return df, review_df   # also return review_df for chart use


# =============================================================================
# STEP 4: CHARTS
# =============================================================================

def save_chart(fig, filename: str, charts_dir: Path):
    path = charts_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Chart saved → {path}")


def chart_sentiment_distribution(df: pd.DataFrame, charts_dir: Path):
    """
    Donut chart: overall distribution of dominant_sentiment across all restaurants.
    WHY donut over bar: part-of-whole story; 3 categories reads cleanly as a donut.
    """
    apply_style()
    counts = df["dominant_sentiment"].value_counts()
    labels = ["positive", "neutral", "negative"]
    values = [counts.get(l, 0) for l in labels]
    colors = [PALETTE["positive"], PALETTE["neutral"], PALETTE["negative"]]

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.78,
        wedgeprops={"linewidth": 2, "edgecolor": PALETTE["bg"]},
    )
    for at in autotexts:
        at.set(fontsize=10, color="white", fontweight="500")

    centre_circle = plt.Circle((0, 0), 0.55, fc=PALETTE["bg"])
    ax.add_patch(centre_circle)

    total = sum(values)
    ax.text(0, 0.08, f"{total:,}", ha="center", va="center",
            fontsize=18, fontweight="500", color=PALETTE["text"])
    ax.text(0, -0.18, "restaurants", ha="center", va="center",
            fontsize=9, color=PALETTE["neutral"])

    ax.legend(
        wedges, [f"{l.title()} ({v:,})" for l, v in zip(labels, values)],
        loc="lower center", bbox_to_anchor=(0.5, -0.08),
        ncol=3, frameon=False, fontsize=9,
    )
    ax.set_title("Sentiment distribution across restaurants", pad=16)
    save_chart(fig, "sentiment_distribution.png", charts_dir)


def chart_sentiment_by_rest_type(df: pd.DataFrame, charts_dir: Path):
    """
    Stacked horizontal bar chart: sentiment breakdown by restaurant type.
    WHY stacked horizontal: easy comparison across categories; labels readable.
    Shows top 8 rest_types by count.
    """
    apply_style()
    if "rest_type" not in df.columns:
        return

    top_types = df["rest_type"].value_counts().head(8).index.tolist()
    sub = df[df["rest_type"].isin(top_types)].copy()

    grouped = sub.groupby("rest_type")[
        ["sentiment_positive_pct", "sentiment_neutral_pct", "sentiment_negative_pct"]
    ].mean().round(1)
    grouped = grouped.loc[grouped.sum(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(grouped))
    bar_h = 0.55

    ax.barh(y, grouped["sentiment_positive_pct"], bar_h,
            color=PALETTE["positive"], label="Positive")
    ax.barh(y, grouped["sentiment_neutral_pct"], bar_h,
            left=grouped["sentiment_positive_pct"],
            color=PALETTE["neutral"], label="Neutral")
    ax.barh(y, grouped["sentiment_negative_pct"], bar_h,
            left=grouped["sentiment_positive_pct"] + grouped["sentiment_neutral_pct"],
            color=PALETTE["negative"], label="Negative")

    ax.set_yticks(y)
    ax.set_yticklabels(grouped.index, fontsize=9)
    ax.set_xlabel("Average % of reviews")
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.axvline(50, color=PALETTE["grid"], linewidth=1, linestyle="--")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("Sentiment breakdown by restaurant type", pad=12)
    ax.grid(axis="x", alpha=0.4)
    ax.set_axisbelow(True)
    save_chart(fig, "sentiment_by_rest_type.png", charts_dir)


def chart_sentiment_vs_rating(df: pd.DataFrame, charts_dir: Path):
    """
    Scatter plot: avg_signed_score (y) vs rate (x), coloured by dominant_sentiment.
    WHY: validates that our DistilBERT scores correlate with Zomato's star rating.
         If they diverge, those restaurants are interesting anomalies for Phase 4.
    Alpha blending handles overplotting at scale.
    """
    apply_style()
    if "rate" not in df.columns:
        return

    sub = df.dropna(subset=["rate", "avg_signed_score"]).copy()
    sub = sub[sub["total_reviews"] > 0]

    color_map = {
        "positive": PALETTE["positive"],
        "negative": PALETTE["negative"],
        "neutral":  PALETTE["neutral"],
    }
    colors = sub["dominant_sentiment"].map(color_map).fillna(PALETTE["neutral"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sub["rate"], sub["avg_signed_score"],
               c=colors, alpha=0.35, s=18, linewidths=0)

    # Trend line
    m, b = np.polyfit(sub["rate"].dropna(), sub["avg_signed_score"].dropna(), 1)
    x_line = np.linspace(sub["rate"].min(), sub["rate"].max(), 100)
    ax.plot(x_line, m * x_line + b, color=PALETTE["text"],
            linewidth=1.2, linestyle="--", alpha=0.6, label=f"trend (slope={m:.2f})")

    ax.axhline(0, color=PALETTE["grid"], linewidth=0.8)
    ax.set_xlabel("Zomato star rating")
    ax.set_ylabel("DistilBERT avg signed score")
    ax.set_title("DistilBERT sentiment score vs Zomato rating", pad=12)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=PALETTE[s], markersize=8, label=s.title())
        for s in ["positive", "neutral", "negative"]
    ]
    legend_handles.append(
        plt.Line2D([0], [0], linestyle="--", color=PALETTE["text"],
                   linewidth=1.2, label=f"trend (slope={m:.2f})")
    )
    ax.legend(handles=legend_handles, frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    save_chart(fig, "sentiment_vs_rating.png", charts_dir)


def chart_top_negative_restaurants(df: pd.DataFrame, charts_dir: Path):
    """
    Horizontal bar: top 15 restaurants by highest negative review %.
    Only includes restaurants with ≥5 reviews (statistical minimum).
    WHY: directly actionable — these are the pain points for Phase 4.
    """
    apply_style()
    sub = df[df["total_reviews"] >= 5].copy()
    sub = sub.nlargest(15, "sentiment_negative_pct")[
        ["name", "sentiment_negative_pct", "sentiment_positive_pct", "total_reviews"]
    ].sort_values("sentiment_negative_pct")

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(sub))

    ax.barh(y, sub["sentiment_negative_pct"], 0.6,
            color=PALETTE["negative"], alpha=0.85, label="Negative %")
    ax.barh(y, sub["sentiment_positive_pct"], 0.6,
            left=sub["sentiment_negative_pct"],
            color=PALETTE["positive"], alpha=0.5, label="Positive %")

    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(row["sentiment_negative_pct"] + 0.5, i,
                f'{row["sentiment_negative_pct"]:.0f}%',
                va="center", fontsize=8, color=PALETTE["text"])

    ax.set_yticks(y)
    name_col = sub["name"].str[:30]
    ax.set_yticklabels(
        [f"{n}  ({int(r)} reviews)" for n, r in zip(name_col, sub["total_reviews"])],
        fontsize=8,
    )
    ax.set_xlabel("% of reviews")
    ax.set_title("Restaurants with highest negative sentiment\n(min. 5 reviews)", pad=12)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    save_chart(fig, "top_negative_restaurants.png", charts_dir)


def chart_score_histogram(review_df: pd.DataFrame, charts_dir: Path):
    """
    Histogram of per-review signed scores (–1 to +1).
    WHY: shows the full distribution of customer sentiment, not just restaurant-level
         averages. Bimodal = polarising restaurant. Left-skew = problem signal.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    pos_scores = review_df.loc[review_df["sentiment"] == "positive", "signed_score"]
    neg_scores = review_df.loc[review_df["sentiment"] == "negative", "signed_score"]
    neu_scores = review_df.loc[review_df["sentiment"] == "neutral",  "signed_score"]

    bins = np.linspace(-1, 1, 40)
    ax.hist(pos_scores, bins=bins, color=PALETTE["positive"],
            alpha=0.75, label=f"Positive ({len(pos_scores):,})")
    ax.hist(neg_scores, bins=bins, color=PALETTE["negative"],
            alpha=0.75, label=f"Negative ({len(neg_scores):,})")
    ax.hist(neu_scores, bins=bins, color=PALETTE["neutral"],
            alpha=0.6,  label=f"Neutral ({len(neu_scores):,})")

    ax.axvline(0, color=PALETTE["text"], linewidth=0.8, linestyle="--")
    ax.set_xlabel("Signed sentiment score  (−1 = max negative, +1 = max positive)")
    ax.set_ylabel("Review count")
    ax.set_title("Distribution of per-review sentiment scores", pad=12)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    save_chart(fig, "review_score_histogram.png", charts_dir)


# =============================================================================
# STEP 5: SAVE OUTPUTS
# =============================================================================

def save_outputs(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "zomato_sentiment.csv"

    # review_texts and reviews_list are list columns — stringify for CSV
    save_df = df.copy()
    for col in ["review_texts", "reviews_list"]:
        if col in save_df.columns:
            save_df[col] = save_df[col].astype(str)

    save_df.to_csv(path, index=False)
    log.info(f"Saved sentiment data → {path}")

    # Summary JSON
    summary = {
        "total_restaurants":   len(df),
        "restaurants_with_reviews": int((df["total_reviews"] > 0).sum()),
        "total_reviews_analysed": int(df["total_reviews"].sum()),
        "dominant_sentiment_counts": df["dominant_sentiment"].value_counts().to_dict(),
        "avg_signed_score_overall": round(df["avg_signed_score"].mean(), 4),
    }
    with open(output_dir / "sentiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Sentiment summary: {summary}")


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

def run_sentiment_pipeline(
    cleaned_csv: str  = "output/zomato_cleaned.csv",
    output_dir:  str  = "output",
) -> pd.DataFrame:

    log.info("=" * 60)
    log.info("ZOMATO PHASE 2: SENTIMENT ANALYSIS PIPELINE")
    log.info("=" * 60)

    out_path    = Path(output_dir)
    charts_dir  = out_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    df       = load_cleaned(cleaned_csv)
    analyser = DistilBERTSentimentAnalyser()

    df, review_df = run_review_level_sentiment(df, analyser)

    log.info("Generating charts...")
    chart_sentiment_distribution(df, charts_dir)
    chart_sentiment_by_rest_type(df, charts_dir)
    chart_sentiment_vs_rating(df, charts_dir)
    chart_top_negative_restaurants(df, charts_dir)
    chart_score_histogram(review_df, charts_dir)

    save_outputs(df, out_path)

    log.info("=" * 60)
    log.info("PHASE 2 COMPLETE")
    log.info(f"Charts saved to: {charts_dir}")
    log.info("=" * 60)
    return df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    df_sentiment = run_sentiment_pipeline(
        cleaned_csv="output/zomato_cleaned.csv",
        output_dir="output",
    )

    # Quick console summary
    print("\n=== SENTIMENT SUMMARY ===")
    print(df_sentiment["dominant_sentiment"].value_counts().to_string())
    print(f"\nAvg signed score: {df_sentiment['avg_signed_score'].mean():.3f}")
    print(f"Reviews analysed: {int(df_sentiment['total_reviews'].sum()):,}")

    # Hand off to Phase 3
    # from src.cluster import run_clustering_pipeline
    # df_clustered = run_clustering_pipeline(df_sentiment)
