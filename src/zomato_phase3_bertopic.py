"""
=============================================================================
ZOMATO PRODUCT INTELLIGENCE PLATFORM
Phase 3: Complaint Clustering — BERTopic
=============================================================================

Architecture:
  Negative reviews (from Phase 2)
    -> Sentence-BERT embeddings        (all-MiniLM-L6-v2)
    -> UMAP dimensionality reduction   (384d -> 10d for clustering, 2d for viz)
    -> HDBSCAN clustering              (density-based, no k needed)
    -> c-TF-IDF topic representation   (class-based TF-IDF)
    -> KeyBERT re-ranking              (embedding-aware keyword selection)
    -> Human-readable label mapping
    -> 5 charts saved to output/charts/

Install:
    pip install bertopic sentence-transformers umap-learn hdbscan
                scikit-learn pandas numpy matplotlib seaborn

Run:
    python zomato_phase3_bertopic.py

Inputs:
    output/zomato_sentiment.csv    (Phase 2 output)

Outputs:
    output/zomato_topics.csv
    output/complaint_topics.csv
    output/review_topic_assignments.csv
    output/topic_report.json
    output/charts/topic_frequency.png
    output/charts/topic_wordweights.png
    output/charts/topic_heatmap.png
    output/charts/topic_over_rating.png
    output/charts/topic_map.png
=============================================================================
"""

import ast
import json
import logging
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("zomato.bertopic")

# =============================================================================
# CHART PALETTE
# =============================================================================

PALETTE = {
    "bg":      "#FAFAF8",
    "text":    "#2C2C2A",
    "grid":    "#E8E6E0",
    "neutral": "#888780",
    "topic_colors": [
        "#534AB7","#D85A30","#0F6E56","#BA7517",
        "#185FA5","#993556","#3B6D11","#993C1D",
        "#3C3489","#854F0B",
    ],
}

def apply_style():
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor":   PALETTE["bg"],
        "axes.edgecolor":   PALETTE["grid"],
        "axes.labelcolor":  PALETTE["text"],
        "axes.titleweight": "500",
        "axes.titlesize":   13,
        "axes.labelsize":   11,
        "xtick.color":      PALETTE["text"],
        "ytick.color":      PALETTE["text"],
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "grid.color":       PALETTE["grid"],
        "grid.linewidth":   0.5,
        "text.color":       PALETTE["text"],
        "font.family":      "DejaVu Sans",
        "figure.dpi":       150,
        "savefig.bbox":     "tight",
        "savefig.facecolor":PALETTE["bg"],
    })


# =============================================================================
# STEP 1: LOAD & EXTRACT NEGATIVE REVIEWS
# =============================================================================

def load_negative_reviews(path: str):
    """
    Load Phase 2 output and extract only negative reviews for clustering.

    WHY only negative reviews?
      BERTopic on all reviews surfaces generic positive themes
      ("great food", "loved it") which are not actionable. We cluster
      complaints so PM teams know exactly what to fix.

    Returns:
      df         -- full restaurant DataFrame
      neg_texts  -- flat list of negative review strings
      neg_urls   -- parallel list of restaurant URLs for each review
                    (needed to map topics back to restaurants)
    """
    log.info(f"Loading sentiment data from {path}")
    df = pd.read_csv(path, dtype=str)

    for col in ["rate","votes","approx_cost","review_count","avg_signed_score","sentiment_negative_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def safe_parse(val):
        if pd.isna(val) or str(val).strip() in ["","[]","nan"]:
            return []
        try:
            return ast.literal_eval(str(val))
        except Exception:
            return []

    df["review_texts"] = df["review_texts"].apply(safe_parse) if "review_texts" in df.columns else [[] for _ in range(len(df))]

    neg_texts, neg_urls = [], []
    neg_mask = df.get("dominant_sentiment", pd.Series()) == "negative"

    for _, row in df[neg_mask].iterrows():
        for text in row["review_texts"]:
            cleaned = _clean_text(text)
            if len(cleaned.split()) >= 5:
                neg_texts.append(cleaned)
                neg_urls.append(row.get("url",""))

    # Fallback: if no dominant_sentiment column, use all reviews
    if not neg_texts:
        log.warning("No dominant_sentiment column found — using all reviews as input")
        for _, row in df.iterrows():
            for text in row["review_texts"]:
                cleaned = _clean_text(text)
                if len(cleaned.split()) >= 5:
                    neg_texts.append(cleaned)
                    neg_urls.append(row.get("url",""))

    log.info(f"Extracted {len(neg_texts):,} negative reviews for clustering")
    return df, neg_texts, neg_urls


def _clean_text(text: str) -> str:
    """
    Minimal cleaning for BERTopic.
    Keep natural language -- Sentence-BERT embeddings benefit from
    grammatical context. Only remove noise (URLs, special chars).
    """
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s.,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\.{2,}", ".", text)
    return text.strip()


# =============================================================================
# STEP 2: BUILD BERTOPIC MODEL
# =============================================================================

def build_bertopic_model():
    """
    Assemble BERTopic from its four components.

    Embedding model: all-MiniLM-L6-v2
      384-dimensional sentence embeddings. 5x faster than large models
      with ~5% accuracy trade-off. Designed for semantic similarity --
      perfect for grouping reviews that complain about the same thing
      in different words ("waited forever" and "took too long" cluster together).

    UMAP: n_components=10, n_neighbors=15, min_dist=0.0
      Reduces 384d -> 10d before clustering. We use 10d (not 2d) here
      because 2d UMAP distorts cluster shapes. The 2d projection for
      visualisation uses a separate UMAP fit.
      min_dist=0.0 produces tighter clusters -- better for HDBSCAN.

    HDBSCAN: min_cluster_size=15, min_samples=5
      Density-based: no need to pre-specify number of clusters.
      min_cluster_size=15: a complaint category needs >=15 examples to
      be a real pattern, not noise.
      cluster_selection_method="eom": Excess of Mass produces fewer,
      more stable clusters than "leaf" mode.
      Reviews that don't fit any cluster get topic -1 (noise).

    CountVectorizer: ngram_range=(1,2), stop_words="english"
      c-TF-IDF uses this to find representative words per topic.
      Bigrams capture "delivery delay", "cold food", "rude staff" which
      are more specific and meaningful than unigrams alone.
      min_df=5: ignores terms appearing in <5 reviews (typos, rare words).
      max_df=0.90: ignores terms in >90% of docs (too generic to distinguish).

    KeyBERTInspired representation:
      Re-ranks c-TF-IDF keywords using embedding similarity to the topic
      centroid. Produces more semantically coherent keywords than raw
      c-TF-IDF (e.g. "delivery delay" ranks over "the delivery").
    """
    try:
        from bertopic import BERTopic
        from bertopic.representation import KeyBERTInspired
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except ImportError as e:
        raise ImportError(
            "Install:\n  pip install bertopic sentence-transformers umap-learn hdbscan"
        ) from e

    log.info("Assembling BERTopic pipeline...")

    topic_model = BERTopic(
        embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
        umap_model=UMAP(
            n_components=10, n_neighbors=15, min_dist=0.0,
            metric="cosine", random_state=42, low_memory=False,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=15, min_samples=5,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        ),
        vectorizer_model=CountVectorizer(
            ngram_range=(1, 2), stop_words="english",
            min_df=5, max_df=0.90, max_features=10_000,
        ),
        representation_model=KeyBERTInspired(),
        nr_topics="auto",   # merge similar topics automatically
        top_n_words=10,
        verbose=True,
        calculate_probabilities=False,
    )
    log.info("BERTopic model assembled.")
    return topic_model


# =============================================================================
# STEP 3: FIT AND EXTRACT TOPICS
# =============================================================================

COMPLAINT_TAXONOMY = {
    "Delivery & wait time":   ["delivery","late","wait","slow","hour","delay","minutes","long","time"],
    "Food quality":           ["food","taste","bland","cold","stale","undercooked","quality","fresh","raw","bad"],
    "Portion & value":        ["portion","small","price","expensive","overpriced","worth","value","quantity","less"],
    "Staff & service":        ["staff","rude","service","behaviour","attitude","waiter","manager","ignored","arrogant"],
    "Hygiene & cleanliness":  ["dirty","hygiene","cockroach","clean","unhygienic","pest","smell","filthy"],
    "Order accuracy":         ["wrong","order","missing","incorrect","item","mistake","received","sent"],
    "Packaging":              ["packaging","spill","leak","damaged","broken","crushed","box","container"],
    "App & online ordering":  ["app","online","website","payment","cancel","refund","zomato","glitch","error"],
    "Ambience & seating":     ["ambience","noise","loud","seating","crowd","parking","dirty","table","space"],
    "Menu availability":      ["menu","unavailable","stock","item","option","variety","choice","offered"],
}

def _derive_label(keywords: list) -> str:
    """
    Map top keywords to a human-readable complaint category using
    the taxonomy above. Falls back to joining top-3 keywords if
    no taxonomy match scores above zero.

    WHY not use BERTopic auto-names?
      Auto-names look like "45_delivery_late_order" -- fine for debugging,
      bad for PM dashboards, internship reports, and RICE scoring (Phase 5).
    """
    kw_set = {k.lower() for k in keywords}
    best_label, best_score = "Other complaints", 0
    for label, signals in COMPLAINT_TAXONOMY.items():
        score = len(kw_set & set(signals))
        if score > best_score:
            best_score, best_label = score, label
    if best_score == 0 and keywords:
        best_label = " / ".join(keywords[:3]).title()
    return best_label


def fit_and_extract(topic_model, texts: list, urls: list):
    """
    Fit BERTopic and extract structured topic + review tables.

    Returns:
      review_topics_df  -- one row per review: text, topic_id, label
      topic_summary_df  -- one row per topic: id, label, keywords, frequency, pct
    """
    log.info(f"Fitting BERTopic on {len(texts):,} reviews...")
    topics, _ = topic_model.fit_transform(texts)

    topic_info = topic_model.get_topic_info()
    rows = []
    for _, row in topic_info.iterrows():
        tid   = int(row["Topic"])
        count = int(row["Count"])
        topic_words = topic_model.get_topic(tid) or []
        keywords   = [w for w, _ in topic_words[:10]]
        kw_weights = {w: round(float(s), 4) for w, s in topic_words[:10]}
        label = "Uncategorised / noise" if tid == -1 else _derive_label(keywords)
        rows.append({
            "topic_id":      tid,
            "topic_label":   label,
            "keywords":      keywords,
            "kw_weights":    kw_weights,
            "frequency":     count,
            "pct":           round(count / len(texts) * 100, 2),
        })

    topic_summary_df = pd.DataFrame(rows).sort_values("frequency", ascending=False)
    id_to_label = {r["topic_id"]: r["topic_label"] for _, r in topic_summary_df.iterrows()}

    review_topics_df = pd.DataFrame({
        "url":         urls,
        "review_text": texts,
        "topic_id":    topics,
        "topic_label": [id_to_label.get(t, "unknown") for t in topics],
    })

    n_topics = int((topic_summary_df["topic_id"] != -1).sum())
    log.info(f"Found {n_topics} complaint topics.")
    return review_topics_df, topic_summary_df


# =============================================================================
# STEP 4: MAP TOPICS TO RESTAURANTS
# =============================================================================

def map_to_restaurants(df: pd.DataFrame, review_topics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate review-level topic assignments back to restaurant level.

    New restaurant columns:
      dominant_complaint       -- most frequent complaint topic (excl. noise)
      complaint_diversity      -- number of distinct complaint topics
                                  (high = wide-spread problems; low = one root cause)
      complaint_review_count   -- total negative reviews that were clustered
      complaint_topic_counts   -- dict of {topic_label: count} for each restaurant

    WHY complaint_diversity matters for Phase 5 (RICE):
      A restaurant with 3 complaint topics is harder to fix than one with 1.
      Diversity is an Effort signal in the RICE formula.
    """
    clean = review_topics_df[review_topics_df["topic_id"] != -1]
    agg = (
        clean.groupby("url")["topic_label"]
        .agg(
            dominant_complaint=lambda x: x.value_counts().idxmax(),
            complaint_diversity="nunique",
            complaint_review_count="count",
        )
        .reset_index()
    )
    topic_counts = (
        clean.groupby(["url","topic_label"]).size()
        .reset_index(name="count")
        .groupby("url")
        .apply(lambda g: dict(zip(g["topic_label"], g["count"])))
        .reset_index()
        .rename(columns={0: "complaint_topic_counts"})
    )
    agg = agg.merge(topic_counts, on="url", how="left")
    df  = df.merge(agg, on="url", how="left")
    df["complaint_diversity"]     = df["complaint_diversity"].fillna(0).astype(int)
    df["complaint_review_count"]  = df["complaint_review_count"].fillna(0).astype(int)
    return df


# =============================================================================
# STEP 5: CHARTS
# =============================================================================

def save_chart(fig, name: str, charts_dir: Path):
    p = charts_dir / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Chart saved -> {p}")


def chart_topic_frequency(topic_summary_df: pd.DataFrame, charts_dir: Path):
    """
    Horizontal bar: complaint topic frequency, annotated with % of all negative reviews.
    Sorted ascending so the most frequent topic is at the top (natural reading order).
    """
    apply_style()
    sub = topic_summary_df[topic_summary_df["topic_id"] != -1].head(10).sort_values("frequency")
    colors = [PALETTE["topic_colors"][i % len(PALETTE["topic_colors"])] for i in range(len(sub))]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(sub["topic_label"], sub["frequency"], color=colors, height=0.6, alpha=0.88)
    max_freq = sub["frequency"].max()
    for bar, freq, pct in zip(bars, sub["frequency"], sub["pct"]):
        ax.text(bar.get_width() + max_freq * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{freq:,}  ({pct:.1f}%)", va="center", fontsize=8.5, color=PALETTE["text"])
    ax.set_xlabel("Number of reviews")
    ax.set_title("Complaint topic frequency  (negative reviews only)", pad=14)
    ax.set_xlim(0, max_freq * 1.20)
    ax.grid(axis="x", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_chart(fig, "topic_frequency.png", charts_dir)


def chart_topic_keyword_weights(topic_summary_df: pd.DataFrame, charts_dir: Path):
    """
    Small-multiples: one mini bar chart per topic showing c-TF-IDF keyword weights.
    Bigrams like "delivery delay" and "cold food" appear here -- richer than unigrams.
    """
    apply_style()
    topics = topic_summary_df[topic_summary_df["topic_id"] != -1].head(8)
    n    = len(topics)
    cols = 2
    rows = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2.8))
    axes = axes.flatten() if n > 1 else [axes]

    for i, (_, row) in enumerate(topics.iterrows()):
        ax    = axes[i]
        kw    = row["kw_weights"]
        if not kw:
            ax.set_visible(False)
            continue
        words   = list(kw.keys())[:8]
        weights = list(kw.values())[:8]
        color   = PALETTE["topic_colors"][i % len(PALETTE["topic_colors"])]
        ax.barh(words, weights, color=color, alpha=0.82, height=0.6)
        ax.set_title(row["topic_label"], fontsize=10, fontweight="500", pad=6)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("c-TF-IDF weight", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top keywords per complaint topic  (c-TF-IDF + KeyBERT weights)",
                 fontsize=13, fontweight="500", y=1.01)
    plt.tight_layout(h_pad=2.5, w_pad=2.0)
    save_chart(fig, "topic_wordweights.png", charts_dir)


def chart_topic_heatmap(df: pd.DataFrame, topic_summary_df: pd.DataFrame, charts_dir: Path):
    """
    Heatmap: complaint topic (row) x restaurant type (column).
    Cell = % of that rest_type's complaints in each topic.
    WHY: reveals which restaurant types have which complaint patterns.
    "Delivery & wait time" being high for Quick Bites but not Fine Dining
    is a very different product problem requiring different solutions.
    """
    apply_style()
    if "rest_type" not in df.columns or "dominant_complaint" not in df.columns:
        log.warning("Skipping heatmap: missing rest_type or dominant_complaint columns")
        return

    top_topics = topic_summary_df[topic_summary_df["topic_id"] != -1].head(8)["topic_label"].tolist()
    top_types  = df["rest_type"].value_counts().head(7).index.tolist()
    sub        = df[df["dominant_complaint"].isin(top_topics) & df["rest_type"].isin(top_types)]

    pivot = sub.groupby(["dominant_complaint","rest_type"]).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=0), axis=1) * 100
    pivot_pct = pivot_pct.reindex(columns=[c for c in top_types if c in pivot_pct.columns]).fillna(0)

    cmap = LinearSegmentedColormap.from_list(
        "complaint", ["#F1EFE8","#F0997B","#D85A30","#712B13"]
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(pivot_pct.values, cmap=cmap, aspect="auto", vmin=0)
    ax.set_xticks(range(len(pivot_pct.columns)))
    ax.set_xticklabels(pivot_pct.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot_pct.index)))
    ax.set_yticklabels(pivot_pct.index, fontsize=9)
    for ri in range(len(pivot_pct.index)):
        for ci in range(len(pivot_pct.columns)):
            val = pivot_pct.values[ri, ci]
            ax.text(ci, ri, f"{val:.0f}%", ha="center", va="center",
                    fontsize=8, color="white" if val > 30 else PALETTE["text"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("% of complaints", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("Complaint topic distribution by restaurant type", pad=12)
    save_chart(fig, "topic_heatmap.png", charts_dir)


def chart_topics_over_rating(df: pd.DataFrame, topic_summary_df: pd.DataFrame, charts_dir: Path):
    """
    Line chart: how top-5 complaint topics distribute across rating tiers.
    WHY: "Hygiene" spikes at 1-2 stars (severe); "Portion size" appears
    even at 3-4 stars (moderate but persistent). Different urgency signals.
    """
    apply_style()
    if "rate" not in df.columns or "dominant_complaint" not in df.columns:
        return

    top5 = topic_summary_df[topic_summary_df["topic_id"] != -1].head(5)["topic_label"].tolist()
    bins   = [1.0, 2.0, 3.0, 3.5, 4.0, 5.0]
    labels = ["1-2 stars","2-3 stars","3-3.5 stars","3.5-4 stars","4-5 stars"]

    df2 = df.dropna(subset=["rate","dominant_complaint"]).copy()
    df2 = df2[df2["dominant_complaint"].isin(top5)]
    df2["rating_bin"] = pd.cut(df2["rate"], bins=bins, labels=labels, right=True)

    pivot = df2.groupby(["rating_bin","dominant_complaint"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(labels).fillna(0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, topic in enumerate(top5):
        if topic in pivot_pct.columns:
            ax.plot(labels, pivot_pct[topic], marker="o",
                    color=PALETTE["topic_colors"][i], linewidth=1.8, markersize=5, label=topic)
    ax.set_xlabel("Rating tier")
    ax.set_ylabel("% of complaints in tier")
    ax.set_title("Complaint topic mix across rating tiers", pad=12)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_chart(fig, "topic_over_rating.png", charts_dir)


def chart_topic_map(topic_model, texts: list, charts_dir: Path):
    """
    2D UMAP scatter: reviews projected to 2D, coloured by topic.
    Cluster separation quality is visible at a glance.
    Well-separated blobs = model found meaningful, distinct complaint categories.

    Uses a fresh 2D UMAP fit (separate from the 10D used for clustering)
    because 2D UMAP is for visualisation only -- it distorts cluster shapes.
    """
    apply_style()
    try:
        from umap import UMAP
    except ImportError:
        log.warning("UMAP not available -- skipping topic map")
        return

    sample = texts[:3000]
    log.info(f"Encoding {len(sample)} reviews for 2D UMAP map...")
    embeddings = topic_model.embedding_model.encode(
        sample, show_progress_bar=True, batch_size=128
    )
    coords = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                  metric="cosine", random_state=42).fit_transform(embeddings)

    sampled_topics, _ = topic_model.transform(sample)
    unique_topics     = sorted(set(sampled_topics))
    color_map = {
        t: (PALETTE["neutral"] if t == -1
            else PALETTE["topic_colors"][i % len(PALETTE["topic_colors"])])
        for i, t in enumerate(unique_topics)
    }
    point_colors = [color_map[t] for t in sampled_topics]
    point_alpha  = [0.12 if t == -1 else 0.55 for t in sampled_topics]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:, 0], coords[:, 1], c=point_colors,
               alpha=point_alpha, s=8, linewidths=0)

    topic_info = topic_model.get_topic_info()
    name_map = dict(zip(topic_info["Topic"], topic_info["Name"]))
    for tid in unique_topics:
        if tid == -1:
            continue
        mask = np.array(sampled_topics) == tid
        if mask.sum() < 5:
            continue
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        raw = str(name_map.get(tid, f"t{tid}"))
        short = raw.split("_")[1] if "_" in raw else raw[:12]
        ax.text(cx, cy, short[:16], fontsize=7.5, ha="center", va="center",
                color=PALETTE["text"], fontweight="500",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.65))

    ax.set_title("2D UMAP projection of complaint topic clusters", pad=12)
    ax.set_xlabel("UMAP dim 1", fontsize=9)
    ax.set_ylabel("UMAP dim 2", fontsize=9)
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.spines[["top","right","bottom","left"]].set_visible(False)
    save_chart(fig, "topic_map.png", charts_dir)


# =============================================================================
# STEP 6: SAVE OUTPUTS
# =============================================================================

def save_outputs(df, topic_summary_df, review_topics_df, out_path: Path):
    out_path.mkdir(parents=True, exist_ok=True)

    # Restaurant-level (Phase 4 input)
    save_df = df.copy()
    for col in ["review_texts","reviews_list","complaint_topic_counts"]:
        if col in save_df.columns:
            save_df[col] = save_df[col].astype(str)
    save_df.to_csv(out_path / "zomato_topics.csv", index=False)

    # Topic summary (dashboard + Phase 5)
    ts = topic_summary_df.copy()
    ts["keywords"]   = ts["keywords"].apply(lambda k: ", ".join(k) if isinstance(k, list) else k)
    ts["kw_weights"] = ts["kw_weights"].astype(str)
    ts.to_csv(out_path / "complaint_topics.csv", index=False)

    # Per-review assignments (Phase 4 severity scoring)
    review_topics_df.to_csv(out_path / "review_topic_assignments.csv", index=False)

    # JSON summary
    report = {
        "n_topics": int((topic_summary_df["topic_id"] != -1).sum()),
        "n_reviews_clustered": int(
            topic_summary_df.loc[topic_summary_df["topic_id"] != -1,"frequency"].sum()
        ),
        "noise_pct": float(
            topic_summary_df.loc[topic_summary_df["topic_id"] == -1,"pct"].values[0]
            if (topic_summary_df["topic_id"] == -1).any() else 0
        ),
        "topics": topic_summary_df[topic_summary_df["topic_id"] != -1][
            ["topic_id","topic_label","frequency","pct"]
        ].to_dict(orient="records"),
    }
    with open(out_path / "topic_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Outputs saved. {report['n_topics']} topics, {report['n_reviews_clustered']:,} reviews clustered.")


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

def run_topic_pipeline(
    sentiment_csv: str = "output/zomato_sentiment.csv",
    output_dir:    str = "output",
):
    log.info("=" * 60)
    log.info("ZOMATO PHASE 3: BERTOPIC COMPLAINT CLUSTERING")
    log.info("=" * 60)

    out_path   = Path(output_dir)
    charts_dir = out_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    df, neg_texts, neg_urls = load_negative_reviews(sentiment_csv)

    if len(neg_texts) < 50:
        log.warning(f"Only {len(neg_texts)} negative reviews. BERTopic needs >=50 for meaningful clusters.")

    topic_model = build_bertopic_model()
    review_topics_df, topic_summary_df = fit_and_extract(topic_model, neg_texts, neg_urls)
    df = map_to_restaurants(df, review_topics_df)

    log.info("Generating charts...")
    chart_topic_frequency(topic_summary_df, charts_dir)
    chart_topic_keyword_weights(topic_summary_df, charts_dir)
    chart_topic_heatmap(df, topic_summary_df, charts_dir)
    chart_topics_over_rating(df, topic_summary_df, charts_dir)
    chart_topic_map(topic_model, neg_texts, charts_dir)

    save_outputs(df, topic_summary_df, review_topics_df, out_path)

    log.info("=" * 60)
    log.info("PHASE 3 COMPLETE")
    log.info("=" * 60)
    return df, topic_summary_df


# =============================================================================
# CONSOLE SUMMARY TABLE
# =============================================================================

def print_topic_table(topic_summary_df: pd.DataFrame):
    print("\n" + "=" * 74)
    print(f"{'ID':<5} {'Topic Label':<32} {'Top 5 Keywords':<27} {'Freq':>6}  {'%':>5}")
    print("-" * 74)
    for _, row in topic_summary_df[topic_summary_df["topic_id"] != -1].iterrows():
        kws = ", ".join(row["keywords"][:5]) if isinstance(row["keywords"], list) else ""
        print(f"{row['topic_id']:<5} {row['topic_label'][:31]:<32} "
              f"{kws[:26]:<27} {row['frequency']:>6,}  {row['pct']:>4.1f}%")
    print("=" * 74)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    df_topics, topic_summary = run_topic_pipeline(
        sentiment_csv="output/zomato_sentiment.csv",
        output_dir="output",
    )
    print_topic_table(topic_summary)

    # Hand off to Phase 4
    # from src.pain_points import run_pain_point_pipeline
    # df_pain = run_pain_point_pipeline(df_topics, topic_summary)
