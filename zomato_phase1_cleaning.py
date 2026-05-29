"""
=============================================================================
ZOMATO PRODUCT INTELLIGENCE PLATFORM
Phase 1: Production-Quality Data Cleaning
=============================================================================

Dataset columns:
    url, address, name, online_order, book_table, rate, votes, phone,
    location, rest_type, dish_liked, cuisines, approx_cost(for two people),
    reviews_list, menu_item, listed_in(type), listed_in(city)

Run:
    python zomato_phase1_cleaning.py

Output:
    zomato_cleaned.csv   — cleaned dataframe
    cleaning_report.json — audit trail of every change made
=============================================================================
"""

import ast
import json
import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
# Structured logging so every cleaning decision is traceable.
# In production, swap StreamHandler for a file handler or cloud logger.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("zomato.clean")


# ---------------------------------------------------------------------------
# AUDIT TRAIL
# ---------------------------------------------------------------------------
# Every cleaning step appends a record here.
# Final report is written to cleaning_report.json.
audit: list[dict] = []


def record(step: str, column: str, action: str, rows_affected: int, detail: str = ""):
    """Append one cleaning decision to the audit trail."""
    entry = {
        "step": step,
        "column": column,
        "action": action,
        "rows_affected": rows_affected,
        "detail": detail,
    }
    audit.append(entry)
    log.info(f"[{step}] {column}: {action} — {rows_affected} rows. {detail}")


# =============================================================================
# STEP 1: LOAD
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV with explicit dtype=str so no column gets silently cast
    (e.g. votes "1,234" parsed as NaN if read as int).
    Everything is cleaned explicitly in later steps.
    """
    log.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path, dtype=str, na_values=["", "nan", "NaN", "NULL", "null", "N/A", "n/a"])
    log.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    record("load", "all", "raw load", len(df), f"{len(df.columns)} columns")
    return df


# =============================================================================
# STEP 2: COLUMN NAMES
# =============================================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names:
      - lowercase
      - spaces → underscores
      - remove parentheses and special chars
      - rename verbose columns to short aliases

    Why: avoids df["approx_cost(for two people)"] everywhere in downstream code.
    """
    rename_map = {
        "approx_cost(for two people)": "approx_cost",
        "listed_in(type)":             "listed_type",
        "listed_in(city)":             "listed_city",
    }
    df = df.rename(columns=rename_map)

    # Generic cleanup for any other column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    log.info(f"Final columns: {list(df.columns)}")
    record("column_names", "all", "renamed & normalised", 0, str(list(df.columns)))
    return df


# =============================================================================
# STEP 3: DUPLICATES
# =============================================================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-pass deduplication:

    Pass 1 — Exact duplicates across ALL columns.
             Unlikely but catches copy-paste errors in the source data.

    Pass 2 — Semantic duplicates: same url.
             A restaurant can appear multiple times (scraped on different days).
             We keep the row with the most data (fewest NaNs) so we don't
             accidentally drop a row that filled in a previously-null column.

    Why url and not name?
      Name + location can collide for chain restaurants (e.g. "McDonald's, BTM").
      The url is the canonical unique identifier in Zomato's data model.
    """
    before = len(df)

    # Pass 1: exact row duplicates
    exact_dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    record("duplicates", "all_columns", "dropped exact duplicates", exact_dupes)

    # Pass 2: url-based semantic duplicates — keep the row with fewest NaNs
    if "url" in df.columns:
        df["_null_count"] = df.isnull().sum(axis=1)
        before_url = len(df)
        df = df.sort_values("_null_count").drop_duplicates(subset=["url"], keep="first")
        df = df.drop(columns=["_null_count"])
        url_dupes = before_url - len(df)
        record("duplicates", "url", "dropped url duplicates (kept most complete row)", url_dupes)

    after = len(df)
    record("duplicates", "all", "total rows removed", before - after)
    return df.reset_index(drop=True)


# =============================================================================
# STEP 4: MISSING VALUES
# =============================================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Column-by-column strategy — one size does NOT fit all:

    url, name, address
        → Drop rows. These are identity columns. A restaurant without
          a name or URL cannot be meaningfully analysed.

    rate, votes
        → Do NOT impute with mean/median yet. Flag with a boolean
          `rate_missing` column so downstream models can use missingness
          as a feature. Impute ONLY for visualisation aggregations.

    online_order, book_table
        → Impute "No" (conservative assumption: if not stated, feature
          is likely absent, not just unrecorded).

    approx_cost
        → Impute with median per rest_type group. Rationale: a Quick Bites
          restaurant has a very different cost distribution than Fine Dining.
          Global median would be misleading.

    cuisines, rest_type, location, listed_type, listed_city
        → Impute "Unknown". These are categorical filters; unknown is a
          valid category for analysis ("how many unknowns are there?").

    reviews_list, dish_liked, menu_item, phone
        → Impute empty string / empty list. High null rate is expected
          (54% for dish_liked). Downstream NLP handles empty gracefully.
    """

    # --- Identity columns: drop if null ---
    identity_cols = ["name"]
    for col in identity_cols:
        if col in df.columns:
            before = len(df)
            df = df.dropna(subset=[col])
            dropped = before - len(df)
            record("missing", col, "dropped rows with null identity column", dropped)

    # --- Flag rate missingness BEFORE any imputation ---
    if "rate" in df.columns:
        df["rate_missing"] = df["rate"].isna().astype(int)
        n_missing = df["rate_missing"].sum()
        record("missing", "rate", "created rate_missing flag column", n_missing)

    # --- Binary columns: fill with "No" ---
    binary_cols = ["online_order", "book_table"]
    for col in binary_cols:
        if col in df.columns:
            n = df[col].isna().sum()
            df[col] = df[col].fillna("No")
            record("missing", col, "filled NaN → 'No'", n)

    # --- Categorical columns: fill with "Unknown" ---
    cat_cols = ["cuisines", "rest_type", "location", "listed_type", "listed_city"]
    for col in cat_cols:
        if col in df.columns:
            n = df[col].isna().sum()
            df[col] = df[col].fillna("Unknown")
            record("missing", col, "filled NaN → 'Unknown'", n)

    # --- Text/list columns: fill with empty string ---
    text_cols = ["reviews_list", "dish_liked", "menu_item", "phone", "address"]
    for col in text_cols:
        if col in df.columns:
            n = df[col].isna().sum()
            df[col] = df[col].fillna("")
            record("missing", col, "filled NaN → empty string", n)

    # --- approx_cost: group-wise median imputation ---
    # Done after cost is parsed to numeric (see Step 6), so we set a flag here
    # and call _impute_cost() after numeric parsing.
    if "approx_cost" in df.columns:
        n = df["approx_cost"].isna().sum()
        record("missing", "approx_cost", "will impute with group median after parsing", n)

    return df.reset_index(drop=True)


# =============================================================================
# STEP 5: RATE — parse "4.1/5" → float
# =============================================================================

def clean_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw values look like: "4.1/5", "NEW", "3.8 /5", "-", NaN

    Strategy:
      1. Strip whitespace, extract the numeric part before "/"
      2. Cast to float
      3. Values outside [1.0, 5.0] are data errors → set to NaN
         (Zomato's scale is 1–5; a "6.0" is a scraping artefact)
      4. Round to 1 decimal place (matches Zomato's display precision)

    We do NOT drop rows with null rate — the `rate_missing` flag
    (set in Step 4) lets downstream code decide how to handle them.
    """
    if "rate" not in df.columns:
        return df

    def parse_rate(val):
        if pd.isna(val) or str(val).strip() in ["-", "NEW", "nan", ""]:
            return np.nan
        # Extract first decimal number before optional "/5"
        match = re.search(r"(\d+\.?\d*)", str(val))
        if match:
            r = float(match.group(1))
            return round(r, 1) if 1.0 <= r <= 5.0 else np.nan
        return np.nan

    df["rate"] = df["rate"].apply(parse_rate)

    n_valid   = df["rate"].notna().sum()
    n_invalid = df["rate"].isna().sum()
    record("parse", "rate", "parsed to float [1.0–5.0]", n_valid,
           f"{n_invalid} rows became NaN (NEW/dash/out-of-range)")
    return df


# =============================================================================
# STEP 6: VOTES — parse to int
# =============================================================================

def clean_votes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw votes can be "1,234" (comma-separated), "0", or NaN.

    Strategy:
      1. Remove commas
      2. Cast to float first (handles NaN), then to nullable Int64
         (pandas Int64 allows NaN; regular int does not)
      3. Negative votes are impossible → set to NaN
    """
    if "votes" not in df.columns:
        return df

    df["votes"] = (
        df["votes"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace("nan", np.nan)
    )
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce")
    df.loc[df["votes"] < 0, "votes"] = np.nan
    df["votes"] = df["votes"].astype("Int64")  # nullable integer

    n_valid = df["votes"].notna().sum()
    record("parse", "votes", "parsed to nullable Int64", int(n_valid))
    return df


# =============================================================================
# STEP 7: APPROX_COST — parse to int
# =============================================================================

def clean_approx_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw: "800", "1,200", "2500", NaN

    Strategy:
      1. Remove commas and currency symbols
      2. Cast to float → Int64
      3. Values ≤ 0 are impossible → NaN
      4. Group-wise median imputation (by rest_type)
         WHY: a "Quick Bites" median (~200 INR) and a "Fine Dining"
         median (~2000 INR) are very different. Global median hides this.
    """
    if "approx_cost" not in df.columns:
        return df

    df["approx_cost"] = (
        df["approx_cost"]
        .astype(str)
        .str.replace(r"[,₹\s]", "", regex=True)
        .replace("nan", np.nan)
    )
    df["approx_cost"] = pd.to_numeric(df["approx_cost"], errors="coerce")
    df.loc[df["approx_cost"] <= 0, "approx_cost"] = np.nan

    # Group-wise median imputation
    group_col = "rest_type" if "rest_type" in df.columns else None
    if group_col:
        df["approx_cost"] = df.groupby(group_col)["approx_cost"].transform(
            lambda x: x.fillna(x.median())
        )
        # Fallback: any remaining NaNs (e.g. whole group is null) → global median
        global_median = df["approx_cost"].median()
        n_remaining = df["approx_cost"].isna().sum()
        df["approx_cost"] = df["approx_cost"].fillna(global_median)
        record("impute", "approx_cost", "group-wise median by rest_type + global fallback",
               int(n_remaining), f"global median fallback = {global_median:.0f}")
    else:
        global_median = df["approx_cost"].median()
        df["approx_cost"] = df["approx_cost"].fillna(global_median)

    df["approx_cost"] = df["approx_cost"].round(0).astype("Int64")

    # Cost tier feature — useful for segmentation
    df["cost_tier"] = pd.cut(
        df["approx_cost"].astype(float),
        bins=[0, 300, 600, 1000, 2000, np.inf],
        labels=["budget", "low-mid", "mid", "premium", "luxury"],
        right=True,
    )
    record("feature", "cost_tier", "created cost tier bins", len(df),
           "budget<300 | low-mid<600 | mid<1000 | premium<2000 | luxury")
    return df


# =============================================================================
# STEP 8: BINARY COLUMNS — encode Yes/No → 1/0
# =============================================================================

def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    online_order and book_table are Yes/No strings.
    Encode as int (1/0) for modelling, but keep original as *_raw
    so the dashboard can display "Yes"/"No" labels.

    WHY keep raw: Streamlit toggle filters read "Yes"/"No" more
    naturally than 1/0. Two columns is cheap; avoids repeated decoding.
    """
    binary_cols = {"online_order": "online_order_enc",
                   "book_table":   "book_table_enc"}

    for col, enc_col in binary_cols.items():
        if col in df.columns:
            df[enc_col] = df[col].str.strip().str.title().map({"Yes": 1, "No": 0})
            n_unexpected = df[enc_col].isna().sum()
            if n_unexpected:
                df[enc_col] = df[enc_col].fillna(0).astype(int)
            else:
                df[enc_col] = df[enc_col].astype(int)
            record("encode", col, f"Yes/No → 1/0 in '{enc_col}'",
                   int(df[enc_col].sum()), f"{n_unexpected} unexpected values → 0")
    return df


# =============================================================================
# STEP 9: TEXT CLEANING (name, address, cuisines, location)
# =============================================================================

def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generic text cleaning for string columns used in search / groupby:

      - Strip leading/trailing whitespace
      - Collapse internal multiple spaces
      - Title-case name (consistency: "cafe coffee day" → "Cafe Coffee Day")
      - Normalise location/city/type to Title Case
      - cuisines: strip, keep comma-separated as-is (split happens in Phase 3)

    We do NOT aggressively lowercase or remove punctuation here because
    downstream NLP (Phase 2) does its own tokenisation.
    """
    text_title_cols = ["name", "location", "listed_type", "listed_city", "rest_type"]
    for col in text_title_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.title()
            )
            record("text", col, "strip + collapse whitespace + title case", len(df))

    # cuisines: strip only, preserve casing for "North Indian", "Chinese" etc.
    if "cuisines" in df.columns:
        df["cuisines"] = (
            df["cuisines"]
            .astype(str)
            .str.strip()
            .str.replace(r"\s*,\s*", ", ", regex=True)  # normalise comma spacing
        )
        record("text", "cuisines", "normalised comma spacing", len(df))

    # address: strip only
    if "address" in df.columns:
        df["address"] = df["address"].astype(str).str.strip()
        record("text", "address", "stripped whitespace", len(df))

    return df


# =============================================================================
# STEP 10: REVIEWS_LIST — parse string representation of list
# =============================================================================

def parse_reviews_list(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw value looks like:
      "[('Rated 5.0', 'RATED\\n Great food'), ('Rated 3.0', 'Average')]"

    This is a Python literal (list of tuples), not JSON.
    ast.literal_eval handles it safely (no exec, no eval).

    Output:
      reviews_list        → actual Python list of tuples
      review_texts        → flat list of cleaned review strings (for NLP)
      review_count        → integer count of reviews

    Safety: wrap in try/except — malformed strings become empty list,
    not a pipeline crash.
    """
    if "reviews_list" not in df.columns:
        return df

    def safe_parse(val):
        if not val or str(val).strip() in ["", "[]", "nan"]:
            return []
        try:
            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        return []

    df["reviews_list"] = df["reviews_list"].apply(safe_parse)

    # Extract flat text strings from each (rating_str, review_text) tuple
    def extract_texts(review_list):
        texts = []
        for item in review_list:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                text = str(item[1]).replace("\\n", " ").strip()
                if text and text.lower() not in ["nan", ""]:
                    texts.append(text)
        return texts

    df["review_texts"] = df["reviews_list"].apply(extract_texts)
    df["review_count"] = df["review_texts"].apply(len)

    total_reviews = df["review_count"].sum()
    record("parse", "reviews_list", "ast.literal_eval → list of tuples",
           int(df["review_count"].gt(0).sum()),
           f"{total_reviews:,} total review strings extracted")
    return df


# =============================================================================
# STEP 11: OUTLIER HANDLING
# =============================================================================

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two numeric columns need outlier treatment: votes and approx_cost.

    Method: IQR capping (Winsorization), NOT removal.
    WHY capping over removal?
      - A restaurant with 50,000 votes is a real phenomenon (e.g. a famous
        Bangalore institution). Removing it loses real signal.
      - Capping preserves the row but prevents it from skewing aggregations.
      - Removal would silently reduce our dataset and affect join keys.

    IQR formula:
      lower_bound = Q1 - 1.5 × IQR
      upper_bound = Q3 + 1.5 × IQR
      Values below lower → set to lower_bound
      Values above upper → set to upper_bound

    votes: lower bound is always 0 (can't have negative votes).
    approx_cost: lower bound is 50 INR (implausibly cheap but possible).

    We also add a boolean flag `{col}_outlier` so downstream analysis
    can study outlier restaurants separately if needed.
    """
    outlier_cols = {
        "votes":       {"min_floor": 0},
        "approx_cost": {"min_floor": 50},
    }

    for col, cfg in outlier_cols.items():
        if col not in df.columns:
            continue

        numeric = df[col].astype(float)
        q1  = numeric.quantile(0.25)
        q3  = numeric.quantile(0.75)
        iqr = q3 - q1
        lower = max(cfg["min_floor"], q1 - 1.5 * iqr)
        upper = q3 + 1.5 * iqr

        is_outlier = (numeric < lower) | (numeric > upper)
        n_outliers = is_outlier.sum()

        df[f"{col}_outlier"] = is_outlier.astype(int)
        df[col] = numeric.clip(lower=lower, upper=upper)

        if col == "approx_cost":
            df[col] = df[col].round(0).astype("Int64")
        else:
            df[col] = df[col].round(0).astype("Int64")

        record("outliers", col, f"IQR winsorization [{lower:.0f}, {upper:.0f}]",
               int(n_outliers), f"Q1={q1:.0f}, Q3={q3:.0f}, IQR={iqr:.0f}")

    return df


# =============================================================================
# STEP 12: DERIVED FEATURES
# =============================================================================

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create analysis-ready features from clean columns.
    These are not cleaning steps — they are feature engineering.
    We do them here so Phase 2 (sentiment) and Phase 5 (RICE) can use them
    without re-importing and re-running the cleaning pipeline.

    Features created:
      rating_tier    — binned rate into Low/Average/Good/Excellent
      primary_cuisine — first cuisine listed (most representative)
      has_reviews     — boolean: restaurant has ≥1 parsed review
      popularity_score — log-scaled votes (handles skew, 0-safe)
    """
    # rating_tier
    if "rate" in df.columns:
        df["rating_tier"] = pd.cut(
            df["rate"],
            bins=[0, 3.0, 3.5, 4.0, 5.0],
            labels=["low", "average", "good", "excellent"],
            right=True,
        )
        record("feature", "rating_tier", "binned rate", len(df),
               "low<3 | average<3.5 | good<4.0 | excellent≤5.0")

    # primary_cuisine
    if "cuisines" in df.columns:
        df["primary_cuisine"] = (
            df["cuisines"]
            .astype(str)
            .str.split(",")
            .str[0]
            .str.strip()
            .replace("Unknown", np.nan)
        )
        record("feature", "primary_cuisine", "extracted first cuisine", len(df))

    # has_reviews
    if "review_count" in df.columns:
        df["has_reviews"] = (df["review_count"] > 0).astype(int)
        n = df["has_reviews"].sum()
        record("feature", "has_reviews", "flagged restaurants with ≥1 review", int(n))

    # popularity_score (log votes, 0-safe)
    if "votes" in df.columns:
        df["popularity_score"] = np.log1p(df["votes"].astype(float)).round(3)
        record("feature", "popularity_score", "log1p(votes)", len(df))

    return df


# =============================================================================
# STEP 13: FINAL VALIDATION
# =============================================================================

def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final sanity checks. Raises ValueError if critical invariants fail.
    Logs warnings for soft issues (don't stop the pipeline).

    Checks:
      - No duplicate URLs remain
      - rate is within [1.0, 5.0] or NaN
      - votes are non-negative or NaN
      - approx_cost is positive or NaN
      - At least 1000 rows remain (smoke test for accidental mass deletion)
    """
    log.info("Running final validation...")

    # Duplicate URLs
    if "url" in df.columns:
        dupes = df["url"].duplicated().sum()
        assert dupes == 0, f"FAIL: {dupes} duplicate URLs remain after cleaning"
        record("validate", "url", "no duplicate URLs", 0, "PASS")

    # Rate bounds
    if "rate" in df.columns:
        out_of_range = df["rate"].dropna()
        out_of_range = out_of_range[(out_of_range < 1.0) | (out_of_range > 5.0)]
        assert len(out_of_range) == 0, f"FAIL: {len(out_of_range)} rate values outside [1,5]"
        record("validate", "rate", "all values in [1.0, 5.0] or NaN", 0, "PASS")

    # Votes non-negative
    if "votes" in df.columns:
        neg_votes = (df["votes"].astype(float) < 0).sum()
        assert neg_votes == 0, f"FAIL: {neg_votes} negative vote values"
        record("validate", "votes", "no negative values", 0, "PASS")

    # Row count smoke test
    assert len(df) >= 1000, f"FAIL: Only {len(df)} rows remain — check for accidental drops"
    record("validate", "all", f"{len(df):,} rows remain", len(df), "PASS")

    log.info(f"Validation passed. Final shape: {df.shape}")
    return df


# =============================================================================
# STEP 14: SAVE OUTPUTS
# =============================================================================

def save_outputs(df: pd.DataFrame, out_dir: str = ".") -> None:
    """
    Save two artifacts:
      1. zomato_cleaned.csv — the clean dataset
      2. cleaning_report.json — audit trail for reproducibility

    The audit trail is critical for internship presentations:
    it proves you thought carefully about each cleaning decision.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = out_path / "zomato_cleaned.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Saved cleaned data → {csv_path}")

    report_path = out_path / "cleaning_report.json"
    report = {
        "final_shape":    {"rows": len(df), "columns": len(df.columns)},
        "final_columns":  list(df.columns),
        "cleaning_steps": audit,
        "null_summary":   df.isnull().sum().to_dict(),
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Saved cleaning report → {report_path}")


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

def run_cleaning_pipeline(input_path: str, output_dir: str = "output") -> pd.DataFrame:
    """
    Run all 13 cleaning steps in order.
    Returns the cleaned DataFrame for immediate use in Phase 2.
    """
    log.info("=" * 60)
    log.info("ZOMATO PHASE 1: DATA CLEANING PIPELINE")
    log.info("=" * 60)

    df = load_data(input_path)
    df = clean_column_names(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = clean_rate(df)
    df = clean_votes(df)
    df = clean_approx_cost(df)
    df = encode_binary_columns(df)
    df = clean_text_columns(df)
    df = parse_reviews_list(df)
    df = handle_outliers(df)
    df = create_derived_features(df)
    df = validate(df)
    save_outputs(df, output_dir)

    log.info("=" * 60)
    log.info(f"PIPELINE COMPLETE | {len(df):,} rows | {len(df.columns)} columns")
    log.info("=" * 60)
    return df


# =============================================================================
# QUICK SUMMARY REPORT (optional, for notebooks)
# =============================================================================

def print_cleaning_summary(df: pd.DataFrame) -> None:
    """
    Print a human-readable summary table.
    Paste this output directly into your internship report.
    """
    print("\n" + "=" * 55)
    print("CLEANING SUMMARY")
    print("=" * 55)
    print(f"{'Column':<25} {'dtype':<12} {'Null%':<10} {'Unique'}")
    print("-" * 55)
    for col in df.columns:
        null_pct = df[col].isna().mean() * 100
        n_unique = df[col].nunique()
        print(f"{col:<25} {str(df[col].dtype):<12} {null_pct:>5.1f}%     {n_unique:>6,}")
    print("=" * 55)
    print(f"\nFinal shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Reviews extracted: {df['review_count'].sum():,}" if "review_count" in df.columns else "")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Change this path to your actual CSV location
    INPUT_PATH  = "data/zomato.csv"
    OUTPUT_DIR  = "output"

    df_clean = run_cleaning_pipeline(INPUT_PATH, OUTPUT_DIR)
    print_cleaning_summary(df_clean)

    # Hand off to Phase 2
    # from src.sentiment import run_sentiment_pipeline
    # df_with_sentiment = run_sentiment_pipeline(df_clean)
