"""
Zomato Product Intelligence Platform
One-command pipeline runner — executes all 6 phases in order.

Usage:
    python pipeline.py                    # full run
    python pipeline.py --phase 1          # single phase
    python pipeline.py --sample 100000    # run on N-row sample
    python pipeline.py --demo             # use synthetic data, no CSV needed
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


def run_phase(name: str, fn, *args, **kwargs):
    log.info(f"{'='*55}")
    log.info(f"  {name}")
    log.info(f"{'='*55}")
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        log.info(f"  Completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        log.error(f"  FAILED: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Zomato Intelligence Pipeline")
    parser.add_argument("--phase",  type=int, default=0,     help="Run only this phase (1-5)")
    parser.add_argument("--sample", type=int, default=100_000, help="Row sample for inference phases")
    parser.add_argument("--demo",   action="store_true",     help="Use synthetic data (no CSV needed)")
    parser.add_argument("--input",  type=str, default="data/zomato.csv")
    parser.add_argument("--output", type=str, default="output")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.demo:
        log.info("Demo mode: generating synthetic data, skipping CSV requirement")
        _run_demo_mode(args.output)
        return

    phases = {
        1: ("Phase 1 · Data cleaning",       _phase1),
        2: ("Phase 2 · Sentiment analysis",  _phase2),
        3: ("Phase 3 · Complaint clustering",_phase3),
        4: ("Phase 4 · Pain point scoring",  _phase4),
        5: ("Phase 5 · RICE prioritization", _phase5),
    }

    to_run = [args.phase] if args.phase else list(phases.keys())

    for pid in to_run:
        name, fn = phases[pid]
        run_phase(name, fn, args)

    log.info("")
    log.info("Pipeline complete.")
    log.info(f"  Outputs:       {args.output}/")
    log.info(f"  Dashboard:     streamlit run dashboard/app.py")
    log.info("")


def _phase1(args):
    from src.clean import run_cleaning_pipeline
    return run_cleaning_pipeline(args.input, args.output)


def _phase2(args):
    from src.sentiment import run_sentiment_pipeline
    cleaned_csv = f"{args.output}/zomato_cleaned.csv"
    return run_sentiment_pipeline(
        cleaned_csv=cleaned_csv,
        output_dir=args.output,
        sample_n=args.sample,
    )


def _phase3(args):
    from src.topics import run_topic_pipeline
    sentiment_csv = f"{args.output}/zomato_sentiment.csv"
    return run_topic_pipeline(sentiment_csv=sentiment_csv, output_dir=args.output)


def _phase4(args):
    log.info("Pain point mapping uses static PM analysis — see dashboard Feature Recommendations page")


def _phase5(args):
    log.info("RICE scores computed dynamically in dashboard — see RICE Prioritization page")


def _run_demo_mode(output_dir: str):
    """Generate synthetic output CSVs so the dashboard works with no real data."""
    import numpy as np
    import pandas as pd

    log.info("Generating synthetic demo data...")
    np.random.seed(42)
    n = 5000

    rest_types = ["Quick Bites","Casual Dining","Cafe","Fine Dining","Delivery","Bar","Bakery"]
    cities     = ["BTM","Koramangala","Indiranagar","Whitefield","JP Nagar","HSR Layout"]
    cuisines   = ["North Indian","South Indian","Chinese","Italian","Biryani","Fast Food"]
    complaints = [
        "Delivery & wait time","Food quality","Staff & service",
        "Portion & value","Order accuracy","Hygiene & cleanliness",
        "App & online ordering","Packaging",
    ]
    dom_sent   = np.random.choice(["positive","negative","neutral"], n, p=[0.53,0.32,0.15])
    rates      = np.clip(np.random.normal(3.7, 0.6, n), 1, 5).round(1)
    pos_pct    = np.where(dom_sent=="positive",
                          np.random.uniform(55,90,n),
                          np.random.uniform(10,35,n)).round(1)
    neg_pct    = np.where(dom_sent=="negative",
                          np.random.uniform(50,85,n),
                          np.random.uniform(5,25,n)).round(1)
    neu_pct    = (100 - pos_pct - neg_pct).clip(0, 100).round(1)

    df = pd.DataFrame({
        "name":                    [f"Restaurant {i}" for i in range(n)],
        "rest_type":               np.random.choice(rest_types, n),
        "listed_city":             np.random.choice(cities, n),
        "primary_cuisine":         np.random.choice(cuisines, n),
        "rate":                    rates,
        "votes":                   np.random.randint(10, 50000, n),
        "approx_cost":             np.random.choice([200,300,400,500,700,1000,1500], n),
        "online_order":            np.random.choice(["Yes","No"], n, p=[0.65,0.35]),
        "book_table":              np.random.choice(["Yes","No"], n, p=[0.30,0.70]),
        "dominant_sentiment":      dom_sent,
        "sentiment_positive_pct":  pos_pct,
        "sentiment_negative_pct":  neg_pct,
        "sentiment_neutral_pct":   neu_pct,
        "avg_signed_score":        np.random.uniform(-0.8, 0.9, n).round(3),
        "review_count":            np.random.randint(0, 300, n),
        "dominant_complaint":      np.random.choice(complaints, n),
        "rating_tier":             pd.cut(rates,[0,3,3.5,4,5],
                                         labels=["low","average","good","excellent"]).astype(str),
        "cost_tier":               np.random.choice(
                                     ["budget","low-mid","mid","premium","luxury"], n,
                                     p=[0.25,0.30,0.25,0.15,0.05]),
    })

    topics_df = pd.DataFrame({
        "topic_id":    range(8),
        "topic_label": complaints,
        "keywords": [
            "delivery,late,wait,slow,delay",
            "cold food,bland,stale,quality,taste",
            "rude staff,service,attitude,waiter,ignored",
            "small portion,expensive,overpriced,value,quantity",
            "wrong order,missing,incorrect,item,mistake",
            "dirty,unhygienic,cockroach,smell,filthy",
            "app crash,refund,cancel,payment,glitch",
            "spill,leak,damaged,packaging,broken",
        ],
        "frequency": [2840,2210,1760,1540,1230,980,820,640],
        "pct":       [22.1,17.2,13.7,11.9, 9.6, 7.6, 6.4, 5.0],
    })

    review_samples = [
        "Waited over an hour, food was cold by the time it arrived.",
        "Amazing biryani! Portions were generous and flavours were spot on.",
        "Rude staff, completely ignored us for 15 minutes.",
        "Packaging was spilled, entire bag was wet.",
        "App crashed during payment, got charged twice.",
        "Best paneer butter masala in Bangalore. Will reorder.",
        "Very small portion for ₹450. Not worth it at all.",
        "Ordered chicken biryani but received veg pulao.",
        "Cockroach found in the food. Never ordering again.",
        "Delivery was late by 45 minutes.",
    ]
    topic_map = [
        "Delivery & wait time","Food quality","Staff & service","Packaging",
        "App & online ordering","Food quality","Portion & value","Order accuracy",
        "Hygiene & cleanliness","Delivery & wait time",
    ]
    idx = np.random.randint(0, len(review_samples), 3000)
    reviews_df = pd.DataFrame({
        "review_text": [review_samples[i] for i in idx],
        "topic_label": [topic_map[i] for i in idx],
        "sentiment":   np.random.choice(["positive","negative","neutral"], 3000, p=[0.53,0.32,0.15]),
    })

    out = Path(output_dir)
    df.to_csv(out / "zomato_sentiment.csv", index=False)
    topics_df.to_csv(out / "complaint_topics.csv", index=False)
    reviews_df.to_csv(out / "review_topic_assignments.csv", index=False)
    log.info(f"Demo data written to {output_dir}/")
    log.info("Run:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
