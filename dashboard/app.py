"""
=============================================================================
ZOMATO PRODUCT INTELLIGENCE PLATFORM
Phase 6: Streamlit Dashboard
=============================================================================

Pages:
  1. Overview           — platform health KPIs, key metrics
  2. Sentiment Analysis — DistilBERT scores, distributions, trends
  3. Complaint Categories — BERTopic clusters, keyword weights, heatmap
  4. Feature Recommendations — PM pain points, root causes, solutions
  5. RICE Prioritization — interactive RICE scorer + roadmap
  6. Search Reviews     — semantic search across all reviews

Run:
    pip install streamlit plotly pandas numpy
    streamlit run dashboard.py

Data: uses Phase 1-3 outputs in ./output/ directory.
      Falls back to rich synthetic data if files not found,
      so the dashboard is always fully functional for demos.
=============================================================================
"""

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="Zomato Product Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "positive": "#1D9E75",
    "neutral":  "#888780",
    "negative": "#D85A30",
    "purple":   "#534AB7",
    "blue":     "#185FA5",
    "amber":    "#BA7517",
    "teal":     "#0F6E56",
    "coral":    "#D85A30",
    "pink":     "#993556",
    "green":    "#3B6D11",
    "bg":       "#FAFAF8",
    "topics": [
        "#534AB7","#D85A30","#0F6E56","#BA7517",
        "#185FA5","#993556","#3B6D11","#993C1D",
        "#3C3489","#854F0B",
    ],
}

# =============================================================================
# GLOBAL CSS
# =============================================================================

st.markdown("""
<style>
    .main .block-container{padding-top:1.5rem;padding-bottom:2rem}
    [data-testid="stSidebar"]{background:#FAFAF8}
    .metric-card{
        background:#fff;border:0.5px solid #e8e6e0;
        border-radius:12px;padding:1rem 1.2rem;
    }
    .metric-val{font-size:28px;font-weight:500;line-height:1.1}
    .metric-lbl{font-size:12px;color:#888780;margin-top:4px}
    .metric-delta{font-size:11px;margin-top:4px}
    .section-header{
        font-size:13px;font-weight:500;letter-spacing:.06em;
        text-transform:uppercase;color:#888780;margin-bottom:.75rem;
    }
    .rice-card{
        background:#fff;border:0.5px solid #e8e6e0;
        border-radius:10px;padding:.9rem 1.1rem;margin-bottom:8px;
    }
    .tag{
        display:inline-block;font-size:11px;padding:2px 9px;
        border-radius:20px;margin-right:4px;font-weight:500;
    }
    div[data-testid="stHorizontalBlock"] > div{gap:12px}
    .stPlotlyChart{border-radius:10px;overflow:hidden}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LAYER — loads real CSVs or synthesises demo data
# =============================================================================

OUTPUT_DIR = Path("output")

@st.cache_data(show_spinner=False)
def load_data():
    """
    Try loading Phase 1-3 CSV outputs.
    Falls back to synthetic data for demo/offline use.
    """

    # ---- Topic summary -------------------------------------------------------
    topic_path = OUTPUT_DIR / "complaint_topics.csv"
    if topic_path.exists():
        topics_df = pd.read_csv(topic_path)
    else:
        topics_df = pd.DataFrame({
            "topic_id":    [0,1,2,3,4,5,6,7],
            "topic_label": [
                "Delivery & wait time","Food quality","Staff & service",
                "Portion & value","Order accuracy","Hygiene & cleanliness",
                "App & online ordering","Packaging",
            ],
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

    # ---- Restaurant data with sentiment --------------------------------------
    sentiment_path = OUTPUT_DIR / "zomato_sentiment.csv"
    if sentiment_path.exists():
        df = pd.read_csv(sentiment_path)
        for col in ["rate","votes","approx_cost","avg_signed_score",
                    "sentiment_positive_pct","sentiment_negative_pct","sentiment_neutral_pct"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        np.random.seed(42)
        n = 5000
        rest_types  = ["Quick Bites","Casual Dining","Cafe","Fine Dining","Delivery","Bar","Bakery"]
        cities      = ["BTM","Koramangala","Indiranagar","Whitefield","JP Nagar","HSR Layout","Jayanagar"]
        cuisines    = ["North Indian","South Indian","Chinese","Italian","Continental","Biryani","Fast Food"]
        sentiments  = ["positive","negative","neutral"]
        sent_counts = [3200,1400,1400]

        dom_sent = np.random.choice(sentiments, n, p=[s/6000 for s in sent_counts])
        rates    = np.clip(np.random.normal(3.7, 0.6, n), 1, 5).round(1)
        pos_pct  = np.where(dom_sent=="positive",
                            np.random.uniform(55,90,n),
                            np.random.uniform(10,35,n)).round(1)
        neg_pct  = np.where(dom_sent=="negative",
                            np.random.uniform(50,85,n),
                            np.random.uniform(5,25,n)).round(1)
        neu_pct  = (100 - pos_pct - neg_pct).clip(0,100).round(1)

        df = pd.DataFrame({
            "name":                    [f"Restaurant {i}" for i in range(n)],
            "rest_type":               np.random.choice(rest_types, n),
            "listed_city":             np.random.choice(cities, n),
            "primary_cuisine":         np.random.choice(cuisines, n),
            "rate":                    rates,
            "votes":                   np.random.randint(10, 50000, n),
            "approx_cost":             np.random.choice([200,300,400,500,700,1000,1500,2000], n),
            "online_order":            np.random.choice(["Yes","No"], n, p=[0.65,0.35]),
            "book_table":              np.random.choice(["Yes","No"], n, p=[0.30,0.70]),
            "dominant_sentiment":      dom_sent,
            "sentiment_positive_pct":  pos_pct,
            "sentiment_negative_pct":  neg_pct,
            "sentiment_neutral_pct":   neu_pct,
            "avg_signed_score":        np.random.uniform(-0.8, 0.9, n).round(3),
            "review_count":            np.random.randint(0, 300, n),
            "dominant_complaint":      np.random.choice(
                ["Delivery & wait time","Food quality","Staff & service",
                 "Portion & value","Order accuracy","Hygiene & cleanliness",
                 "App & online ordering","Packaging"], n
            ),
            "rating_tier":             pd.cut(
                rates,[0,3,3.5,4,5],labels=["low","average","good","excellent"]
            ),
            "cost_tier":               np.random.choice(
                ["budget","low-mid","mid","premium","luxury"], n,
                p=[0.25,0.30,0.25,0.15,0.05]
            ),
        })

    # ---- Synthetic reviews for search page -----------------------------------
    review_path = OUTPUT_DIR / "review_topic_assignments.csv"
    if review_path.exists():
        reviews_df = pd.read_csv(review_path).head(5000)
    else:
        sample_texts = [
            "Waited over an hour for delivery, food was cold by the time it arrived.",
            "Absolutely amazing biryani! Portions were generous and flavours were spot on.",
            "Rude staff at the counter, completely ignored us for 15 minutes.",
            "The packaging was spilled, entire bag was wet. Unacceptable.",
            "App crashed during payment, got charged twice. Support took 3 days to respond.",
            "Best paneer butter masala I've had in Bangalore. Will definitely reorder.",
            "Very small portion for the price. Not worth ₹450 at all.",
            "Ordered chicken biryani but received veg pulao. Wrong order delivered.",
            "Cockroach found in the food. Completely horrified. Never ordering again.",
            "Delivery was late by 45 minutes and the driver was unapologetic.",
            "The restaurant was clean and cozy. Staff was very polite and attentive.",
            "Refund not processed after 7 days. Customer support is useless.",
            "Loved the ambience! Great place for a date night. Food was decent.",
            "Box was completely crushed. Fries were scattered everywhere inside.",
            "Fresh ingredients, hot food, quick delivery. Exactly what I expected.",
        ]
        topic_labels = [
            "Delivery & wait time","Food quality","Staff & service","Packaging",
            "App & online ordering","Food quality","Portion & value","Order accuracy",
            "Hygiene & cleanliness","Delivery & wait time","Staff & service",
            "App & online ordering","Staff & service","Packaging","Food quality",
        ]
        n_rev = 3000
        idxs = np.random.randint(0, len(sample_texts), n_rev)
        reviews_df = pd.DataFrame({
            "review_text": [sample_texts[i] for i in idxs],
            "topic_label": [topic_labels[i] for i in idxs],
            "sentiment":   np.random.choice(["positive","negative","neutral"], n_rev,
                                            p=[0.53,0.32,0.15]),
        })

    return df, topics_df, reviews_df


# =============================================================================
# RICE DATA (static — from Phase 5)
# =============================================================================

RICE_DATA = [
    {"feature":"Price breakdown upfront",    "topic":"Portion",       "reach":88,"impact":2,"conf":85,"effort":0.5,"sprint":1},
    {"feature":"Service rating split",       "topic":"Staff",         "reach":85,"impact":2,"conf":88,"effort":0.5,"sprint":1},
    {"feature":"Freshness score algo",       "topic":"Food quality",  "reach":88,"impact":3,"conf":90,"effort":1,  "sprint":1},
    {"feature":"Hygiene fast-track alerts",  "topic":"Hygiene",       "reach":45,"impact":3,"conf":88,"effort":0.5,"sprint":1},
    {"feature":"Serving size tag on menu",   "topic":"Portion",       "reach":80,"impact":2,"conf":82,"effort":1,  "sprint":1},
    {"feature":"Delivery buffer toggle",     "topic":"Delivery",      "reach":78,"impact":2,"conf":90,"effort":1,  "sprint":1},
    {"feature":"Priority support escalation","topic":"Staff",         "reach":55,"impact":2,"conf":80,"effort":1,  "sprint":1},
    {"feature":"Rider surge signal in UI",   "topic":"Delivery",      "reach":70,"impact":1,"conf":80,"effort":1,  "sprint":1},
    {"feature":"Digital packing checklist",  "topic":"Order accuracy","reach":76,"impact":3,"conf":85,"effort":2,  "sprint":2},
    {"feature":"Auto-refund missing items",  "topic":"Order accuracy","reach":62,"impact":3,"conf":80,"effort":2,  "sprint":2},
    {"feature":"Dynamic ETA engine",         "topic":"Delivery",      "reach":92,"impact":3,"conf":85,"effort":6,  "sprint":2},
    {"feature":"Thermal packaging mandate",  "topic":"Food quality",  "reach":74,"impact":2,"conf":75,"effort":3,  "sprint":2},
    {"feature":"Refund tracker in app",      "topic":"App",           "reach":70,"impact":2,"conf":85,"effort":1,  "sprint":2},
    {"feature":"Substitution consent flow",  "topic":"Order accuracy","reach":58,"impact":2,"conf":75,"effort":1,  "sprint":2},
    {"feature":"Value score badge",          "topic":"Portion",       "reach":72,"impact":1,"conf":78,"effort":1,  "sprint":2},
    {"feature":"Photo verify on delivery",   "topic":"Food quality",  "reach":65,"impact":2,"conf":70,"effort":2,  "sprint":2},
    {"feature":"Packaging quality rating",   "topic":"Packaging",     "reach":65,"impact":1,"conf":72,"effort":0.5,"sprint":2},
    {"feature":"Staff training badge",       "topic":"Staff",         "reach":60,"impact":1,"conf":65,"effort":2,  "sprint":3},
]

def rice_score(r):
    return round((r["reach"] * r["impact"] * r["conf"] / 100) / r["effort"])

rice_df = pd.DataFrame(RICE_DATA)
rice_df["rice"] = rice_df.apply(rice_score, axis=1)
rice_df = rice_df.sort_values("rice", ascending=False).reset_index(drop=True)
rice_df.index += 1

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

with st.sidebar:
    st.markdown("### 🍽️ Zomato Intelligence")
    st.markdown("<div style='font-size:12px;color:#888780;margin-bottom:1.5rem'>Product Analytics Platform</div>",
                unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Overview","Sentiment Analysis","Complaint Categories",
         "Feature Recommendations","RICE Prioritization","Search Reviews"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    df, topics_df, reviews_df = load_data()

    # Sidebar filters (apply to all pages)
    st.markdown("**Filters**")
    sel_city = st.multiselect(
        "City", sorted(df["listed_city"].dropna().unique()) if "listed_city" in df.columns else [],
        placeholder="All cities",
    )
    sel_type = st.multiselect(
        "Restaurant type", sorted(df["rest_type"].dropna().unique()) if "rest_type" in df.columns else [],
        placeholder="All types",
    )
    sel_online = st.selectbox("Online order", ["All","Yes","No"])

    # Apply filters
    fdf = df.copy()
    if sel_city:
        fdf = fdf[fdf["listed_city"].isin(sel_city)]
    if sel_type:
        fdf = fdf[fdf["rest_type"].isin(sel_type)]
    if sel_online != "All" and "online_order" in fdf.columns:
        fdf = fdf[fdf["online_order"] == sel_online]

    st.markdown("---")
    st.markdown(f"<div style='font-size:11px;color:#888780'>{len(fdf):,} restaurants in view</div>",
                unsafe_allow_html=True)


# =============================================================================
# PAGE 1 — OVERVIEW
# =============================================================================

if page == "Overview":
    st.markdown("## Platform overview")
    st.markdown("<div style='color:#888780;font-size:14px;margin-bottom:1.5rem'>Zomato Bangalore · Product Intelligence Dashboard · Phase 1–5 outputs</div>",
                unsafe_allow_html=True)

    # KPI row
    total    = len(fdf)
    pos_pct  = (fdf["dominant_sentiment"] == "positive").mean() * 100 if "dominant_sentiment" in fdf.columns else 0
    neg_pct  = (fdf["dominant_sentiment"] == "negative").mean() * 100 if "dominant_sentiment" in fdf.columns else 0
    avg_rate = fdf["rate"].mean() if "rate" in fdf.columns else 0
    tot_rev  = int(fdf["review_count"].sum()) if "review_count" in fdf.columns else 0

    k1,k2,k3,k4,k5 = st.columns(5)
    for col, val, lbl, delta, dcol in [
        (k1, f"{total:,}",       "Restaurants",        "",           ""),
        (k2, f"{pos_pct:.1f}%",  "Positive sentiment", "▲ vs last Q","green"),
        (k3, f"{neg_pct:.1f}%",  "Negative sentiment", "▼ target <20%","red"),
        (k4, f"{avg_rate:.2f}★", "Avg rating",         "",           ""),
        (k5, f"{tot_rev:,}",     "Reviews analysed",   "",           ""),
    ]:
        col.markdown(f"""<div class='metric-card'>
            <div class='metric-val'>{val}</div>
            <div class='metric-lbl'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-header'>Sentiment split</div>", unsafe_allow_html=True)
        if "dominant_sentiment" in fdf.columns:
            sc = fdf["dominant_sentiment"].value_counts().reset_index()
            sc.columns = ["sentiment","count"]
            fig = px.pie(sc, names="sentiment", values="count",
                         color="sentiment",
                         color_discrete_map={"positive":PALETTE["positive"],
                                             "negative":PALETTE["negative"],
                                             "neutral": PALETTE["neutral"]},
                         hole=0.6)
            fig.update_traces(textposition="outside", textinfo="percent+label")
            fig.update_layout(showlegend=False, margin=dict(t=10,b=10,l=10,r=10),
                              height=280, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>Top complaint topics</div>", unsafe_allow_html=True)
        fig2 = px.bar(
            topics_df.head(8).sort_values("frequency"),
            x="frequency", y="topic_label", orientation="h",
            color="pct",
            color_continuous_scale=[[0,"#EEEDFE"],[1,"#534AB7"]],
            labels={"frequency":"Reviews","topic_label":""},
        )
        fig2.update_layout(
            showlegend=False, coloraxis_showscale=False,
            margin=dict(t=10,b=10,l=10,r=10), height=280,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Rating distribution + cost tier
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("<div class='section-header'>Rating distribution</div>", unsafe_allow_html=True)
        if "rate" in fdf.columns:
            fig3 = px.histogram(fdf.dropna(subset=["rate"]), x="rate", nbins=30,
                                color_discrete_sequence=[PALETTE["purple"]])
            fig3.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=220,
                               paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",
                               bargap=0.05,
                               xaxis_title="Star rating", yaxis_title="Count")
            st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown("<div class='section-header'>Restaurant type breakdown</div>", unsafe_allow_html=True)
        if "rest_type" in fdf.columns:
            rt = fdf["rest_type"].value_counts().head(7).reset_index()
            rt.columns = ["type","count"]
            fig4 = px.bar(rt, x="count", y="type", orientation="h",
                          color_discrete_sequence=[PALETTE["teal"]])
            fig4.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=220,
                               paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",
                               yaxis_title="", xaxis_title="Count")
            st.plotly_chart(fig4, use_container_width=True)


# =============================================================================
# PAGE 2 — SENTIMENT ANALYSIS
# =============================================================================

elif page == "Sentiment Analysis":
    st.markdown("## Sentiment analysis")
    st.markdown("<div style='color:#888780;font-size:14px;margin-bottom:1.5rem'>DistilBERT · distilbert-base-uncased-finetuned-sst-2-english</div>",
                unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    for col, label, key, color in [
        (c1,"Positive","positive",PALETTE["positive"]),
        (c2,"Neutral", "neutral", PALETTE["neutral"]),
        (c3,"Negative","negative",PALETTE["negative"]),
    ]:
        if "dominant_sentiment" in fdf.columns:
            pct = (fdf["dominant_sentiment"] == label.lower()).mean() * 100
            col.markdown(f"""<div class='metric-card' style='border-top:3px solid {color}'>
                <div class='metric-val' style='color:{color}'>{pct:.1f}%</div>
                <div class='metric-lbl'>{label} restaurants</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-header'>Avg signed score vs Zomato rating</div>", unsafe_allow_html=True)
        if "rate" in fdf.columns and "avg_signed_score" in fdf.columns:
            sub = fdf.dropna(subset=["rate","avg_signed_score","dominant_sentiment"])
            fig = px.scatter(sub.sample(min(2000, len(sub))),
                             x="rate", y="avg_signed_score",
                             color="dominant_sentiment",
                             color_discrete_map={"positive":PALETTE["positive"],
                                                 "negative":PALETTE["negative"],
                                                 "neutral": PALETTE["neutral"]},
                             opacity=0.45, size_max=5,
                             labels={"rate":"Zomato star rating",
                                     "avg_signed_score":"DistilBERT score"})
            fig.update_traces(marker=dict(size=4))
            fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=300,
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              legend=dict(title="",orientation="h",y=-0.15))
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>Sentiment by restaurant type</div>", unsafe_allow_html=True)
        if "rest_type" in fdf.columns and "dominant_sentiment" in fdf.columns:
            top_types = fdf["rest_type"].value_counts().head(7).index
            sub = fdf[fdf["rest_type"].isin(top_types)]
            agg = sub.groupby(["rest_type","dominant_sentiment"]).size().reset_index(name="count")
            agg["pct"] = agg.groupby("rest_type")["count"].transform(lambda x: x/x.sum()*100)
            fig2 = px.bar(agg, x="rest_type", y="pct", color="dominant_sentiment",
                          color_discrete_map={"positive":PALETTE["positive"],
                                              "negative":PALETTE["negative"],
                                              "neutral": PALETTE["neutral"]},
                          barmode="stack",
                          labels={"rest_type":"","pct":"% of restaurants"})
            fig2.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=300,
                               paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",
                               legend=dict(title="",orientation="h",y=-0.2),
                               xaxis_tickangle=-25)
            st.plotly_chart(fig2, use_container_width=True)

    # Sentiment score histogram
    st.markdown("<div class='section-header'>Signed score distribution</div>", unsafe_allow_html=True)
    if "avg_signed_score" in fdf.columns:
        fig3 = px.histogram(
            fdf.dropna(subset=["avg_signed_score","dominant_sentiment"]),
            x="avg_signed_score", color="dominant_sentiment", nbins=50,
            barmode="overlay", opacity=0.7,
            color_discrete_map={"positive":PALETTE["positive"],
                                "negative":PALETTE["negative"],
                                "neutral": PALETTE["neutral"]},
            labels={"avg_signed_score":"DistilBERT avg signed score",
                    "dominant_sentiment":"Sentiment"},
        )
        fig3.add_vline(x=0, line_dash="dash", line_color="#888780", line_width=1)
        fig3.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=240,
                           paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)",
                           legend=dict(title="",orientation="h",y=-0.2))
        st.plotly_chart(fig3, use_container_width=True)

    # Top negative restaurants
    st.markdown("<div class='section-header'>Restaurants with highest negative sentiment (min 5 reviews)</div>",
                unsafe_allow_html=True)
    if "sentiment_negative_pct" in fdf.columns and "review_count" in fdf.columns:
        neg_top = (fdf[fdf["review_count"] >= 5]
                   .nlargest(12, "sentiment_negative_pct")
                   [["name","sentiment_negative_pct","sentiment_positive_pct","rest_type","rate"]]
                   .reset_index(drop=True))
        neg_top.index += 1
        neg_top.columns = ["Restaurant","Negative %","Positive %","Type","Rating"]
        st.dataframe(neg_top, use_container_width=True)


# =============================================================================
# PAGE 3 — COMPLAINT CATEGORIES
# =============================================================================

elif page == "Complaint Categories":
    st.markdown("## Complaint categories")
    st.markdown("<div style='color:#888780;font-size:14px;margin-bottom:1.5rem'>BERTopic · all-MiniLM-L6-v2 · HDBSCAN clustering</div>",
                unsafe_allow_html=True)

    # Topic frequency bar
    st.markdown("<div class='section-header'>Topic frequency — negative reviews</div>", unsafe_allow_html=True)
    colors = PALETTE["topics"][:len(topics_df)]
    fig = go.Figure(go.Bar(
        x=topics_df["frequency"],
        y=topics_df["topic_label"],
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1f}%" for p in topics_df["pct"]],
        textposition="outside",
    ))
    fig.update_layout(
        margin=dict(t=10,b=10,l=10,r=90), height=320,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Reviews", yaxis_title="",
        yaxis=dict(tickfont=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-header'>Topic distribution by city</div>", unsafe_allow_html=True)
        if "listed_city" in fdf.columns and "dominant_complaint" in fdf.columns:
            top_cities = fdf["listed_city"].value_counts().head(6).index
            sub = fdf[fdf["listed_city"].isin(top_cities)]
            agg = sub.groupby(["listed_city","dominant_complaint"]).size().reset_index(name="count")
            agg["pct"] = agg.groupby("listed_city")["count"].transform(lambda x: x/x.sum()*100).round(1)
            fig2 = px.bar(agg, x="pct", y="listed_city", color="dominant_complaint",
                          orientation="h", barmode="stack",
                          color_discrete_sequence=PALETTE["topics"],
                          labels={"listed_city":"","pct":"% of complaints"})
            fig2.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=300,
                               paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",
                               legend=dict(title="",font=dict(size=9),
                                          orientation="v",x=1.01))
            st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>Complaint mix across rating tiers</div>", unsafe_allow_html=True)
        if "rating_tier" in fdf.columns and "dominant_complaint" in fdf.columns:
            sub = fdf.dropna(subset=["rating_tier","dominant_complaint"])
            agg = sub.groupby(["rating_tier","dominant_complaint"]).size().reset_index(name="count")
            agg["pct"] = agg.groupby("rating_tier")["count"].transform(lambda x: x/x.sum()*100).round(1)
            fig3 = px.bar(agg, x="rating_tier", y="pct", color="dominant_complaint",
                          barmode="stack",
                          color_discrete_sequence=PALETTE["topics"],
                          labels={"rating_tier":"Rating tier","pct":"% of complaints"})
            fig3.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=300,
                               paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",
                               legend=dict(title="",font=dict(size=9),x=1.01))
            st.plotly_chart(fig3, use_container_width=True)

    # Topic deep-dive expanders
    st.markdown("<div class='section-header'>Topic deep-dive</div>", unsafe_allow_html=True)
    for _, row in topics_df.iterrows():
        kws = row["keywords"].split(",") if isinstance(row["keywords"], str) else []
        with st.expander(f"{row['topic_label']}  ·  {int(row['frequency']):,} reviews  ({row['pct']:.1f}%)"):
            c1, c2 = st.columns([2,1])
            with c1:
                st.markdown("**Top keywords**")
                kw_html = "".join([
                    f"<span style='background:#EEEDFE;color:#3C3489;padding:3px 9px;"
                    f"border-radius:20px;font-size:11px;margin:2px;display:inline-block'>{k.strip()}</span>"
                    for k in kws[:8]
                ])
                st.markdown(kw_html, unsafe_allow_html=True)
            with c2:
                st.metric("Frequency", f"{int(row['frequency']):,}")
                st.metric("% of complaints", f"{row['pct']:.1f}%")


# =============================================================================
# PAGE 4 — FEATURE RECOMMENDATIONS
# =============================================================================

elif page == "Feature Recommendations":
    st.markdown("## Feature recommendations")
    st.markdown("<div style='color:#888780;font-size:14px;margin-bottom:1.5rem'>PM analysis — pain points · root causes · product solutions · expected impact</div>",
                unsafe_allow_html=True)

    PM_DATA = [
        {
            "topic":"Delivery & wait time","color":"#534AB7","freq":2840,"pct":22.1,
            "pain":["Unpredictable ETAs erode platform trust beyond individual orders",
                    "No real-time tracking granularity between restaurant, rider, and door"],
            "root":["Surge routing assigns mid-delivery riders to new orders silently",
                    "ETA uses static averages, ignores live kitchen load"],
            "solutions":[
                ("Dynamic ETA engine","Feed live rider GPS + kitchen throughput + traffic into real-time model","ML · 6 months","#534AB7"),
                ("Delivery buffer toggle","Let users opt into conservative ETA with auto-credit on late delivery","Product · 4 weeks","#0F6E56"),
                ("Rider surge signal","Show live rider availability in ordering UI","UX · 3 weeks","#BA7517"),
            ],
            "impact":{"Complaint volume":"↓ 31%","Delivery NPS":"↑ 4.2pt","Repeat order rate":"↑ 8%"},
        },
        {
            "topic":"Food quality","color":"#D85A30","freq":2210,"pct":17.2,
            "pain":["Temperature drops during long delivery windows blamed on Zomato",
                    "Order accuracy and freshness failures compound into 1-star reviews"],
            "root":["No mandatory insulated-packaging requirement at onboarding",
                    "Quality ratings lag actual quality — 50 old 4-stars mask today's 2-star service"],
            "solutions":[
                ("Freshness score algo","Weight recent reviews (30 days) 3× more in rating formula","Data · 3 weeks","#534AB7"),
                ("Thermal packaging mandate","Require insulated packaging for hot food above ₹300","Ops · 2 months","#D85A30"),
                ("Photo verify at pickup","Prompt riders to photograph sealed packaging before pickup","App · 6 weeks","#0F6E56"),
            ],
            "impact":{"Food complaint rate":"↓ 24%","Avg restaurant rating":"↑ 0.3★","Reorder likelihood":"↑ 12%"},
        },
        {
            "topic":"Staff & service","color":"#0F6E56","freq":1760,"pct":13.7,
            "pain":["Rider-restaurant friction transferred to users as delays and wrong handoffs",
                    "Support templated replies worsen the original incident"],
            "root":["No NPS signal specifically for staff interaction quality",
                    "First response SLA is 24h — users expect real-time resolution"],
            "solutions":[
                ("Service rating split","Separate food quality and service into two post-order rating prompts","Product · 2 weeks","#534AB7"),
                ("Priority support escalation","Auto-escalate to human agent if same user files 2+ complaints in 14 days","Support · 4 weeks","#185FA5"),
                ("Staff training badge","Service certified badge for restaurants completing hospitality micro-course","Growth · 6 weeks","#BA7517"),
            ],
            "impact":{"Service complaints":"↓ 19%","Support CSAT":"↑ 6pt","Dine-in bookings":"↑ 3%"},
        },
        {
            "topic":"Hygiene & cleanliness","color":"#993556","freq":980,"pct":7.6,
            "pain":["Single viral hygiene incident affects platform-level brand perception",
                    "No pre-emptive hygiene signal available to users pre-order"],
            "root":["FSSAI inspection data not integrated into Zomato listings",
                    "Hygiene complaints sit in queue until rating drops — no fast escalation"],
            "solutions":[
                ("FSSAI hygiene score","Partner with FSSAI to ingest inspection scores, display hygiene verified badge","Partnership · 3 months","#534AB7"),
                ("Hygiene complaint fast-track","Keywords (cockroach, mold, rat) trigger ops alert within 2h not 48h","Ops · 2 weeks","#D85A30"),
                ("Temporary listing suspension","3 hygiene complaints in 30 days → auto-suspend pending resolution","Policy · 3 weeks","#993556"),
            ],
            "impact":{"Hygiene escalations":"↓ 62%","Safety trust NPS":"↑ 15pt","Viral incidents target":"0"},
        },
    ]

    topic_filter = st.selectbox("Filter by topic", ["All"] + [d["topic"] for d in PM_DATA])

    for d in PM_DATA:
        if topic_filter != "All" and d["topic"] != topic_filter:
            continue
        with st.expander(f"{d['topic']}  ·  {d['freq']:,} reviews  ({d['pct']:.1f}%)  ", expanded=True):
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**User pain points**")
                for p in d["pain"]:
                    st.markdown(f"<div style='background:var(--background-color);border-left:3px solid {d['color']};padding:6px 10px;margin-bottom:6px;font-size:13px;border-radius:0 6px 6px 0'>{p}</div>",
                                unsafe_allow_html=True)
            with c2:
                st.markdown("**Root causes**")
                for r in d["root"]:
                    st.markdown(f"<div style='background:var(--background-color);border-left:3px solid #888780;padding:6px 10px;margin-bottom:6px;font-size:13px;border-radius:0 6px 6px 0'>{r}</div>",
                                unsafe_allow_html=True)

            st.markdown("**Product solutions**")
            sc = st.columns(len(d["solutions"]))
            for col, (title, desc, tag, tc) in zip(sc, d["solutions"]):
                col.markdown(
                    f"<div style='border:0.5px solid #e8e6e0;border-radius:10px;padding:10px 12px;height:100%'>"
                    f"<div style='font-size:12px;font-weight:500;margin-bottom:5px'>{title}</div>"
                    f"<div style='font-size:11.5px;color:#888780;margin-bottom:8px'>{desc}</div>"
                    f"<span style='background:{tc}22;color:{tc};font-size:10px;padding:2px 8px;border-radius:20px;font-weight:500'>{tag}</span>"
                    f"</div>", unsafe_allow_html=True
                )

            st.markdown("**Expected impact**")
            imp_cols = st.columns(len(d["impact"]))
            for col, (k, v) in zip(imp_cols, d["impact"].items()):
                col.metric(k, v)


# =============================================================================
# PAGE 5 — RICE PRIORITIZATION
# =============================================================================

elif page == "RICE Prioritization":
    st.markdown("## RICE prioritization")
    st.markdown("<div style='color:#888780;font-size:14px;margin-bottom:1.5rem'>Reach × Impact × Confidence ÷ Effort · 18 features scored</div>",
                unsafe_allow_html=True)

    # Interactive RICE sliders for custom scoring
    with st.expander("Score a custom feature", expanded=False):
        cf1, cf2 = st.columns(2)
        with cf1:
            c_name   = st.text_input("Feature name", placeholder="e.g. Dark mode for app")
            c_reach  = st.slider("Reach (0–100, users per quarter)", 0, 100, 50)
            c_impact = st.select_slider("Impact", options=[0.25,0.5,1,2,3],
                                        format_func=lambda x: {0.25:"0.25 minimal",0.5:"0.5 low",
                                                                1:"1 medium",2:"2 high",3:"3 massive"}[x])
        with cf2:
            c_conf   = st.slider("Confidence (%)", 0, 100, 75)
            c_effort = st.slider("Effort (person-months)", 0.5, 12.0, 2.0, step=0.5)
            c_score  = round((c_reach * c_impact * c_conf / 100) / c_effort)
        st.metric("RICE score", c_score,
                  delta=f"{'Above' if c_score > rice_df['rice'].median() else 'Below'} median ({int(rice_df['rice'].median())})")

    st.markdown("<br>", unsafe_allow_html=True)

    # Summary stats
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Features scored", len(rice_df))
    k2.metric("Sprint 1 features", len(rice_df[rice_df["sprint"]==1]))
    k3.metric("Highest score", rice_df["rice"].max())
    k4.metric("Median score", int(rice_df["rice"].median()))

    st.markdown("<br>", unsafe_allow_html=True)

    # Filter controls
    col1, col2, col3 = st.columns(3)
    sprint_f = col1.selectbox("Sprint", ["All","1","2","3"])
    topic_f  = col2.selectbox("Topic", ["All"] + sorted(rice_df["topic"].unique()))
    sort_f   = col3.selectbox("Sort by", ["RICE score","Reach","Impact","Effort ↑"])

    rdf = rice_df.copy()
    if sprint_f != "All": rdf = rdf[rdf["sprint"] == int(sprint_f)]
    if topic_f  != "All": rdf = rdf[rdf["topic"]  == topic_f]
    sort_map = {"RICE score":"rice","Reach":"reach","Impact":"impact","Effort ↑":"effort"}
    asc = sort_f == "Effort ↑"
    rdf = rdf.sort_values(sort_map[sort_f], ascending=asc).reset_index(drop=True)
    rdf.index += 1

    # Waterfall chart
    st.markdown("<div class='section-header'>RICE score — ranked features</div>", unsafe_allow_html=True)
    sprint_color_map = {1:PALETTE["purple"], 2:PALETTE["amber"], 3:PALETTE["neutral"]}
    bar_colors = [sprint_color_map.get(s, "#888780") for s in rdf["sprint"]]

    fig = go.Figure(go.Bar(
        x=rdf["feature"], y=rdf["rice"],
        marker_color=bar_colors,
        text=rdf["rice"],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig.update_layout(
        margin=dict(t=20,b=120,l=10,r=10), height=340,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_tickangle=-40, xaxis=dict(tickfont=dict(size=9)),
        yaxis_title="RICE score",
        shapes=[dict(type="line", y0=rice_df["rice"].median(), y1=rice_df["rice"].median(),
                     x0=-0.5, x1=len(rdf)-0.5,
                     line=dict(color="#888780", width=1, dash="dash"))],
    )
    st.plotly_chart(fig, use_container_width=True)

    # Full table
    st.markdown("<div class='section-header'>Full RICE table</div>", unsafe_allow_html=True)
    display_df = rdf[["feature","topic","reach","impact","conf","effort","rice","sprint"]].copy()
    display_df.columns = ["Feature","Topic","Reach","Impact","Confidence %","Effort (pm)","RICE","Sprint"]

    def color_sprint(val):
        colors = {1:"background-color:#EEEDFE;color:#3C3489",
                  2:"background-color:#FAEEDA;color:#633806",
                  3:"background-color:#F1EFE8;color:#5F5E5A"}
        return colors.get(val, "")

    styled = display_df.style.map(color_sprint, subset=["Sprint"]) \
                             .bar(subset=["RICE"], color=PALETTE["purple"]+"55") \
                             .format({"Confidence %": "{}%", "Effort (pm)": "{:.1f}"})
    st.dataframe(styled, use_container_width=True)

    # Scatter: effort vs rice
    st.markdown("<div class='section-header'>Effort vs RICE score (bubble = reach)</div>", unsafe_allow_html=True)
    fig2 = px.scatter(
        rice_df, x="effort", y="rice", size="reach", color="sprint",
        color_discrete_map={1:PALETTE["purple"],2:PALETTE["amber"],3:PALETTE["neutral"]},
        hover_name="feature",
        labels={"effort":"Effort (person-months)","rice":"RICE score","sprint":"Sprint"},
        size_max=30,
    )
    fig2.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=320,
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       legend=dict(title="Sprint",orientation="h",y=-0.15))
    st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# PAGE 6 — SEARCH REVIEWS
# =============================================================================

elif page == "Search Reviews":
    st.markdown("## Search reviews")
    st.markdown("<div style='color:#888780;font-size:14px;margin-bottom:1.5rem'>Keyword search across all parsed review texts</div>",
                unsafe_allow_html=True)

    query = st.text_input("Search reviews", placeholder="e.g. delivery late  ·  cold food  ·  rude staff  ·  refund")

    c1,c2,c3 = st.columns(3)
    topic_filter  = c1.selectbox("Filter topic", ["All"] + sorted(reviews_df["topic_label"].dropna().unique()))
    sent_filter   = c2.selectbox("Filter sentiment", ["All","positive","negative","neutral"])
    max_results   = c3.slider("Max results", 10, 200, 50, step=10)

    results = reviews_df.copy()
    if query:
        mask = results["review_text"].str.contains(query, case=False, na=False)
        results = results[mask]
    if topic_filter != "All":
        results = results[results["topic_label"] == topic_filter]
    if sent_filter != "All":
        results = results[results["sentiment"] == sent_filter]

    results = results.head(max_results)

    # Stats bar
    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Results", len(results))
    if len(results):
        s2.metric("Positive", f"{(results['sentiment']=='positive').mean()*100:.0f}%")
        s3.metric("Negative", f"{(results['sentiment']=='negative').mean()*100:.0f}%")
        s4.metric("Topics found", results["topic_label"].nunique())

    st.markdown("<br>", unsafe_allow_html=True)

    if len(results) == 0:
        st.info("No reviews match your search. Try a different keyword.")
    else:
        sent_colors = {"positive":PALETTE["positive"],"negative":PALETTE["negative"],"neutral":PALETTE["neutral"]}
        for _, row in results.iterrows():
            sc   = sent_colors.get(row.get("sentiment","neutral"), PALETTE["neutral"])
            text = row["review_text"]
            if query:
                text = re.sub(f"({re.escape(query)})", r"<mark style='background:#FAEEDA;color:#633806'>\1</mark>",
                              text, flags=re.IGNORECASE)
            topic = row.get("topic_label","")
            sent  = row.get("sentiment","")
            st.markdown(
                f"<div style='border:0.5px solid #e8e6e0;border-radius:10px;padding:10px 14px;margin-bottom:8px'>"
                f"<div style='display:flex;gap:8px;margin-bottom:6px'>"
                f"<span style='background:{sc}22;color:{sc};font-size:10px;padding:2px 8px;border-radius:20px;font-weight:500'>{sent}</span>"
                f"<span style='background:#F1EFE8;color:#5F5E5A;font-size:10px;padding:2px 8px;border-radius:20px'>{topic}</span>"
                f"</div>"
                f"<div style='font-size:13px;line-height:1.6'>{text}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Mini chart of results
        if len(results) >= 5:
            st.markdown("<div class='section-header'>Result topic breakdown</div>", unsafe_allow_html=True)
            tc = results["topic_label"].value_counts().reset_index()
            tc.columns = ["topic","count"]
            fig = px.bar(tc, x="count", y="topic", orientation="h",
                         color_discrete_sequence=[PALETTE["purple"]])
            fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=200,
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              yaxis_title="", xaxis_title="Reviews")
            st.plotly_chart(fig, use_container_width=True)
