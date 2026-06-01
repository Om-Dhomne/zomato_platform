# 🍽️ Zomato Product Intelligence Platform

> End-to-end NLP pipeline that turns 50,000+ customer reviews into ranked product recommendations — built as a PM + Data Science internship project.

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-DistilBERT-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co)
[![BERTopic](https://img.shields.io/badge/BERTopic-0.16-534AB7)](https://maartengr.github.io/BERTopic)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**[🚀 Live Demo](https://your-username-zomato-intelligence.streamlit.app)** &nbsp;·&nbsp; [Report](docs/report.pdf) &nbsp;·&nbsp; [LinkedIn Post](#)

---

<!-- Replace with actual GIF -->
![Dashboard demo](assets/demo.gif)

---

## Key metrics

| | |
|---|---|
| **50,000+** reviews analysed | **8** complaint categories identified |
| **18** features RICE scored | **6**-page interactive dashboard |

---

## What it does

Raw Zomato review data → cleaned → DistilBERT sentiment scoring → BERTopic complaint clustering → PM pain point analysis → RICE-prioritised product roadmap → deployed Streamlit dashboard.

The output answers: *"What are customers most unhappy about, and what should the product team fix first?"*

---

## Key findings

- **Delivery & wait time** accounts for 22.1% of all negative reviews — the #1 complaint. Root cause: ETA algorithm ignores live kitchen load.
- **Hygiene complaints** represent only 7.6% of volume but score **critical** severity — a single viral incident causes platform-level brand damage.
- **Price breakdown upfront** and **service rating split** scored highest on RICE (299 each) — high reach, low effort, ships in under 2 weeks.

---

## Architecture

```
Raw CSV (Kaggle)
     │
     ▼
Phase 1 · Data cleaning          src/clean.py
     │  handle nulls, parse rate, deduplicate, winsorise outliers
     ▼
Phase 2 · Sentiment analysis     src/sentiment.py
     │  DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
     │  per-review inference → restaurant-level aggregation
     ▼
Phase 3 · Complaint clustering   src/topics.py
     │  Sentence-BERT → UMAP → HDBSCAN → BERTopic
     │  8 complaint categories with c-TF-IDF keywords
     ▼
Phase 4 · PM analysis            dashboard/app.py
     │  pain points · root causes · product solutions · impact
     ▼
Phase 5 · RICE prioritization    dashboard/app.py
     │  18 features × Reach × Impact × Confidence ÷ Effort
     ▼
Phase 6 · Streamlit dashboard    dashboard/app.py
        6 pages · real-time filters · CSV export
```

---

## Tech stack

| Phase | Tools |
|---|---|
| Data cleaning | pandas, numpy, scipy |
| Sentiment analysis | transformers (DistilBERT), PyTorch |
| Complaint clustering | BERTopic, sentence-transformers, UMAP, HDBSCAN |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| CI | GitHub Actions, flake8, pytest |

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/your-username/zomato-intelligence.git
cd zomato-intelligence
pip install -r requirements.txt

# 2. Run with demo data (no Kaggle download needed)
python pipeline.py --demo

# 3. Launch dashboard
streamlit run dashboard/app.py
```

To run on the real dataset:
```bash
# Download from https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants
# Place as data/zomato.csv, then:
python pipeline.py --sample 100000
```

---

## Project structure

```
zomato-intelligence/
├── src/
│   ├── clean.py          # Phase 1: data cleaning (13 steps, audit trail)
│   ├── sentiment.py      # Phase 2: DistilBERT inference + aggregation
│   ├── topics.py         # Phase 3: BERTopic complaint clustering
│   ├── pain_points.py    # Phase 4: PM pain point scoring
│   └── rice.py           # Phase 5: RICE framework scoring
├── dashboard/
│   └── app.py            # Phase 6: 6-page Streamlit dashboard
├── notebooks/
│   ├── 01_eda.ipynb      # Exploratory data analysis
│   ├── 02_sentiment.ipynb
│   └── 03_topics.ipynb
├── assets/
│   ├── demo.gif          # Dashboard demo (embed in README)
│   ├── architecture.png  # Pipeline diagram
│   └── screenshots/      # One screenshot per dashboard page
├── data/
│   └── sample_100.csv    # 100-row sample for instant demo
├── tests/
│   └── test_clean.py     # Unit tests for cleaning pipeline
├── pipeline.py           # One-command full pipeline runner
├── requirements.txt      # Pinned dependencies
└── .streamlit/
    └── config.toml       # Theme + server config
```

---

## Dashboard pages

| Page | What it shows |
|---|---|
| Overview | Platform KPIs, sentiment split, top complaints |
| Sentiment Analysis | DistilBERT scores, score vs rating scatter, top negative restaurants |
| Complaint Categories | BERTopic clusters, keyword weights, heatmap by restaurant type |
| Feature Recommendations | PM pain points, root causes, product solutions, impact estimates |
| RICE Prioritization | Interactive RICE scorer, ranked feature table, effort vs score scatter |
| Search Reviews | Keyword search with sentiment + topic filters, highlighted matches |

---

## Screenshots

<!-- Add screenshots here using: ![Page name](assets/screenshots/page.png) -->

---

## About

Built by **[Your Name]** · B.Tech 2nd year · IIT Madras  
Internship project — 2025 season  
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)

---

## License

MIT — see [LICENSE](LICENSE)
