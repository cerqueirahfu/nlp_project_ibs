# PricePulse — *Precision Analytics*

An NLP prototype that analyzes Amazon electronics reviews to explore **how discount depth relates to aspect-level customer sentiment**.

> **Research question:** *Do customers complain differently when a product is heavily discounted?*

Built as a university project — Furtwangen University, IBS Summer Semester 2026.

---

## What it does

Instead of scoring whole reviews, PricePulse works at the **aspect** level: it detects which product aspects each review sentence mentions (battery, quality, performance, durability, delivery, value, sound), classifies the sentiment of each aspect mention, and aggregates the result by product and by **discount tier** (0–15%, 15–30%, 30–50%, 50%+). The Streamlit dashboard then lets you compare how sentiment shifts across discount levels.

### Dashboard features
- **Cascading category filter** + discount-tier selector
- **3 key insights up front** — top complaint aspect, strongest positive aspect, overall sentiment
- **Sentiment by discount level** and **aspect-based insights** tables/charts
- **Top complaints** and **top discount drivers**
- **AI features** (OpenAI `gpt-4o-mini`): on-demand insight summaries, per-product Pros/Cons, and a dataset chat — all grounded in the precomputed aggregates

---

## Architecture

The system is split into two phases to keep the cloud cost ≈ **$0** while staying publicly demoable.

```
Laptop (Phase 1: heavy NLP, offline)            Cloud (Phase 2: serve, light)
┌──────────────────────────────────┐            ┌───────────────────────────┐
│ amazon.csv                        │            │ EC2 t2.micro (free tier)  │
│   → clean / sentence-split        │            │   Streamlit (app.py)      │
│   → SetFitABSA inference          │  parquet   │   reads parquet, filters, │
│   → aspect map + polarity         │ ─────────► │   builds charts + prompts │
│   → aggregate                     │   (S3)     │        │                  │
│   → results.parquet               │            │        └─► OpenAI API     │
│   → products.parquet              │            │            (gpt-4o-mini)  │
└──────────────────────────────────┘            └───────────────────────────┘
```

- **All model inference runs locally** — the EC2 server never loads SetFitABSA. It only reads the precomputed `results.parquet` / `products.parquet` (from S3 in production) and calls the OpenAI API for text generation.
- The only artifact crossing the laptop→cloud boundary is the two parquet files. Re-running the pipeline = re-uploading them.

See **[EC2_DEPLOYMENT_GUIDE.md](EC2_DEPLOYMENT_GUIDE.md)** for deployment steps and **[CLAUDE.md](CLAUDE.md)** for the full design spec.

---

## Tech stack

| Layer | Tool |
|---|---|
| NLP model | SetFitABSA (`bge-small-en-v1.5` backbone) via Hugging Face |
| Aspect spans | spaCy `en_core_web_sm` |
| Dashboard | Streamlit |
| Data | pandas + pyarrow (Parquet) |
| AI text | OpenAI API (`gpt-4o-mini`) |
| Storage / hosting | AWS S3 + EC2 `t2.micro` (free tier) |

---

## Project structure

```
price-pulse/
├── app.py                       # Streamlit dashboard entry point
├── pages/
│   ├── 01_Product.py            # per-product drill-down + AI Pros/Cons
│   └── 02_AI_Chat.py            # chat grounded in the current selection
├── pipeline/
│   ├── clean.py                 # load + clean Amazon CSV, sentence split
│   ├── preprocess.py            # explode reviews into per-sentence rows
│   ├── absa.py                  # SetFitABSA inference wrapper
│   └── aggregate.py             # aspect keyword map + aggregation
├── scripts/
│   └── build_results.py         # Phase-1 pipeline → results/products.parquet
├── ai/
│   ├── llm_client.py            # OpenAI wrapper (timeout, retries, graceful degrade)
│   └── insights.py              # prompt building (summary / Pros-Cons / chat)
├── utils/helpers.py             # data loaders (local or s3://) + brand CSS
├── training/                    # polarity fine-tuning data + script
├── validation/                  # evaluation harness + results docs
├── models/setfit_absa_polarity_finetuned/   # locally fine-tuned polarity head
├── data/amazon.csv              # pre-downloaded dataset
├── requirements.txt             # full pipeline deps
└── requirements-dashboard.txt   # dashboard-only deps (for EC2)
```

---

## Getting started

**Prerequisites:** Python 3.11.

```bash
# 1. Create / activate a virtual environment, then install deps
pip install -r requirements.txt

# 2. Download the spaCy model used for aspect-span detection
python -m spacy download en_core_web_sm

# 3. (For AI features) set your OpenAI key
export OPENAI_API_KEY="sk-..."        # PowerShell: $env:OPENAI_API_KEY = "sk-..."
```

### 1. Build the data (Phase 1, local)

Runs the full ABSA pipeline over `data/amazon.csv` and writes the two parquet files. Full dataset takes ~20–40 min on CPU.

```bash
# Fast smoke test on a small slice first
python scripts/build_results.py --limit 50

# Full run
python scripts/build_results.py
```

The build is **checkpointed**: if it crashes mid-run it resumes from the last completed chunk (checkpoints in `.absa_checkpoints/`, auto-cleaned on success; keep them with `--keep-checkpoints`).

### 2. Run the dashboard

```bash
streamlit run app.py
```

By default it reads the local `results.parquet` / `products.parquet`. To read from S3 instead, set `RESULTS_S3_URI` / `PRODUCTS_S3_URI` (e.g. `s3://price-pulse/results.parquet`).

---

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — (required for AI) | OpenAI auth; AI features hide gracefully if unset |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model for summaries / chat |
| `OPENAI_TIMEOUT` | `30` | Per-request timeout (seconds) |
| `OPENAI_MAX_RETRIES` | `3` | Retries on transient errors (exponential backoff) |
| `OPENAI_BASE_URL` | — | Optional, for OpenAI-compatible gateways |
| `RESULTS_S3_URI` / `PRODUCTS_S3_URI` | local file | Read parquet from S3 in production |

No secrets live in code. On EC2, AWS credentials come from the instance's IAM role.

---

## NLP details

- **Method:** Aspect-Based Sentiment Analysis (ABSA) with SetFit (few-shot, contrastive). spaCy proposes candidate aspect spans → the aspect model keeps real aspects → the polarity model scores each as positive / negative / neutral → spans are mapped to 7 tracked aspects (keyword lists in [pipeline/aggregate.py](pipeline/aggregate.py)) → aggregated per product × aspect × discount tier.
- **Checkpoints:** off-the-shelf `setfit-absa-bge-small-en-v1.5-restaurants-{aspect,polarity}` (restaurant-domain, used for transfer), with a **polarity head fine-tuned locally on hand-labeled electronics reviews** — auto-loaded from `models/setfit_absa_polarity_finetuned/` if present.

### Validation

The polarity classifier is evaluated on a held-out hand-labeled split plus a dedicated negative probe. Macro-F1 is the primary metric because the data is ~81% positive and accuracy alone hides minority-class failures.

- **Off-the-shelf SetFitABSA baseline** (held-out, n=47): **accuracy 0.915, Macro-F1 0.789** — beating a VADER-lexicon ablation by **+0.21 Macro-F1** and a majority-class baseline (Macro-F1 0.30) by a wide margin. Strong on the natural positive-skewed mix, but under-detects hard negatives.
- **Deployed (rebalanced fine-tuned) head:** trained on mined negative spans to fix missed complaints. On a held-out 132-negative probe it lifts **negative recall to 0.90**, at the cost of ~**9%** of positive spans wrongly flagged. Its lower Macro-F1 on the positive-skewed gold test (0.67) is a distribution artifact — for a complaint-detection task, 0.90 negative recall is the chosen operating point. (An earlier fine-tune that only added neutrals regressed and was abandoned.)

Full two-iteration story and numbers:
- [validation/polarity_metrics.md](validation/polarity_metrics.md) — per-class baseline metrics
- [validation/baseline_comparison.md](validation/baseline_comparison.md) — naive vs ablation vs full
- [validation/error_analysis.md](validation/error_analysis.md) — error categories + both fine-tune iterations
- [validation/runtime_performance.md](validation/runtime_performance.md) — latency & cost (~$0.0002/query, ~3.4 s)

---

## Limitations & non-goals

- **No live scraping** — uses the prepared dataset only.
- **English reviews only.**
- **No user authentication**, web dashboard only (no mobile app).
- The aspect extractor is restaurant-domain (transfer-applied to electronics); aspect *detection* is not yet quantitatively validated.
- The held-out test set is small (47 spans); per-aspect claims need more labeled data.

---

## Author

Developed by [Juliana Cerqueira](https://www.linkedin.com/in/cerqueirajoc/) — Furtwangen University, IBS Summer Semester 2026. Inspired by internship experience as a BIE at Amazon.
