# Baseline Comparison — Aspect Polarity Classifier

**Task:** aspect-based sentiment classification (`positive` / `negative` / `neutral`).
**Test set:** identical to [polarity_metrics.md](polarity_metrics.md) — the 47-span
held-out split (`seed=42`, `test_size=0.2`), with the full 233-span labeled set as a
more stable estimate. **Primary metric:** Macro-F1 (higher is better) — averages all
three classes equally so the positive-skewed data can't hide minority-class failures.

Reproduce with: `nlpenv/Scripts/python.exe validation/compare_baselines.py`

## The three systems

| | System | What it does | Key NLP component |
|---|---|---|---|
| A | **Naive** | Always predicts the majority class (`positive`) | none — no text, no ML |
| B | **Ablation** | VADER lexicon sentiment on the span's sentence | fixed word list, **no learned semantic model** |
| C | **Full** | SetFitABSA (sentence-transformer + classifier) | the actual deployed model |

The ablation keeps everything except the one thing being tested — the learned
SetFitABSA semantic model — and swaps in a rule-based lexicon, so the gap B→C is
exactly the value added by the transformer-based ABSA model.

## Side-by-side comparison

### Held-out test set (n = 47)

| System | Accuracy | **Macro-F1** | Weighted-F1 |
|---|---:|---:|---:|
| A: Naive (majority=positive) | 0.809 | 0.298 | 0.723 |
| B: Ablation (VADER lexicon) | 0.809 | 0.575 | 0.800 |
| **C: Full (SetFitABSA)** | **0.915** | **0.789** | **0.905** |

### Full labeled set (n = 233)

| System | Accuracy | **Macro-F1** | Weighted-F1 |
|---|---:|---:|---:|
| A: Naive (majority=positive) | 0.682 | 0.270 | 0.554 |
| B: Ablation (VADER lexicon) | 0.717 | 0.537 | 0.700 |
| **C: Full (SetFitABSA)** | **0.880** | **0.823** | **0.872** |

## Insight

The learned SetFitABSA model adds **+0.21 Macro-F1 over the VADER ablation on the
held-out set** (0.575 → 0.789, **+37% relative**) and **+0.29 on the full set**
(0.537 → 0.823, **+53% relative**) — confirming that the transformer-based aspect
sentiment model, not the surrounding pipeline, is the highest-value component;
the naive majority baseline collapses to ~0.27–0.30 Macro-F1 because it cannot
detect a single complaint.

## Why accuracy alone would have lied

The naive baseline reaches **0.81 accuracy** on the held-out set just by guessing
`positive` every time — yet its Macro-F1 is **0.30** because it never identifies a
negative or neutral span. Since the whole point of Price Pulse is spotting how
complaints shift with discount level, accuracy is a vanity number here and
Macro-F1 is the metric that actually separates the three systems.
