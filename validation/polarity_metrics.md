# Performance Metrics — Aspect Polarity Classifier

**Task type:** classification — aspect-based sentiment analysis (ABSA). Given a
review and an aspect span (e.g. *"battery capacity"*), predict the polarity in
{`positive`, `negative`, `neutral`}. This is the core NLP step behind every
number on the dashboard.

**Test set:** 47 hand-labeled aspect spans held out with `seed=42`, `test_size=0.2`
— the exact split [training/train_polarity.py](../training/train_polarity.py)
reserves, so these examples were **never used to develop the fine-tuned model**.
47 falls inside the required 20–50 range. Labels were corrected by hand in
[training/labels.csv](../training/labels.csv); the scored predictions come from
the off-the-shelf SetFitABSA model, which never saw any of these labels (verified:
train-portion agreement 0.87 is *not* higher than held-out 0.91 → no leakage),
making this a fair **baseline** of "what good looks like" before fine-tuning.

Reproduce with: `nlpenv/Scripts/python.exe validation/evaluate_polarity.py`

## Metrics table (held-out set, n = 47)

| Metric | Score | Justification |
|---|---:|---|
| Accuracy | **0.915** | Overall share of spans whose polarity is predicted correctly — the intuitive headline number. |
| Macro-F1 | **0.789** | Averages F1 equally across all 3 classes, so the rare `negative`/`neutral` spans count as much as `positive`; the right metric when 81% of the data is positive and accuracy alone would hide minority-class failures. |
| Weighted-F1 | **0.905** | F1 weighted by class frequency — reflects expected quality on the real, positive-skewed review mix the dashboard actually aggregates. |

### Per-class breakdown (held-out)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 1.000 | 0.600 | 0.750 | 5 |
| neutral  | 1.000 | 0.500 | 0.667 | 4 |
| positive | 0.905 | 1.000 | 0.950 | 38 |

## Sanity check — full labeled set (n = 233)

The held-out slice has only 4–5 minority examples, so the same metrics on all 233
labeled spans give a more stable estimate (still zero training leakage for the
baseline model):

| Metric | Score |
|---|---:|
| Accuracy | 0.880 |
| Macro-F1 | 0.823 |
| Weighted-F1 | 0.872 |

Per-class (full set): negative F1 **0.842**, neutral F1 **0.704**, positive F1 **0.923**.

## Why these metrics?

- **Accuracy** is the standard headline for a balanced-enough classifier and is
  easy to communicate, but on its own it is misleading here because the dataset
  is 81% positive — a "always positive" baseline would already score ~0.68.
- **Macro-F1** is the primary metric: the research question is about *complaints*,
  so correctly catching `negative` spans matters as much as `positive` ones, and
  macro-averaging refuses to let the dominant positive class paper over weak
  minority-class recall.
- **Weighted-F1** complements it by reporting expected quality on the true class
  distribution the dashboard aggregates over.

## Reading of the results

- The classifier is **precise on minority classes** (negative/neutral precision
  ≈ 0.9–1.0): when it flags a complaint, it is almost always a real complaint, so
  "Top Complaints" on the dashboard is trustworthy.
- The main weakness is **neutral recall** (0.50–0.58): borderline neutral spans
  are most often mislabeled `positive` (see confusion matrix), slightly inflating
  positive sentiment. This is the obvious target for future fine-tuning.
- Per-aspect accuracy ranges from 0.94 (`quality`) down to 0.67 (`sound`, n=9) and
  0.71 (`delivery`, n=7); the low-confidence aspects are simply under-sampled and
  need more labeled spans before per-aspect claims are reliable.

## Limitations / next steps

- These are **baseline** (pre-fine-tuning) numbers. The fine-tuned head has since
  been scored on the same gold spans — see the verification section in
  [error_analysis.md](error_analysis.md). Surprisingly it does **not** beat this
  baseline (Macro-F1 0.776 vs 0.820 on the common 224-span subset): it reduces the
  positive bias but halves negative recall. Regenerate with
  [regen_finetuned_preds.py](regen_finetuned_preds.py).
- Aspect **detection** (which of the 7 aspects appear) is not scored here — we
  have gold polarity per span but no gold "expected aspect set" per review.
  Labeling ~30 reviews for aspect presence would let us report Precision/Recall
  on aspect extraction too.
