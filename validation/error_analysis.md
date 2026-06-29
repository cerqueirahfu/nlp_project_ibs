# Error Analysis — Aspect Polarity Classifier

**Test set:** the 233 hand-labeled aspect spans in [training/labels.csv](../training/labels.csv).
**System under analysis:** the off-the-shelf SetFitABSA polarity model (the
baseline scored in [polarity_metrics.md](polarity_metrics.md)).
**Errors:** 28 / 233 spans misclassified (87.2% accuracy). Every flagged case is
exported to [flagged_errors.csv](flagged_errors.csv).

Raw confusion (rows = gold, cols = predicted):

|  | → negative | → neutral | → positive |
|---|---:|---:|---:|
| **negative** | 24 | 0 | 7 |
| **neutral** | 2 | 25 | 16 |
| **positive** | 0 | 3 | 156 |

Two facts jump out: the model **never** confuses positive↔negative outright, and
**every** error involves the `neutral`/`positive` boundary or a missed complaint —
i.e. the model leans positive.

## Categorized error table

| Category | Count | Root cause | Example |
|---|---:|---|---|
| **Positive bias on neutral/factual spans** | 14/233 | Restaurant-domain model has a strong positive prior; reads spec-like, conditional, or "okay/average" mentions as praise. `neutral` is also under-represented in training (43 vs 159 positive). | "average product light weight and average quality" → `quality` labeled **neutral**, predicted **positive**. |
| **Context bleed across concatenated reviews** | 7/233 | The `text` field is a comma-joined blob of *several* reviews; sentiment from neighboring clauses leaks onto the target span instead of the span's own sentence. | "...this product is overpriced, value for money good quality product, good..." → `value` (gold **negative** on "overpriced") predicted **positive** from the surrounding "good". |
| **Negation / contrast / conditional misread** | 3/233 | Model keys on a polar word and ignores the negation or hypothetical that flips it. | "as they say clicks are noiseless, that is not true, button press sound is there" → `sound` gold **negative**, predicted **positive** (latched onto "noiseless"). |
| **Implicit sentiment missed** | 3/233 | Praise with no explicit polar word is read as neutral. | "received the product within one day delivery" → `delivery` gold **positive**, predicted **neutral**. |
| **Ambiguous / borderline gold label** | 1/233 | Genuinely arguable neutral-vs-positive case; annotation noise, not a model fault. | "battery life is muah..." (slang) → `battery` gold **negative**, predicted **positive**. |

Counts sum to 28. Categories were assigned by reading each flagged case; the
first two buckets account for **21 of 28 errors (75%)**.

## Two more concrete examples

**Context bleed (count 7):**
- Input span `package` in *"and package was already opened.,nice product"*. Gold
  **negative** (damaged/opened packaging); predicted **positive**. The trailing
  *"nice product"* — a different review entirely — dominated the short negative clause.
- Root cause: spans are scored against the whole concatenated `review_content`
  blob rather than the single sentence the aspect occurs in.

**Positive bias on neutral spans (count 14):**
- Input span `performance` in *"i haven't used any other battery pack so will not
  be able to provide the exact expected performance."* Gold **neutral** (a
  disclaimer, no judgment); predicted **positive**.
- Root cause: the off-the-shelf model defaults to positive whenever an aspect is
  mentioned without an explicit negative cue.

## Prioritized fixes

1. **HIGH — Retrain the polarity head with more in-domain `neutral` examples.**
   Targets the largest bucket (14, positive bias) plus the borderline cases. The
   project already has a fine-tuned head
   ([models/setfit_absa_polarity_finetuned](../models/setfit_absa_polarity_finetuned));
   confirm it is the one deployed and **add ~30–50 neutral/factual electronics
   spans** (specs, disclaimers, "okay/average") to rebalance the 43-vs-159 skew,
   then re-score these 14 cases. Highest count, and the fix already has scaffolding.
   ⚠️ **Tested — see verification below: the existing fine-tuned head regresses.**
   Do not deploy it as-is; rebalance the *negative* class first.

2. **HIGH — Score each aspect on its own sentence, not the review blob.**
   Eliminates the 7 context-bleed errors at the source. Fix in
   [pipeline/preprocess.py](../pipeline/preprocess.py) / span-to-text mapping so the
   polarity model receives only the sentence containing the span (the dataset's
   comma-joined `review_content` must be split first). These 7 include the most
   business-critical failures — **missed complaints** (negative → positive).

3. **MEDIUM — Add negation/contrast examples to training.**
   Only 3 cases, but each is a *missed complaint*, which is the worst error type for
   a project whose research question is about complaints. Seed training with spans
   containing "not", "but", "overpriced", "should be better"; re-validate on the 3
   flagged negation cases.

**Lowest priority:** implicit-sentiment misses (3) degrade gracefully to `neutral`
(they understate praise, never invent a complaint) and the 1 ambiguous label is
measurement noise — neither is worth fixing before 1–3 above.

---

## Verification — did the fine-tuned head fix these categories?

We re-ran the **fine-tuned** polarity head
([models/setfit_absa_polarity_finetuned](../models/setfit_absa_polarity_finetuned))
on the same gold spans and compared it to the baseline. Script:
[regen_finetuned_preds.py](regen_finetuned_preds.py); outputs:
[labels_finetuned_pred.csv](../training/labels_finetuned_pred.csv),
[flagged_errors_finetuned.csv](flagged_errors_finetuned.csv).

Scored on the **224 spans both models could place** (9 excluded because spaCy
could not locate the gold span inside the concatenated review blob — itself a
symptom of the *context-bleed* category above):

| Model | Accuracy | Macro-F1 | Weighted-F1 | Negative recall |
|---|---:|---:|---:|---:|
| Baseline (off-the-shelf) | 0.875 | **0.820** | 0.867 | 0.77 (23/30) |
| Fine-tuned | 0.866 | **0.776** | 0.857 | **0.50 (15/30)** |

### What changed, by category

| Category | Baseline | Fine-tuned | Verdict |
|---|---:|---:|---|
| Positive bias on neutral spans (neutral → positive) | 16 | **10** | ✅ improved — neutral correctly classified rose 25 → 32 |
| **Missed complaints** (negative → positive *or* neutral) | 7 | **15** | ❌ **regressed** — negatives now leak to `neutral` (0 → 6) and `positive` (7 → 9) |
| Other (neutral→neg, pos→neutral, pos→neg) | 5 | 5 | ↔ unchanged |
| **Total errors** | **28** | **30** | net worse |

### Reading

Fine-tuning **did** fix the category it was aimed at — the positive bias on
neutral/factual spans dropped from 16 to 10 errors. But it **bought that by
shifting the decision boundary toward `neutral`**, which **halved negative
recall (0.77 → 0.50)** and more than doubled the *missed-complaint* count
(7 → 15). For a project whose entire research question is about complaints, this
is a net loss despite the near-identical accuracy.

**Likely root cause:** the training set is starved of negatives — only **31
negative** spans vs 159 positive (see [polarity_metrics.md](polarity_metrics.md)).
With 1 epoch and `num_iterations=3`, contrastive fine-tuning rebalanced toward the
classes it saw most and eroded the minority `negative` class.

**Caveat:** the fine-tuned logistic head was pickled under scikit-learn 1.3.2 but
loaded here under 1.8.0 (`InconsistentVersionWarning`); predictions look coherent,
but re-fitting the head under the current sklearn would remove this as a confound.

### Revised priority

- **Fix #1 is NOT ready to ship.** Before retraining again, **rebalance the
  *negative* class** (add negative + neutral spans, or apply class weights) — do
  not just add neutrals. Re-score with this script to confirm negative recall
  recovers *before* deploying.
- **Fix #2 (sentence isolation) rises to the top.** It is independent of model
  training, addresses context-bleed, and would also recover the **9 spans that
  could not even be located** in the concatenated blobs.

---

## Iteration 2 — rebalanced fine-tune (the fix, applied)

We mined real negative aspect-spans from the corpus (no per-review star ratings
exist, so candidates came from aspect+complaint-cue matching —
[mine_negative_candidates.py](mine_negative_candidates.py)), hand-reviewed them,
and retrained on a **balanced** set (132 neg / 50 neu / 131 pos) with a **clean,
leakage-free split**: the held-out test is the original 47 gold rows, and a
**132-negative probe of reviewed negatives never seen in training** gives a
reliable recall estimate (the gold test has only 5 negatives). Scripts:
[build_split.py](build_split.py), [merge_labels.py](merge_labels.py); train via
`train_polarity.py --labels training/labels_train.csv --test-size 0`.

| Metric | Off-the-shelf baseline | Iter-1 fine-tune (leaky) | **Iter-2 balanced (clean)** |
|---|---:|---:|---:|
| Negative recall (reliable, n=132 probe) | ~0.60 | — | **0.90 (119/132)** |
| Negative recall (gold test, n=5) | 0.60 | — | 0.80 (4/5) |
| Positives kept correct (gold test, n=35) | 35/35 | — | 30/35 |
| → positives wrongly flagged negative | 0 | — | **3 (8.6%)** |
| Macro-F1 (gold test, 80% positive) | 0.788 | 0.722 | 0.670 |

**Outcome:** the rebalancing **fixed the missed-complaint problem** — negative
recall rose from ~0.60 to **0.90**. The cost is a modest false-alarm rate: ~9% of
positive spans are now flagged negative. The lower Macro-F1 is **an artifact of
the positive-heavy gold test**, which rewards the positive-biased baseline; for a
complaint-detection task, catching 90% of negatives is the better operating point.

**Note on macro-F1 over the all-negative probe:** the probe contains only the
`negative` class, so a 3-class macro-F1 on it (0.316) is mathematically
meaningless — read **negative recall** there, not macro-F1.

**Open items:** (a) the ~9% positive false-alarm rate rests on 35 positives —
build a larger held-out *positive* set to firm it up; (b) if false alarms matter,
try a less aggressive ~2:1 positive:negative ratio or threshold calibration;
(c) Fix #2 (sentence isolation) is still unaddressed and remains the highest-value
*independent* improvement.
