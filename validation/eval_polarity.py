"""Per-aspect / per-class evaluation of the ABSA polarity head on a labeled CSV.

This is the validation harness (Sprint P1 #3). It scores a polarity model on a
hand-labeled set and reports overall accuracy, per-class precision/recall/F1,
negative-class recall (our documented failure mode), and per-aspect accuracy.

The labeled CSV needs columns: text, span, polarity (gold), aspect, [ordinal].

Score the *current* fine-tuned head (default) on the held-out test set:

    python validation/eval_polarity.py --labels training/labels_test.csv

Score a different head, or an already-predicted column without running a model:

    python validation/eval_polarity.py --labels training/labels_test.csv \
        --polarity models/setfit_absa_polarity_finetuned.bak_pre_rebalance
    python validation/eval_polarity.py --labels training/labels_test.csv \
        --pred-col polarity_predicted
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.metrics import classification_report, recall_score

from pipeline.absa import (
    ASPECT_MODEL_ID,
    SPACY_MODEL,
    MODEL_CACHE_DIR,
    LOCAL_POLARITY_PATH,
)

LABELS = ("positive", "negative", "neutral")
# Acceptance thresholds for the validation harness (documented, not vibes):
NEG_RECALL_TARGET = 0.60   # must catch a majority of real complaints
ACCURACY_TARGET = 0.70     # overall multi-class accuracy floor


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip().replace({"nan": "", "none": ""})


def predict_with_model(df: pd.DataFrame, polarity_path: str) -> pd.Series:
    """Run only the polarity head on gold (text, span) pairs."""
    from datasets import Dataset
    from setfit import AbsaModel

    work = df.copy()
    work["ordinal"] = work.get("ordinal", 0)
    work["ordinal"] = work["ordinal"].fillna(0).astype(int)
    ds = Dataset.from_pandas(work[["text", "span", "ordinal"]], preserve_index=False)
    model = AbsaModel.from_pretrained(
        ASPECT_MODEL_ID, polarity_path,
        spacy_model=SPACY_MODEL, cache_dir=str(MODEL_CACHE_DIR),
    )
    return pd.Series(model.predict(ds)["pred_polarity"], index=df.index)


def report(df: pd.DataFrame, gold_col: str, pred_col: str, title: str) -> dict:
    work = pd.DataFrame({"g": _norm(df[gold_col]), "p": _norm(df[pred_col])})
    work["aspect"] = df["aspect"].values if "aspect" in df else ""
    n_total = len(work)
    work = work[(work["g"] != "") & (work["p"] != "")]  # drop unlabeled / no-prediction
    g, p = work["g"].tolist(), work["p"].tolist()
    skipped = n_total - len(g)

    print(f"\n{'='*60}\n{title}  (n={len(g)}"
          f"{f', {skipped} no-prediction skipped' if skipped else ''})\n{'='*60}")
    print(classification_report(g, p, labels=list(LABELS), zero_division=0, digits=3))

    acc = float((work["g"] == work["p"]).mean())
    neg_recall = recall_score(g, p, labels=["negative"], average="macro", zero_division=0)
    print(f"overall accuracy : {acc:.3f}   (target >= {ACCURACY_TARGET})  "
          f"{'PASS' if acc >= ACCURACY_TARGET else 'FAIL'}")
    print(f"negative recall  : {neg_recall:.3f}   (target >= {NEG_RECALL_TARGET})  "
          f"{'PASS' if neg_recall >= NEG_RECALL_TARGET else 'FAIL'}")

    if (work["aspect"] != "").any():
        print("\nper-aspect accuracy:")
        per = (work["g"] == work["p"]).groupby(work["aspect"].values).agg(["mean", "size"])
        for aspect, row in per.iterrows():
            print(f"  {aspect:12s} {row['mean']:.3f}  (n={int(row['size'])})")

    return {"n": int(len(g)), "accuracy": round(acc, 3),
            "negative_recall": round(float(neg_recall), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="training/labels_test.csv")
    ap.add_argument("--polarity", default=None,
                    help="Polarity model dir. Default: the current fine-tuned head.")
    ap.add_argument("--pred-col", default=None,
                    help="Use an existing prediction column instead of running a model.")
    args = ap.parse_args()

    df = pd.read_csv(Path(args.labels)).dropna(subset=["text", "span", "polarity"])

    if args.pred_col:
        pred_col, title = args.pred_col, f"{args.labels} :: column '{args.pred_col}'"
    else:
        path = args.polarity or str(LOCAL_POLARITY_PATH)
        df["_pred"] = predict_with_model(df, path)
        pred_col, title = "_pred", f"{args.labels} :: model '{path}'"

    summary = report(df, "polarity", pred_col, title)
    print(f"\nsummary: {summary}")


if __name__ == "__main__":
    main()
