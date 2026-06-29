"""Regenerate polarity predictions with the FINE-TUNED model and re-run the
metric + error analysis, so we can see what fine-tuning actually fixed.

`training/labels.csv` already holds:
    polarity            -> gold (hand-corrected)
    polarity_predicted  -> BASELINE off-the-shelf SetFitABSA prediction

This script adds:
    polarity_finetuned  -> the fine-tuned polarity head's prediction on the SAME
                           gold spans, via AbsaModel.predict(Dataset[text,span,ordinal])
                           which reuses SetFit's own spaCy windowing (span_context=3),
                           matching how the model was trained.

Outputs:
    training/labels_finetuned_pred.csv   (full table incl. new column)
    validation/flagged_errors_finetuned.csv
    prints a baseline-vs-fine-tuned comparison (accuracy, macro/weighted-F1,
    confusion, and the error-category counts from error_analysis.md)

Run from project root:
    nlpenv/Scripts/python.exe validation/regen_finetuned_preds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.absa import load_model  # picks up the fine-tuned head if present

LABEL_ORDER = ["negative", "neutral", "positive"]
SEED, TEST_SIZE = 42, 0.2


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "polarity_predicted" not in df.columns:
        df["polarity_predicted"] = pd.NA  # custom test files may lack the baseline col
    df = df[["text", "span", "polarity", "ordinal", "polarity_predicted", "aspect"]]
    df = df.dropna(subset=["text", "span", "polarity"])
    df["polarity"] = df["polarity"].astype(str).str.strip().str.lower()
    df["polarity_predicted"] = df["polarity_predicted"].astype(str).str.strip().str.lower()
    df["ordinal"] = df["ordinal"].fillna(0).astype(int)
    return df.reset_index(drop=True)


def held_out_mask(df: pd.DataFrame) -> pd.Series:
    shuffled = df.sample(frac=1.0, random_state=SEED)
    cut = max(1, int(len(shuffled) * (1 - TEST_SIZE)))
    eval_idx = shuffled.index[cut:]
    return df.index.isin(eval_idx)


def metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro-F1": f1_score(y_true, y_pred, average="macro", labels=LABEL_ORDER, zero_division=0),
        "Weighted-F1": f1_score(y_true, y_pred, average="weighted", labels=LABEL_ORDER, zero_division=0),
    }


def show_block(name: str, df: pd.DataFrame, has_baseline: bool = True) -> None:
    print(f"\n{'=' * 66}\n{name}  (n={len(df)})\n{'=' * 66}")
    rows = {}
    cols = []
    if has_baseline:
        rows["Baseline (off-the-shelf)"] = metrics(df["polarity"], df["polarity_predicted"])
        cols.append(("BASELINE", "polarity_predicted"))
    rows["Fine-tuned"] = metrics(df["polarity"], df["polarity_finetuned"])
    cols.append(("FINE-TUNED", "polarity_finetuned"))
    print(pd.DataFrame(rows).T.to_string(float_format=lambda x: f"{x:.3f}"))
    # negative recall is the headline metric for this rebalancing experiment
    neg = df[df["polarity"] == "negative"]
    if len(neg):
        for who, col in cols:
            r = (neg[col] == "negative").mean()
            print(f"  {who} negative recall: {r:.3f}  ({int((neg[col]=='negative').sum())}/{len(neg)})")
    for who, col in cols:
        print(f"\n{who} confusion (rows=gold {LABEL_ORDER}, cols=pred):")
        print(confusion_matrix(df["polarity"], df[col], labels=LABEL_ORDER))


def error_pairs(df: pd.DataFrame, col: str) -> pd.Series:
    e = df[df["polarity"] != df[col]]
    return (e["polarity"] + " -> " + e[col]).value_counts()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ROOT / "training" / "labels.csv"),
                    help="CSV to score. A dedicated test file is scored whole; "
                         "the default labels.csv also gets the seed-42 sub-split.")
    args = ap.parse_args()
    labels_path = Path(args.labels)
    is_default = labels_path.resolve() == (ROOT / "training" / "labels.csv").resolve()

    df = load(labels_path)
    has_baseline = df["polarity_predicted"].notna().any() and (df["polarity_predicted"] != "nan").any()

    print("Loading fine-tuned AbsaModel (this loads aspect + fine-tuned polarity heads)...")
    model = load_model()
    print(f"Predicting polarity for {len(df)} gold spans (row-by-row)...")
    # Row-by-row: SetFit's batched predict_dataset miscounts rows when the same
    # `text` repeats (our data has that) and crashes; a 1-row Dataset avoids the
    # grouping bug and returns None when spaCy can't locate the span.
    preds: list[str | None] = []
    for i, r in enumerate(df.itertuples(), 1):
        one = Dataset.from_dict({"text": [r.text], "span": [r.span], "ordinal": [int(r.ordinal)]})
        try:
            p = model.predict(one)["pred_polarity"][0]
        except Exception:
            p = None
        preds.append(p.lower() if isinstance(p, str) else None)
        if i % 25 == 0:
            print(f"  {i}/{len(df)}")
    df["polarity_finetuned"] = preds

    n_skipped = df["polarity_finetuned"].isna().sum()
    if n_skipped:
        print(f"\nNOTE: {n_skipped} spans could not be located by spaCy (pred=None); "
              "excluded from scoring.")
    scored = df.dropna(subset=["polarity_finetuned"]).copy()

    # Persist artifacts
    df.to_csv(ROOT / "training" / "labels_finetuned_pred.csv", index=False)
    ft_err = scored[scored["polarity"] != scored["polarity_finetuned"]]
    ft_err.assign(error_pair=ft_err["polarity"] + " -> " + ft_err["polarity_finetuned"])[
        ["aspect", "span", "polarity", "polarity_finetuned", "text"]
    ].to_csv(ROOT / "validation" / "flagged_errors_finetuned.csv", index=False)

    title = "FULL LABELED SET" if is_default else f"TEST SET ({labels_path.name})"
    show_block(title, scored, has_baseline)
    if is_default:
        show_block("HELD-OUT TEST SET (seed=42, 20%)", scored[held_out_mask(scored)], has_baseline)

    print(f"\n{'=' * 66}\nERROR-PAIR COUNTS\n{'=' * 66}")
    cols = {"fine_tuned": error_pairs(scored, "polarity_finetuned")}
    if has_baseline:
        cols = {"baseline": error_pairs(scored, "polarity_predicted"), **cols}
    print(pd.DataFrame(cols).fillna(0).astype(int).to_string())
    base_errs = (len(scored) - int((scored.polarity == scored.polarity_predicted).sum())
                 if has_baseline else "n/a")
    print(f"\nTotal errors  baseline={base_errs}  fine_tuned={len(ft_err)}  of {len(scored)}")
    print("\nWrote training/labels_finetuned_pred.csv and validation/flagged_errors_finetuned.csv")


if __name__ == "__main__":
    main()
