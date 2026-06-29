"""Quantitative evaluation of the aspect-polarity classifier on a held-out set.

The task: aspect-based sentiment classification — given a review and an aspect
span (e.g. "battery capacity"), predict polarity in {positive, negative, neutral}.

Gold labels live in training/labels.csv:
    - `polarity`            -> human-corrected gold label
    - `polarity_predicted`  -> off-the-shelf SetFitABSA prediction (the baseline,
                               generated BEFORE any fine-tuning, so it never saw
                               these labels — no leakage)

This script reproduces the EXACT held-out split that training/train_polarity.py
reserved (seed=42, test_size=0.2) so the 47-row eval set was never used to
develop the fine-tuned model. It reports accuracy and F1 variants on both the
held-out split (primary) and the full labeled set (more stable estimate).

Run from the project root:
    nlpenv/Scripts/python.exe validation/evaluate_polarity.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "training" / "labels.csv"
LABEL_ORDER = ["negative", "neutral", "positive"]
SEED = 42          # must match train_polarity.py
TEST_SIZE = 0.2    # must match train_polarity.py


def load() -> pd.DataFrame:
    df = pd.read_csv(LABELS)
    df = df[["text", "span", "polarity", "ordinal", "polarity_predicted", "aspect"]]
    df = df.dropna(subset=["text", "span", "polarity"])
    for col in ("polarity", "polarity_predicted"):
        df[col] = df[col].astype(str).str.strip().str.lower()
    df["ordinal"] = df["ordinal"].fillna(0).astype(int)
    return df.reset_index(drop=True)


def held_out(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce train_polarity.py's eval split exactly."""
    shuffled = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    cut = max(1, int(len(shuffled) * (1 - TEST_SIZE)))
    return shuffled.iloc[cut:].reset_index(drop=True)


def report(name: str, df: pd.DataFrame) -> None:
    y_true, y_pred = df["polarity"], df["polarity_predicted"]
    print(f"\n{'=' * 60}\n{name}  (n={len(df)})\n{'=' * 60}")
    print(f"Accuracy     : {accuracy_score(y_true, y_pred):.3f}")
    print(f"Macro-F1     : {f1_score(y_true, y_pred, average='macro', labels=LABEL_ORDER, zero_division=0):.3f}")
    print(f"Weighted-F1  : {f1_score(y_true, y_pred, average='weighted', labels=LABEL_ORDER, zero_division=0):.3f}")
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, labels=LABEL_ORDER, zero_division=0, digits=3))
    print("Confusion matrix (rows=gold, cols=pred), order=" + str(LABEL_ORDER))
    print(confusion_matrix(y_true, y_pred, labels=LABEL_ORDER))


def per_aspect(df: pd.DataFrame) -> None:
    print(f"\n{'=' * 60}\nPer-aspect accuracy (full labeled set)\n{'=' * 60}")
    g = df.groupby("aspect").apply(
        lambda d: pd.Series({
            "n": len(d),
            "accuracy": accuracy_score(d["polarity"], d["polarity_predicted"]),
        }),
        include_groups=False,
    ).sort_values("n", ascending=False)
    print(g.to_string(float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    df = load()
    report("HELD-OUT TEST SET (seed=42, 20% never used in fine-tuning)", held_out(df))
    report("FULL LABELED SET (off-the-shelf baseline, no training leakage)", df)
    per_aspect(df)


if __name__ == "__main__":
    main()
