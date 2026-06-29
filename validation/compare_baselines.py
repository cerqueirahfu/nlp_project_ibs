"""Baseline comparison for the aspect-polarity classifier.

Scores three systems on the SAME test set with the SAME metrics (accuracy,
macro-F1) used in validation/evaluate_polarity.py:

    A) Naive      -> majority class: always predict the dominant label (positive).
                     No text, no ML.
    B) Ablation   -> VADER lexicon sentiment on the span's sentence. Keeps the
                     pipeline but strips out the learned SetFitABSA semantic
                     model — sentiment from a fixed word list instead.
    C) Full       -> SetFitABSA prediction (the `polarity_predicted` column).

Run from the project root:
    nlpenv/Scripts/python.exe validation/compare_baselines.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "training" / "labels.csv"
LABEL_ORDER = ["negative", "neutral", "positive"]
SEED = 42
TEST_SIZE = 0.2

# Standard VADER thresholds for 3-way classification.
VADER_POS = 0.05
VADER_NEG = -0.05


def load() -> pd.DataFrame:
    df = pd.read_csv(LABELS)
    df = df[["text", "span", "polarity", "ordinal", "polarity_predicted", "aspect"]]
    df = df.dropna(subset=["text", "span", "polarity"])
    for col in ("polarity", "polarity_predicted"):
        df[col] = df[col].astype(str).str.strip().str.lower()
    df["ordinal"] = df["ordinal"].fillna(0).astype(int)
    return df.reset_index(drop=True)


def held_out(df: pd.DataFrame) -> pd.DataFrame:
    shuffled = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    cut = max(1, int(len(shuffled) * (1 - TEST_SIZE)))
    return shuffled.iloc[cut:].reset_index(drop=True)


def naive_pred(df: pd.DataFrame) -> pd.Series:
    majority = df["polarity"].value_counts().idxmax()
    return pd.Series([majority] * len(df), index=df.index)


def vader_pred(df: pd.DataFrame) -> pd.Series:
    analyzer = SentimentIntensityAnalyzer()

    def classify(sentence: str) -> str:
        c = analyzer.polarity_scores(str(sentence))["compound"]
        if c >= VADER_POS:
            return "positive"
        if c <= VADER_NEG:
            return "negative"
        return "neutral"

    return df["text"].map(classify)


def scores(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro-F1": f1_score(y_true, y_pred, average="macro", labels=LABEL_ORDER, zero_division=0),
        "Weighted-F1": f1_score(y_true, y_pred, average="weighted", labels=LABEL_ORDER, zero_division=0),
    }


def table(name: str, df: pd.DataFrame) -> None:
    rows = {
        "A) Naive (majority=positive)": scores(df["polarity"], naive_pred(df)),
        "B) Ablation (VADER lexicon)": scores(df["polarity"], vader_pred(df)),
        "C) Full (SetFitABSA)": scores(df["polarity"], df["polarity_predicted"]),
    }
    out = pd.DataFrame(rows).T
    print(f"\n{'=' * 64}\n{name}  (n={len(df)})\n{'=' * 64}")
    print(out.to_string(float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    df = load()
    table("HELD-OUT TEST SET (seed=42, 20%)", held_out(df))
    table("FULL LABELED SET", df)


if __name__ == "__main__":
    main()
