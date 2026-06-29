"""Validate the hand-reviewed negative candidates and merge them into a balanced
training file (non-destructive: writes training/labels_balanced.csv, leaves
training/labels.csv untouched).

Checks each candidate row:
  - polarity in {positive, negative, neutral}
  - span is a substring of text, and is locatable as a STRICT spaCy token span
    (same requirement SetFit's polarity model has — otherwise it skips the row)
  - recomputes `ordinal` to the first valid occurrence
  - drops rows whose (text, span) already exist in labels.csv or duplicate each other

Run from project root:
    nlpenv/Scripts/python.exe validation/merge_labels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import spacy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LABELS = ROOT / "training" / "labels.csv"
CANDIDATES = ROOT / "training" / "negative_candidates.csv"
OUT = ROOT / "training" / "labels_balanced.csv"
VALID = {"positive", "negative", "neutral"}
COLS = ["text", "span", "polarity", "polarity_predicted", "aspect", "ordinal"]

nlp = spacy.load("en_core_web_sm")


def locatable_ordinal(text: str, span: str) -> int | None:
    """Return the occurrence index of `span` that aligns to spaCy token
    boundaries (what SetFit needs), or None if no occurrence does."""
    doc = nlp(text)
    start = text.find(span)
    occ = 0
    while start != -1:
        if doc.char_span(start, start + len(span), alignment_mode="strict") is not None:
            return occ
        occ += 1
        start = text.find(span, start + 1)
    return None


def main() -> None:
    base = pd.read_csv(LABELS)
    cand = pd.read_csv(CANDIDATES)
    for c in ("polarity",):
        cand[c] = cand[c].astype(str).str.strip().str.lower()
    cand["text"] = cand["text"].astype(str)
    cand["span"] = cand["span"].astype(str)

    existing_pairs = {(str(t).strip().lower(), str(s).strip().lower())
                      for t, s in zip(base["text"], base["span"])}

    kept, dropped = [], []
    seen: set[tuple[str, str]] = set()
    for r in cand.itertuples():
        text, span, pol = r.text, r.span, r.polarity
        key = (text.strip().lower(), span.strip().lower())
        if pol not in VALID:
            dropped.append((span, "invalid polarity")); continue
        if span not in text:
            dropped.append((span, "span not substring of text")); continue
        if key in existing_pairs:
            dropped.append((span, "already in labels.csv")); continue
        if key in seen:
            dropped.append((span, "duplicate within candidates")); continue
        ordinal = locatable_ordinal(text, span)
        if ordinal is None:
            dropped.append((span, "span not a strict spaCy token span")); continue
        seen.add(key)
        kept.append({
            "text": text, "span": span, "polarity": pol,
            "polarity_predicted": pd.NA, "aspect": r.aspect, "ordinal": ordinal,
        })

    kept_df = pd.DataFrame(kept, columns=COLS)
    merged = pd.concat([base[COLS], kept_df], ignore_index=True)
    merged.to_csv(OUT, index=False)

    print(f"Candidates in:           {len(cand)}")
    print(f"Kept (validated):        {len(kept_df)}")
    print(f"Dropped:                 {len(dropped)}")
    if dropped:
        reasons = pd.Series([d[1] for d in dropped]).value_counts()
        print("  drop reasons:")
        for reason, n in reasons.items():
            print(f"    {n:>3}  {reason}")
    print(f"\nWrote {OUT}  ({len(merged)} rows total)\n")

    def dist(df, name):
        print(f"{name}: " + ", ".join(
            f"{k}={v}" for k, v in df["polarity"].astype(str).str.lower().value_counts().items()))
    dist(base, "labels.csv (before)   ")
    dist(merged, "labels_balanced (after)")
    print("\nNegative class: "
          f"{(base['polarity'].str.lower()=='negative').sum()} -> "
          f"{(merged['polarity'].str.lower()=='negative').sum()}")
    print("\nNext: train on the balanced file, then re-evaluate:")
    print("  nlpenv/Scripts/python.exe training/train_polarity.py --labels training/labels_balanced.csv")
    print("  nlpenv/Scripts/python.exe validation/regen_finetuned_preds.py")


if __name__ == "__main__":
    main()
