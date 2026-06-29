"""Build a clean, leakage-free train/test split for the rebalanced experiment.

Decisions (locked in):
  - TEST = the original seed-42 / 20% gold held-out split from labels.csv (47 rows).
    Pure gold, comparable to every prior report, EXCLUDED from training.
  - TRAIN = the other 186 gold rows + reviewed mined rows, with mined NEGATIVES
    CAPPED so total train negatives ~= total train positives (no over-correction).

Writes:
  training/labels_train.csv   (model trains on this)
  training/labels_test.csv    (held-out gold; regen_finetuned_preds.py scores this)

Run from project root:
    nlpenv/Scripts/python.exe validation/build_split.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LABELS = ROOT / "training" / "labels.csv"
BALANCED = ROOT / "training" / "labels_balanced.csv"
COLS = ["text", "span", "polarity", "polarity_predicted", "aspect", "ordinal"]
SEED, TEST_SIZE = 42, 0.2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="Target positive:negative ratio in the TRAIN set. 1.0 = fully "
                         "balanced (over-predicts negatives on this positive-skewed data); "
                         "~2.5 keeps recall gains while pulling absolute negatives toward "
                         "the real ~13%% prior.")
    args = ap.parse_args()
    ratio = args.ratio

    # --- reproduce the exact seed-42 gold held-out split (matches train_polarity.py) ---
    base = pd.read_csv(LABELS)[COLS].copy()
    base["polarity"] = base["polarity"].astype(str).str.strip().str.lower()
    base = base.dropna(subset=["text", "span", "polarity"]).reset_index(drop=True)
    shuffled = base.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    cut = max(1, int(len(shuffled) * (1 - TEST_SIZE)))
    test = shuffled.iloc[cut:].reset_index(drop=True)          # 47 gold rows
    train_gold = shuffled.iloc[:cut].reset_index(drop=True)    # 186 gold rows

    # --- mined rows = everything appended after the original 233 in labels_balanced ---
    balanced = pd.read_csv(BALANCED)[COLS].copy()
    balanced["polarity"] = balanced["polarity"].astype(str).str.strip().str.lower()
    mined = balanced.iloc[len(base):].reset_index(drop=True)
    mined_neg = mined[mined["polarity"] == "negative"].reset_index(drop=True)
    mined_other = mined[mined["polarity"] != "negative"].reset_index(drop=True)

    # --- cap mined negatives so train negatives ~= train positives ---
    train_pos = int((train_gold["polarity"] == "positive").sum()) + \
        int((mined_other["polarity"] == "positive").sum())
    gold_neg = int((train_gold["polarity"] == "negative").sum())
    target_neg_total = int(round(train_pos / ratio))
    need_neg = max(0, target_neg_total - gold_neg)
    # stratified sample across aspects, deterministic, then trim/top-up to need_neg
    if need_neg < len(mined_neg):
        frac = need_neg / len(mined_neg)
        per = mined_neg.groupby("aspect", group_keys=False).sample(frac=frac, random_state=SEED)
        if len(per) < need_neg:
            extra = mined_neg.drop(per.index).sample(need_neg - len(per), random_state=SEED)
            per = pd.concat([per, extra])
        elif len(per) > need_neg:
            per = per.sample(need_neg, random_state=SEED)
        mined_neg_keep = per.reset_index(drop=True)
    else:
        mined_neg_keep = mined_neg

    train = pd.concat([train_gold, mined_other, mined_neg_keep], ignore_index=True)
    # Guarantee zero leakage: drop any train row whose (text, span, ordinal) is in TEST
    # (guards against duplicate gold tuples landing on both sides of the split).
    test_keys = set(map(tuple, test[["text", "span", "ordinal"]].astype(str).values))
    mask = [tuple(map(str, k)) not in test_keys
            for k in train[["text", "span", "ordinal"]].values]
    train = train[mask].reset_index(drop=True)
    train = train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    train.to_csv(ROOT / "training" / "labels_train.csv", index=False)
    test.to_csv(ROOT / "training" / "labels_test.csv", index=False)

    def dist(df):
        vc = df["polarity"].value_counts()
        return ", ".join(f"{k}={int(vc.get(k,0))}" for k in ("negative", "neutral", "positive"))

    print(f"TRAIN  -> training/labels_train.csv  ({len(train)} rows):  {dist(train)}")
    print(f"  (gold {len(train_gold)} + mined_other {len(mined_other)} + mined_neg {len(mined_neg_keep)} "
          f"of {len(mined_neg)} available; target pos:neg = {ratio:.1f}:1 "
          f"-> ~{train_pos} pos / ~{target_neg_total} neg)")
    print(f"TEST   -> training/labels_test.csv   ({len(test)} rows):  {dist(test)}")
    print("\nLeakage check: TEST rows present in TRAIN ->",
          len(pd.merge(test[["text", "span", "ordinal"]], train[["text", "span", "ordinal"]])))
    print("\nNext:")
    print("  nlpenv/Scripts/python.exe training/train_polarity.py --labels training/labels_train.csv --test-size 0")
    print("  nlpenv/Scripts/python.exe validation/regen_finetuned_preds.py --labels training/labels_test.csv")


if __name__ == "__main__":
    main()
