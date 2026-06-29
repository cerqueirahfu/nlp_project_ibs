"""Mine candidate NEGATIVE aspect-spans from the review corpus, to be labeled by
hand and added to training/labels.csv (rebalancing the 31-negative minority class).

Why not rating-based? `products.parquet.rating` is the product's AGGREGATE star
rating (mean 4.1, min 2.8) — the Amazon Sales Dataset has no per-review stars, so
there is no "1-2 star review" signal. The only weak-label signal available is
LINGUISTIC: a sentence that mentions an aspect AND carries a complaint/negation cue.

These are CANDIDATES, not gold. `polarity` is pre-filled "negative" as a guess;
you MUST review training/negative_candidates.csv and fix/delete wrong rows before
training. Output schema matches labels.csv (text, span, polarity, ordinal, aspect)
so confirmed rows can be appended directly.

Anti-context-bleed (the failure mode from error_analysis.md):
  - only keep reasonably short sentences (run-on blobs are dropped), and
  - require the complaint cue to sit CLOSE to the aspect keyword.
Sentences already present in labels.csv are excluded so we never mine a test-set
sentence (no leakage).

Run from project root:
    nlpenv/Scripts/python.exe validation/mine_negative_candidates.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.aggregate import ASPECT_KEYWORDS  # reuse the canonical keyword lists
from pipeline.clean import _clean_text, _split_sentences  # reuse exact cleaning

# Complaint / negation cues. Some overlap with aspect keywords (slow, lag, cheap,
# expensive) — that's fine, they're still negativity signals.
CUES = (
    "not", "no ", "n't", "never", "but ", "however", "issue", "issues", "problem",
    "problems", "poor", "bad", "worst", "worse", "broke", "broken", "stopped",
    "stop working", "fails", "failed", "fault", "faulty", "defect", "defective",
    "damaged", "damage", "disappoint", "waste", "return", "refund", "useless",
    "overpriced", "expensive", "costly", "slow", "lag", "drain", "drains", "heats",
    "heating", "hot ", "cheap", "flimsy", "doesn", "didn", "won't", "wont", "can't",
    "cant", "not worth", "stopped working", "dead", "died", "loose", "missing",
)

MIN_LEN, MAX_LEN = 25, 220   # chars; drop run-on blobs and trivially short frags
MAX_CUE_DIST = 60            # chars between aspect keyword and nearest cue
PER_ASPECT_CAP = 40          # keep candidates balanced across aspects
KEYWORD_RE = {
    aspect: re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")\b")
    for aspect, kws in ASPECT_KEYWORDS.items()
}


def existing_sentences() -> set[str]:
    df = pd.read_csv(ROOT / "training" / "labels.csv")
    return {str(t).strip().lower() for t in df["text"].dropna()}


def cue_hits(sentence: str) -> list[str]:
    return [c.strip() for c in CUES if c in sentence]


def nearest_cue_dist(sentence: str, kw_start: int, kw_end: int) -> int:
    best = 10**6
    for c in CUES:
        i = sentence.find(c)
        while i != -1:
            d = 0 if i <= kw_end and i + len(c) >= kw_start else min(
                abs(i - kw_end), abs(kw_start - (i + len(c)))
            )
            best = min(best, d)
            i = sentence.find(c, i + 1)
    return best


def mine() -> pd.DataFrame:
    products = pd.read_parquet(ROOT / "products.parquet")
    seen = existing_sentences()
    rows: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()

    for prod in products.itertuples():
        pid = prod.product_id
        for sent in _split_sentences(_clean_text(prod.review_content)):
            if not (MIN_LEN <= len(sent) <= MAX_LEN):
                continue
            if sent.lower() in seen:
                continue
            hits = cue_hits(sent)
            if not hits:
                continue
            for aspect, rx in KEYWORD_RE.items():
                m = rx.search(sent)
                if not m:
                    continue
                dist = nearest_cue_dist(sent, m.start(), m.end())
                if dist > MAX_CUE_DIST:
                    continue
                span = m.group(0)
                ordinal = sent[: m.start()].count(span)
                key = (sent, span)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                rows.append({
                    "text": sent,
                    "span": span,
                    "polarity": "negative",      # CANDIDATE guess — verify!
                    "ordinal": ordinal,
                    "aspect": aspect,
                    "cue_hits": "|".join(sorted(set(hits))),
                    "cue_dist": dist,
                    "product_id": pid,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Rank: cue close to aspect first, then more cues. Cap per aspect for balance.
    df["n_cues"] = df["cue_hits"].str.count(r"\|") + 1
    df = df.sort_values(["aspect", "cue_dist", "n_cues"], ascending=[True, True, False])
    df = df.groupby("aspect", group_keys=False).head(PER_ASPECT_CAP).reset_index(drop=True)
    return df


def main() -> None:
    df = mine()
    if df.empty:
        print("No candidates found.")
        return
    out = ROOT / "training" / "negative_candidates.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} candidate negative spans to {out}\n")
    print("Per-aspect counts (target: top up the thin aspects):")
    print(df["aspect"].value_counts().to_string())
    print("\nReminder: these are WEAK candidates. Review the `polarity` column "
          "(some will be neutral/positive), delete bad rows, then append the kept "
          "rows to training/labels.csv and re-run train_polarity.py + "
          "validation/regen_finetuned_preds.py.")
    print("\nSample (closest cue-to-aspect first):")
    with pd.option_context("display.max_colwidth", 90, "display.width", 200):
        print(df.head(12)[["aspect", "span", "cue_hits", "text"]].to_string(index=False))


if __name__ == "__main__":
    main()
