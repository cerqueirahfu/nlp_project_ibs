"""Run the full ABSA pipeline on the Amazon dataset and write results.parquet.

This is the Phase 1 deliverable per CLAUDE.md: aggregated sentiment per
product x aspect x discount tier, ready to upload to S3 for the EC2-hosted
Streamlit dashboard to read at startup.

Run from project root (full dataset takes ~20-40 min on CPU):

    python scripts/build_results.py

Smoke test on a small slice first (recommended):

    python scripts/build_results.py --limit 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from pipeline.absa import predict
from pipeline.aggregate import attach_predictions
from pipeline.clean import load_clean_df
from pipeline.preprocess import explode_sentences

GROUP_COLS = [
    "product_id", "product_name",
    "main_category", "sub_category_1", "sub_category_2",
    "discount_group", "aspect",
]
POLARITIES = ("positive", "negative", "neutral")

PRODUCT_COLS = [
    "product_id", "product_name", "category",
    "main_category", "sub_category_1", "sub_category_2",
    "actual_price", "discounted_price", "discount_percentage",
    "discount_amount", "discount_group",
    "rating", "rating_count", "review_title", "review_content",
]


def aggregate(spans_df: pd.DataFrame) -> pd.DataFrame:
    if spans_df.empty:
        return spans_df
    counts = (
        spans_df.groupby(GROUP_COLS + ["polarity"], observed=True)
        .size()
        .unstack("polarity", fill_value=0)
        .reset_index()
    )
    for col in POLARITIES:
        if col not in counts.columns:
            counts[col] = 0
    counts["total"] = counts[list(POLARITIES)].sum(axis=1)
    counts["positive_share"] = counts["positive"] / counts["total"]
    counts["negative_share"] = counts["negative"] / counts["total"]
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/amazon.csv")
    parser.add_argument("--out", default="results.parquet")
    parser.add_argument("--products-out", default="products.parquet")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap reviews processed; use for a fast smoke test before the full run.")
    args = parser.parse_args()

    t0 = time.time()
    df = load_clean_df(Path(args.data))
    if args.limit:
        df = df.head(args.limit)
    print(f"[{time.time()-t0:5.1f}s] cleaned df: {len(df)} reviews")

    sentences_df = explode_sentences(df)
    print(f"[{time.time()-t0:5.1f}s] exploded to {len(sentences_df)} sentences")

    print(f"[{time.time()-t0:5.1f}s] running ABSA inference...")
    preds = predict(sentences_df["sentence"].tolist())
    spans_df = attach_predictions(sentences_df, preds)
    print(f"[{time.time()-t0:5.1f}s] {len(spans_df)} aspect spans after keyword filter")

    results = aggregate(spans_df)
    print(f"[{time.time()-t0:5.1f}s] aggregated to {len(results)} product x aspect x tier rows")

    out_path = Path(args.out)
    results.to_parquet(out_path, index=False)
    print(f"[{time.time()-t0:5.1f}s] wrote {out_path} "
          f"({out_path.stat().st_size / 1024:.1f} KB)")

    products = df[[c for c in PRODUCT_COLS if c in df.columns]].copy()
    products = products.drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    products_path = Path(args.products_out)
    products.to_parquet(products_path, index=False)
    print(f"[{time.time()-t0:5.1f}s] wrote {products_path} "
          f"({products_path.stat().st_size / 1024:.1f} KB, {len(products)} products)")


if __name__ == "__main__":
    main()
