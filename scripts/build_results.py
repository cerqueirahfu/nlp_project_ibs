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
import json
import shutil
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
CHUNK_SIZE = 500  # sentences per ABSA batch; controls how often progress prints

PRODUCT_COLS = [
    "product_id", "product_name", "category",
    "main_category", "sub_category_1", "sub_category_2",
    "actual_price", "discounted_price", "discount_percentage",
    "discount_amount", "discount_group",
    "rating", "rating_count", "review_title", "review_content",
]


def _span_columns(sentences_df: pd.DataFrame) -> list[str]:
    """Column set a chunk's span DataFrame must have, so empty and non-empty
    chunk checkpoints concat cleanly. Mirrors attach_predictions' output order."""
    return list(sentences_df.columns) + ["span", "polarity", "aspect"]


def _predict_chunk_safe(chunk: list[str]) -> list:
    """Predict on a chunk; if the batch raises, isolate the bad sentence(s).

    A single malformed sentence shouldn't cost the whole 500-sentence batch, so
    on failure we fall back to per-sentence inference and skip only the sentences
    that actually error (recording an empty prediction so alignment is preserved).
    """
    try:
        return predict(chunk)
    except Exception as exc:  # noqa: BLE001 - survive any model/spacy error
        print(f"  ! batch failed ({exc}); retrying sentence-by-sentence", flush=True)
        preds: list = []
        for s in chunk:
            try:
                preds.extend(predict([s]))
            except Exception as exc2:  # noqa: BLE001
                print(f"  ! skipping poison sentence ({exc2}): {s[:80]!r}", flush=True)
                preds.append([])
        return preds


def _prepare_checkpoint_dir(ckpt_dir: Path, n_sentences: int, chunk_size: int) -> None:
    """Create/validate the checkpoint dir; clear it if the inputs changed.

    A manifest pins the run to a (n_sentences, chunk_size) pair so a resume only
    reuses chunks that belong to the *same* dataset slice. Change --limit or the
    data and stale chunks are dropped instead of being silently reused.
    """
    manifest = ckpt_dir / "manifest.json"
    current = {"n_sentences": n_sentences, "chunk_size": chunk_size}
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous != current:
            print(f"  checkpoint inputs changed {previous} -> {current}; "
                  "clearing stale checkpoints")
            shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(current))


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
    parser.add_argument("--checkpoint-dir", default=".absa_checkpoints",
                        help="Where per-chunk ABSA outputs are saved so an "
                             "interrupted run resumes instead of restarting.")
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="Keep the checkpoint dir after a successful run "
                             "(default: delete it once results are written).")
    args = parser.parse_args()

    t0 = time.time()
    df = load_clean_df(Path(args.data))
    if args.limit:
        df = df.head(args.limit)
    print(f"[{time.time()-t0:5.1f}s] cleaned df: {len(df)} reviews")

    sentences_df = explode_sentences(df)
    print(f"[{time.time()-t0:5.1f}s] exploded to {len(sentences_df)} sentences")

    sentences = sentences_df["sentence"].tolist()
    n = len(sentences)
    span_cols = _span_columns(sentences_df)

    ckpt_dir = Path(args.checkpoint_dir)
    _prepare_checkpoint_dir(ckpt_dir, n, CHUNK_SIZE)

    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"[{time.time()-t0:5.1f}s] running ABSA inference on {n} sentences "
          f"in {n_chunks} chunks of {CHUNK_SIZE} (checkpoints in {ckpt_dir}/)...")

    span_frames: list[pd.DataFrame] = []
    inferred = 0          # sentences actually run this session (excludes resumed)
    infer_t0: float | None = None  # timer starts at the first real inference
    for idx, start in enumerate(range(0, n, CHUNK_SIZE)):
        cpath = ckpt_dir / f"chunk_{idx:05d}.parquet"
        if cpath.exists():
            span_frames.append(pd.read_parquet(cpath))
            print(f"[{time.time()-t0:5.1f}s] chunk {idx+1}/{n_chunks} resumed "
                  "from checkpoint", flush=True)
            continue

        if infer_t0 is None:
            infer_t0 = time.time()
        chunk = sentences[start:start + CHUNK_SIZE]
        chunk_df = sentences_df.iloc[start:start + CHUNK_SIZE]
        preds = _predict_chunk_safe(chunk)

        spans = attach_predictions(chunk_df, preds)
        if spans.empty:
            spans = pd.DataFrame(columns=span_cols)
        spans.to_parquet(cpath, index=False)  # checkpoint before moving on
        span_frames.append(spans)

        inferred += len(chunk)
        infer_elapsed = time.time() - infer_t0
        rate = inferred / infer_elapsed if infer_elapsed else 0
        eta = (n - start - len(chunk)) / rate if rate else 0
        print(f"[{time.time()-t0:5.1f}s] chunk {idx+1}/{n_chunks} done "
              f"({(start+len(chunk))/n*100:4.1f}%) ~{rate:.0f} sent/s, "
              f"eta {eta/60:4.1f} min", flush=True)

    spans_df = (
        pd.concat(span_frames, ignore_index=True)
        if span_frames else pd.DataFrame(columns=span_cols)
    )
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

    # Both parquet files are written, so the per-chunk checkpoints have served
    # their purpose. Keep them only if asked (e.g. for debugging a run).
    if args.keep_checkpoints:
        print(f"[{time.time()-t0:5.1f}s] keeping checkpoints in {ckpt_dir}/")
    else:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        print(f"[{time.time()-t0:5.1f}s] cleaned up checkpoints")


if __name__ == "__main__":
    main()
