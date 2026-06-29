"""Fine-tune the SetFit-ABSA polarity head on a hand-labeled CSV.

Workflow:
    1. Generate `training/labels.csv` from the notebook cell that calls
       `predict()` on a sample and writes predictions to disk.
    2. Open `training/labels.csv` and correct the `polarity` column. Valid
       values: positive, negative, neutral. Delete rows you can't label.
    3. Run this script from the project root:

           python training/train_polarity.py --labels training/labels.csv

    4. pipeline/absa.py will pick up the fine-tuned polarity model from
       models/setfit_absa_polarity_finetuned/ on the next predict() call.

CSV columns required: text, span, polarity, ordinal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from datasets import Dataset
from setfit import AbsaModel, AbsaTrainer, TrainingArguments

from pipeline.absa import (
    ASPECT_MODEL_ID,
    POLARITY_MODEL_ID,
    SPACY_MODEL,
    MODEL_CACHE_DIR,
    LOCAL_POLARITY_PATH,
)

REQUIRED_COLS = ("text", "span", "polarity", "ordinal")
VALID_POLARITIES = {"positive", "negative", "neutral"}


def load_labels(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"labels CSV is missing required columns: {missing}")
    df = df[list(REQUIRED_COLS)].dropna(subset=["text", "span", "polarity"])
    df["polarity"] = df["polarity"].astype(str).str.strip().str.lower()
    bad = sorted(set(df["polarity"]) - VALID_POLARITIES)
    if bad:
        raise SystemExit(
            f"Found invalid polarity values: {bad}. "
            f"Use one of {sorted(VALID_POLARITIES)}."
        )
    df["ordinal"] = df["ordinal"].fillna(0).astype(int)
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="training/labels.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-iterations", type=int, default=3,
                        help="SetFit contrastive iterations. Default 3 keeps CPU "
                             "training to ~30 min; SetFit's own default of 20 takes hours.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap training set size. Useful for a fast first pass.")
    args = parser.parse_args()

    labels = load_labels(Path(args.labels))
    print(f"Loaded {len(labels)} labeled rows. Polarity distribution:")
    print(labels["polarity"].value_counts())

    shuffled = labels.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    shuffled = shuffled.rename(columns={"polarity": "label"})
    cut = max(1, int(len(shuffled) * (1 - args.test_size)))
    train_df = shuffled.iloc[:cut].reset_index(drop=True)
    eval_df = shuffled.iloc[cut:].reset_index(drop=True)
    if args.max_samples is not None and len(train_df) > args.max_samples:
        train_df = train_df.sample(args.max_samples, random_state=args.seed).reset_index(drop=True)
    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(eval_df)
    print(f"Train: {len(train_ds)} rows | Eval: {len(eval_ds)} rows | "
          f"num_iterations={args.num_iterations}")

    model = AbsaModel.from_pretrained(
        ASPECT_MODEL_ID,
        POLARITY_MODEL_ID,
        spacy_model=SPACY_MODEL,
        cache_dir=str(MODEL_CACHE_DIR),
    )

    has_eval = len(eval_ds) > 0
    training_args = TrainingArguments(
        output_dir=str(LOCAL_POLARITY_PATH / "checkpoints"),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        eval_strategy="epoch" if has_eval else "no",
        save_strategy="no",
        load_best_model_at_end=False,
    )

    trainer = AbsaTrainer(
        model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    trainer.train_polarity()

    if len(eval_ds) > 0:
        metrics = trainer.evaluate(eval_ds)
        print("\nEval metrics:")
        print(metrics)

    LOCAL_POLARITY_PATH.mkdir(parents=True, exist_ok=True)
    model.polarity_model.save_pretrained(str(LOCAL_POLARITY_PATH))
    print(f"\nSaved fine-tuned polarity model to {LOCAL_POLARITY_PATH}")
    print("pipeline/absa.py will pick it up automatically on next predict() call.")


if __name__ == "__main__":
    main()
