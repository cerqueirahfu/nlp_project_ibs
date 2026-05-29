"""Load and clean the Amazon dataset, mirroring the steps in data/data_analysis.ipynb.

The notebook is the exploratory workbench; this module is the importable version
that scripts (training, build_results) use so cleaning stays in sync.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

_URL_RE = re.compile(r"http\S+|www\S+")
_HTML_RE = re.compile(r"<.*?>")
_NON_PRINT_RE = re.compile(r"[^a-zA-Z0-9\s.,!?']")
_WS_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

KEEP_COLS = [
    "product_id", "product_name", "category", "main_category",
    "sub_category_1", "sub_category_2",
    "discounted_price", "actual_price", "discount_percentage",
    "rating", "rating_count", "review_title", "review_content",
]

# The ABSA aspect/polarity models are trained for electronics-style reviews,
# so we scope the dataset to these two top-level categories. The raw Amazon
# Sales Dataset also contains Home&Kitchen, OfficeProducts, etc. — excluded.
ALLOWED_MAIN_CATEGORIES = ("Electronics", "Computers&Accessories")


def _clean_text(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = _URL_RE.sub("", text)
    text = _HTML_RE.sub("", text)
    text = _NON_PRINT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def load_clean_df(csv_path: Path | str) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    data = data.dropna(subset=["rating_count"])

    for c in ("discounted_price", "actual_price", "rating_count"):
        data[c] = (
            data[c].astype(str)
            .str.replace("₹", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

    data["discount_percentage"] = (
        data["discount_percentage"].astype(str).str.replace("%", "", regex=False).astype(int)
    )
    data["rating"] = (
        data["rating"].astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )
    data = data.dropna(subset=["rating"])
    category_parts = data["category"].str.split("|")
    data["main_category"] = category_parts.str[0]
    data["sub_category_1"] = category_parts.str[1]
    data["sub_category_2"] = category_parts.str[2]

    data = data[data["main_category"].isin(ALLOWED_MAIN_CATEGORIES)]

    df = data[KEEP_COLS].copy()
    df["discount_amount"] = df["actual_price"] - df["discounted_price"]
    df["discount_group"] = pd.cut(
        df["discount_percentage"],
        bins=[0, 15, 30, 50, 100],
        labels=["0-15%", "15-30%", "30-50%", "50%+"],
        include_lowest=True,
    )
    df["clean_review"] = df["review_content"].apply(_clean_text)
    df["review_sentences"] = df["clean_review"].apply(_split_sentences)
    return df.reset_index(drop=True)
