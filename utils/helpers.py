"""Shared UI + data loaders for the Streamlit dashboard pages."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_PATH = ROOT / "results.parquet"
DEFAULT_PRODUCTS_PATH = ROOT / "products.parquet"

DISCOUNT_TIERS = ("0-15%", "15-30%", "30-50%", "50%+")
HIGH_DISCOUNT_TIERS = ("30-50%", "50%+")
ALL = "All"

SIDEBAR_CSS = """
<style>
[data-testid="stSidebar"] {
    background-color: #1E2A3A;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #F5F0E8 !important;
}
</style>
"""


def apply_brand_css() -> None:
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)


def _resolve(uri_env: str, default: Path) -> str | None:
    uri = os.environ.get(uri_env, str(default))
    if str(uri).startswith("s3://"):
        return uri
    return uri if Path(uri).exists() else None


@st.cache_data(ttl=3600, show_spinner="Loading results...")
def load_results() -> pd.DataFrame | None:
    uri = _resolve("RESULTS_S3_URI", DEFAULT_RESULTS_PATH)
    return pd.read_parquet(uri) if uri else None


@st.cache_data(ttl=3600, show_spinner="Loading product catalog...")
def load_products() -> pd.DataFrame | None:
    uri = _resolve("PRODUCTS_S3_URI", DEFAULT_PRODUCTS_PATH)
    return pd.read_parquet(uri) if uri else None


def require_data(*dfs: pd.DataFrame | None) -> None:
    if any(d is None for d in dfs):
        st.error(
            "Pre-computed parquet files not found. Run "
            "`python scripts/build_results.py` first, or set "
            "`RESULTS_S3_URI` / `PRODUCTS_S3_URI` env vars."
        )
        st.stop()
