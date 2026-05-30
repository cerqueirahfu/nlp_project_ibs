"""AI chat page — ask questions about the current selection's sentiment data.

The model is grounded on the same pre-aggregated counts the dashboard shows
(via ai.insights.build_context), so answers stay consistent with the charts and
never depend on raw review text.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from ai.llm_client import LLMError, available
from ai.insights import chat
from utils.helpers import (
    ALL,
    DISCOUNT_TIERS,
    apply_brand_css,
    load_products,
    load_results,
    require_data,
)


def _cascading_filter(results: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    mains = sorted(results["main_category"].dropna().unique())
    main = st.selectbox("Category", [ALL] + mains, key="chat_main")
    df = results if main == ALL else results[results["main_category"] == main]

    sub1s = sorted(df["sub_category_1"].dropna().unique())
    sub1 = st.selectbox("Sub-category", [ALL] + sub1s, key="chat_sub1")
    if sub1 != ALL:
        df = df[df["sub_category_1"] == sub1]

    crumbs = " › ".join(c for c in (main, sub1) if c != ALL) or "All categories"
    return df, crumbs


def main() -> None:
    st.set_page_config(page_title="PricePulse — AI Chat", page_icon="💬", layout="wide")
    apply_brand_css()

    results = load_results()
    products = load_products()
    require_data(results, products)

    with st.sidebar:
        st.markdown("## PricePulse")
        st.caption("Precision Analytics")
        st.markdown("---")
        st.markdown("**1. Scope the data**")
        filtered, crumbs = _cascading_filter(results)

        st.markdown("**2. Discount tiers**")
        tiers = st.multiselect(
            "tiers", list(DISCOUNT_TIERS), default=list(DISCOUNT_TIERS),
            label_visibility="collapsed",
        )
        if tiers:
            filtered = filtered[filtered["discount_group"].astype(str).isin(tiers)]

        st.markdown("---")
        if st.button("Clear chat"):
            st.session_state.pop("chat_history", None)
            st.rerun()

    st.title("Ask the data 💬")
    scope_label = f"{crumbs} | tiers: {', '.join(tiers) or 'none'}"
    st.caption(
        f"Grounded on: **{scope_label}** — "
        f"{filtered['product_id'].nunique():,} products, "
        f"{int(filtered['total'].sum()):,} aspect mentions."
    )

    if not available():
        st.info(
            "AI chat needs an OpenAI API key. Set the `OPENAI_API_KEY` "
            "environment variable (locally or on EC2) and `pip install openai`, "
            "then reload."
        )
        return

    if filtered.empty:
        st.warning("No data in this scope. Broaden the category or tiers in the sidebar.")
        return

    history: list[dict] = st.session_state.setdefault("chat_history", [])
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    if prompt := st.chat_input("e.g. Which aspect gets worse at high discounts?"):
        history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer = chat(history, filtered, scope_label)
                    st.markdown(answer)
                    history.append({"role": "assistant", "content": answer})
                except LLMError as exc:
                    st.error(str(exc))
                    history.pop()  # drop the unanswered user turn


if __name__ == "__main__":
    main()
