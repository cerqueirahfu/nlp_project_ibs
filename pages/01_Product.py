"""Product drill-down page.

Lets the user pick a product (search by name or browse by category) and shows
full metadata + aspect-level sentiment broken out across discount tiers.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from utils.helpers import (
    ALL,
    apply_brand_css,
    load_products,
    load_results,
    require_data,
)


def _filter_products(products: pd.DataFrame) -> pd.DataFrame:
    mains = sorted(products["main_category"].dropna().unique())
    main = st.selectbox("Category", [ALL] + mains, key="prod_main")
    if main != ALL:
        products = products[products["main_category"] == main]

    sub1s = sorted(products["sub_category_1"].dropna().unique())
    sub1 = st.selectbox("Sub-category", [ALL] + sub1s, key="prod_sub1")
    if sub1 != ALL:
        products = products[products["sub_category_1"] == sub1]

    return products


def main() -> None:
    st.set_page_config(page_title="PricePulse — Product", page_icon="🔍", layout="wide")
    apply_brand_css()

    results = load_results()
    products = load_products()
    require_data(results, products)

    with st.sidebar:
        st.markdown("## PricePulse")
        st.caption("Precision Analytics")
        st.markdown("---")
        st.markdown("**1. Narrow the catalog**")
        scoped = _filter_products(products)

        st.markdown("**2. Search products**")
        query = st.text_input("search", placeholder="type part of a product name…",
                              label_visibility="collapsed")
        if query:
            mask = scoped["product_name"].str.contains(query, case=False, na=False)
            scoped = scoped[mask]

        st.caption(f"{len(scoped):,} products in scope")

    if scoped.empty:
        st.warning("No products match those filters. Try broadening category or clearing the search.")
        return

    options = scoped.sort_values("product_name")
    label_to_id = {
        f"{row.product_name[:90]} — {row.product_id}": row.product_id
        for row in options.itertuples()
    }
    pick = st.selectbox("Pick a product", list(label_to_id.keys()))
    product_id = label_to_id[pick]
    row = products[products["product_id"] == product_id].iloc[0]

    st.title(row["product_name"])
    st.caption(row["category"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Discounted price", f"₹{row['discounted_price']:,.0f}")
    c2.metric("Actual price", f"₹{row['actual_price']:,.0f}",
              f"-{row['discount_percentage']}%", delta_color="inverse")
    c3.metric("Rating", f"{row['rating']:.1f} ★",
              f"{int(row['rating_count']):,} reviews")
    c4.metric("Discount tier", str(row["discount_group"]))

    st.markdown("---")

    product_spans = results[results["product_id"] == product_id]
    if product_spans.empty:
        st.info(
            "No aspect spans were detected for this product. This usually means the "
            "review text was short or didn't trigger any of the tracked aspect keywords."
        )
    else:
        left, right = st.columns(2)

        with left:
            st.subheader("Aspect breakdown")
            by_aspect = (
                product_spans.groupby("aspect", observed=True)[
                    ["positive", "negative", "neutral", "total"]
                ]
                .sum()
                .reset_index()
            )
            by_aspect["positive_share"] = by_aspect["positive"] / by_aspect["total"].clip(lower=1)
            by_aspect["negative_share"] = by_aspect["negative"] / by_aspect["total"].clip(lower=1)
            by_aspect = by_aspect.sort_values("total", ascending=False)
            st.dataframe(
                by_aspect[["aspect", "total", "positive_share", "negative_share"]]
                .rename(columns={"total": "mentions"})
                .style.format({"positive_share": "{:.0%}", "negative_share": "{:.0%}"}),
                hide_index=True,
                use_container_width=True,
            )

        with right:
            st.subheader("Sentiment shape")
            counts = product_spans[["positive", "negative", "neutral"]].sum()
            st.bar_chart(counts.rename("mentions"))

    st.markdown("---")
    st.subheader("Review excerpts")
    if isinstance(row.get("review_title"), str) and row["review_title"].strip():
        st.markdown("**Titles:**")
        st.write(row["review_title"])
    if isinstance(row.get("review_content"), str) and row["review_content"].strip():
        with st.expander("Full review content", expanded=False):
            st.write(row["review_content"])


if __name__ == "__main__":
    main()
