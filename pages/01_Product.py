"""Product drill-down page.

Lets the user pick a product (search by name or browse by category) and shows
full metadata + aspect-level sentiment broken out across discount tiers.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from ai.llm_client import LLMError, available
from ai.insights import summarize_reviews
from utils.helpers import (
    ALL,
    apply_brand_css,
    load_products,
    load_results,
    require_data,
)

_URL_RE = re.compile(r"https?://\S+")


def _strip_urls(text: str) -> str:
    return _URL_RE.sub("", str(text)).strip()


def _split_titles(text: str) -> list[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


TITLE_LIMIT = 70


def _shorten(name: str, limit: int = TITLE_LIMIT) -> str:
    """Trim a long product name to `limit` chars on a word boundary, add '…'."""
    name = str(name).strip()
    if len(name) <= limit:
        return name
    return name[:limit].rsplit(" ", 1)[0] + "…"


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

    full_name = str(row["product_name"]).strip()
    st.subheader(_shorten(full_name))
    if len(full_name) > TITLE_LIMIT:
        st.caption(full_name)
    crumbs = " › ".join(p for p in str(row["category"]).split("|") if p)
    st.caption(crumbs)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Discounted price", f"{row['discounted_price']:,.0f}",
              f"-{row['discount_percentage']}%", delta_color="inverse")
    c2.metric("Actual price", f"{row['actual_price']:,.0f}")
    c3.metric("Rating", f"{row['rating']:.1f} ★",
              f"{int(row['rating_count']):,} ratings")
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
            shape_df = pd.DataFrame({
                "Sentiment": ["Positive", "Negative", "Neutral"],
                "Mentions": [
                    counts["positive"], counts["negative"], counts["neutral"],
                ],
            })
            chart = (
                alt.Chart(shape_df)
                .mark_bar()
                .encode(
                    x=alt.X("Sentiment:N", title="Sentiment",
                            sort=["Positive", "Negative", "Neutral"]),
                    y=alt.Y("Mentions:Q", title="Mentions"),
                    color=alt.Color(
                        "Sentiment:N",
                        scale=alt.Scale(
                            domain=["Positive", "Negative", "Neutral"],
                            range=["#1EBDA4", "#E4572E", "#94A3B8"],
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("Sentiment:N", title="Sentiment"),
                        alt.Tooltip("Mentions:Q", title="Mentions",
                                    format=".0f"),
                    ],
                )
            )
            st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    st.subheader("What customers say")

    titles = row.get("review_title")
    content = row.get("review_content")
    has_titles = isinstance(titles, str) and titles.strip()
    has_content = isinstance(content, str) and content.strip()

    if not (has_titles or has_content):
        st.caption("No review text available for this product.")
        return

    if available():
        if st.button("✨ Summarize customer reviews"):
            with st.spinner("Reading the reviews…"):
                try:
                    st.markdown(
                        summarize_reviews(full_name, titles or "", content or "")
                    )
                except LLMError as exc:
                    st.error(str(exc))
    else:
        st.caption(
            "✨ AI review summary (Pros / Cons) appears here once an OpenAI API "
            "key is configured (set OPENAI_API_KEY). Raw reviews below."
        )

    with st.expander("See raw customer reviews", expanded=not available()):
        if has_titles:
            st.markdown("**Highlights**")
            for title in _split_titles(titles):
                st.markdown(f"- {title}")
        if has_content:
            st.markdown("**Full review text**")
            st.write(_strip_urls(content))


if __name__ == "__main__":
    main()
