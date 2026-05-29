"""PricePulse Streamlit dashboard.

Reads pre-aggregated results.parquet produced by scripts/build_results.py.
Local file by default; set RESULTS_S3_URI to read from S3 in production
(e.g. RESULTS_S3_URI=s3://price-pulse/results.parquet).

Run locally:
    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from ai.bedrock_client import BedrockError, available
from ai.insights import summarize
from utils.helpers import (
    ALL,
    DISCOUNT_TIERS,
    HIGH_DISCOUNT_TIERS,
    apply_brand_css,
    load_products,
    load_results,
    require_data,
)


def _cascading_filter(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str, str]:
    mains = sorted(results["main_category"].dropna().unique())
    main = st.selectbox("Category", [ALL] + mains, key="dash_main")
    main_filter = mains if main == ALL else [main]
    df = results[results["main_category"].isin(main_filter)]

    sub1s = sorted(df["sub_category_1"].dropna().unique())
    sub1 = st.selectbox("Sub-category", [ALL] + sub1s, key="dash_sub1")
    sub1_filter = sub1s if sub1 == ALL else [sub1]
    df = df[df["sub_category_1"].isin(sub1_filter)]

    sub2s = sorted(df["sub_category_2"].dropna().unique())
    sub2 = st.selectbox("Type", [ALL] + sub2s, key="dash_sub2")
    sub2_filter = sub2s if sub2 == ALL else [sub2]
    df = df[df["sub_category_2"].isin(sub2_filter)]

    return df, main, sub1, sub2


def main() -> None:
    st.set_page_config(page_title="PricePulse", page_icon="📈", layout="wide")
    apply_brand_css()

    results = load_results()
    products = load_products()
    require_data(results, products)

    with st.sidebar:
        st.markdown("## PricePulse")
        st.caption("Precision Analytics")
        st.markdown("---")

        st.markdown("**1. Select category**")
        filtered, main, sub1, sub2 = _cascading_filter(results)

        st.markdown("**2. Discount tiers**")
        selected_tiers = st.multiselect(
            "tiers",
            list(DISCOUNT_TIERS),
            default=list(DISCOUNT_TIERS),
            label_visibility="collapsed",
        )

        st.markdown("---")
        if st.button("Refresh data from storage"):
            load_results.clear()
            load_products.clear()
            st.rerun()

    if selected_tiers:
        filtered = filtered[filtered["discount_group"].astype(str).isin(selected_tiers)]

    if filtered.empty:
        st.warning(
            "No data for the selected filters. Try broadening the category "
            "or selecting more discount tiers."
        )
        return

    overall = filtered[["positive", "negative", "neutral", "total"]].sum()
    total_mentions = int(overall["total"]) or 1
    pos_share = overall["positive"] / total_mentions
    neg_share = overall["negative"] / total_mentions

    by_aspect = (
        filtered.groupby("aspect", observed=True)[
            ["positive", "negative", "neutral", "total"]
        ]
        .sum()
        .reset_index()
    )
    by_aspect["positive_share"] = by_aspect["positive"] / by_aspect["total"].clip(lower=1)
    by_aspect["negative_share"] = by_aspect["negative"] / by_aspect["total"].clip(lower=1)

    top_negative = by_aspect.sort_values("negative_share", ascending=False).iloc[0]
    top_positive = by_aspect.sort_values("positive_share", ascending=False).iloc[0]

    crumbs = " › ".join(c for c in (main, sub1, sub2) if c != ALL) or "All categories"
    st.title("Dashboard")
    st.caption(
        f"**{crumbs}** | Tiers: **{', '.join(selected_tiers) or 'none'}** | "
        f"Aspect mentions: **{total_mentions:,}** | "
        f"Products in scope: **{filtered['product_id'].nunique():,}**"
    )

    st.subheader("Key insights")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Top complaint aspect",
        top_negative["aspect"],
        f"{top_negative['negative_share']*100:.0f}% negative",
        delta_color="inverse",
    )
    c2.metric(
        "Strongest positive aspect",
        top_positive["aspect"],
        f"{top_positive['positive_share']*100:.0f}% positive",
    )
    c3.metric(
        "Overall sentiment",
        f"{pos_share*100:.0f}% positive",
        f"{neg_share*100:.0f}% negative",
        delta_color="inverse",
    )

    scope_label = f"{crumbs} | tiers: {', '.join(selected_tiers) or 'none'}"
    if available():
        if st.button("✨ Generate AI insight summary"):
            with st.spinner("Asking the model…"):
                try:
                    st.markdown(summarize(filtered, scope_label))
                except BedrockError as exc:
                    st.error(str(exc))
    else:
        st.caption(
            "✨ AI insight summary is available once AWS Bedrock access is "
            "configured (see EC2_DEPLOYMENT_GUIDE.md)."
        )

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("Sentiment by discount level")
        by_tier = (
            filtered.groupby("discount_group", observed=True)[
                ["positive", "negative", "neutral", "total"]
            ]
            .sum()
            .reset_index()
        )
        by_tier["Positive"] = (by_tier["positive"] / by_tier["total"].clip(lower=1)) * 100
        by_tier["Negative"] = (by_tier["negative"] / by_tier["total"].clip(lower=1)) * 100
        tier_long = by_tier.melt(
            id_vars="discount_group",
            value_vars=["Positive", "Negative"],
            var_name="Sentiment",
            value_name="Share",
        )
        chart = (
            alt.Chart(tier_long)
            .mark_bar()
            .encode(
                x=alt.X("discount_group:N", title="Discount tier",
                        sort=list(DISCOUNT_TIERS)),
                xOffset="Sentiment:N",
                y=alt.Y("Share:Q", title="Share of mentions (%)"),
                color=alt.Color(
                    "Sentiment:N",
                    scale=alt.Scale(
                        domain=["Positive", "Negative"],
                        range=["#1EBDA4", "#E4572E"],
                    ),
                    title="Sentiment",
                ),
                tooltip=[
                    alt.Tooltip("discount_group:N", title="Discount tier"),
                    alt.Tooltip("Sentiment:N", title="Sentiment"),
                    alt.Tooltip("Share:Q", title="Share of mentions (%)",
                                format=".0f"),
                ],
            )
        )
        st.altair_chart(chart, use_container_width=True)

    with right:
        st.subheader("Aspect-based insights")
        display_aspects = by_aspect.sort_values("total", ascending=False)[
            ["aspect", "total", "positive_share", "negative_share"]
        ].rename(columns={"total": "mentions"})
        st.dataframe(
            display_aspects.style.format(
                {"positive_share": "{:.0%}", "negative_share": "{:.0%}"}
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("---")

    left2, right2 = st.columns(2)

    with left2:
        st.subheader("Top complaints")
        st.caption("Aspects with the most negative mentions in the current filter")
        complaints = (
            filtered.groupby("aspect", observed=True)["negative"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
            .rename(columns={"negative": "negative mentions"})
        )
        st.dataframe(complaints, hide_index=True, use_container_width=True)

    with right2:
        st.subheader("Top discount drivers")
        st.caption(
            "Aspects whose negative share is highest among heavily-discounted "
            f"products ({', '.join(HIGH_DISCOUNT_TIERS)})"
        )
        high_discount = filtered[
            filtered["discount_group"].astype(str).isin(HIGH_DISCOUNT_TIERS)
        ]
        if high_discount.empty:
            st.info("No heavily-discounted products in the current filter.")
        else:
            drivers = (
                high_discount.groupby("aspect", observed=True)[["negative", "total"]]
                .sum()
                .reset_index()
            )
            drivers["negative share"] = drivers["negative"] / drivers["total"].clip(lower=1)
            drivers = drivers.sort_values("negative share", ascending=False).head(5)
            st.dataframe(
                drivers[["aspect", "negative share"]].style.format(
                    {"negative share": "{:.0%}"}
                ),
                hide_index=True,
                use_container_width=True,
            )

    st.markdown("---")
    st.caption(
        "Want a deeper look at a specific product? Open the **Product** page from the "
        "sidebar nav to drill down by name."
    )


if __name__ == "__main__":
    main()
