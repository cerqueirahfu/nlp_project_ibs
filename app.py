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

from ai.llm_client import LLMError, available
from ai.insights import summarize
from utils.helpers import (
    ALL,
    ASPECT_HELP,
    DISCOUNT_TIERS,
    HIGH_DISCOUNT_TIERS,
    LOW_DISCOUNT_TIERS,
    TAGLINE,
    apply_brand_css,
    load_products,
    load_results,
    require_data,
    tier_group_summary,
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

    # Category-filtered data BEFORE the tier filter — the discount comparison
    # always spans low vs high tiers regardless of the tier multiselect (Finding 2).
    cat_filtered = filtered

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
    pos_n, neu_n, neg_n = (int(overall[c]) for c in ("positive", "neutral", "negative"))
    pos_share = pos_n / total_mentions
    neu_share = neu_n / total_mentions
    neg_share = neg_n / total_mentions

    by_aspect = (
        filtered.groupby("aspect", observed=True)[
            ["positive", "negative", "neutral", "total"]
        ]
        .sum()
        .reset_index()
    )
    by_aspect["positive_share"] = by_aspect["positive"] / by_aspect["total"].clip(lower=1)
    by_aspect["neutral_share"] = by_aspect["neutral"] / by_aspect["total"].clip(lower=1)
    by_aspect["negative_share"] = by_aspect["negative"] / by_aspect["total"].clip(lower=1)

    top_negative = by_aspect.sort_values("negative_share", ascending=False).iloc[0]
    top_positive = by_aspect.sort_values("positive_share", ascending=False).iloc[0]

    crumbs = " › ".join(c for c in (main, sub1, sub2) if c != ALL) or "All categories"

    # Finding 1 — make the product's purpose obvious before anything else.
    st.title("📈 PricePulse")
    st.markdown(f"#### {TAGLINE}")
    st.caption(
        "Each number below counts review *sentences* that mention a product aspect "
        "(battery, quality, value, …), classified as positive, neutral, or negative."
    )
    with st.expander("ℹ️ What does “aspect-based sentiment” mean?"):
        st.markdown(ASPECT_HELP)

    st.divider()
    st.caption(
        f"**{crumbs}** | Tiers: **{', '.join(selected_tiers) or 'none'}** | "
        f"Aspect mentions: **{total_mentions:,}** | "
        f"Products in scope: **{filtered['product_id'].nunique():,}**"
    )

    st.subheader(
        "Key insights",
        help="Top complaint / strongest positive are the aspects with the highest "
             "negative / positive share of their own mentions. Overall sentiment is "
             "across all aspect mentions in the current filter.",
    )
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Top complaint aspect",
        top_negative["aspect"],
        f"{top_negative['negative_share']*100:.0f}% negative "
        f"({int(top_negative['negative']):,} of {int(top_negative['total']):,})",
        delta_color="inverse",
        help="The aspect with the highest share of negative mentions.",
    )
    c2.metric(
        "Strongest positive aspect",
        top_positive["aspect"],
        f"{top_positive['positive_share']*100:.0f}% positive "
        f"({int(top_positive['positive']):,} of {int(top_positive['total']):,})",
        help="The aspect with the highest share of positive mentions.",
    )
    c3.metric(
        "Total aspect mentions",
        f"{total_mentions:,}",
        f"across {filtered['product_id'].nunique():,} products",
        delta_color="off",
        help="Review sentences mentioning a tracked aspect, in the current filter.",
    )

    # Finding 3 — show positive + neutral + negative together (they sum to 100%),
    # each with both its absolute count and its percentage.
    st.markdown(
        f"**Overall sentiment** &nbsp; "
        f"🟢 {pos_share*100:.0f}% positive ({pos_n:,}) &nbsp;·&nbsp; "
        f"⚪ {neu_share*100:.0f}% neutral ({neu_n:,}) &nbsp;·&nbsp; "
        f"🔴 {neg_share*100:.0f}% negative ({neg_n:,})",
        help="Positive + neutral + negative always sum to 100% of the "
             f"{total_mentions:,} aspect mentions in this filter.",
    )
    sentiment_split = pd.DataFrame(
        {"Sentiment": ["Positive", "Neutral", "Negative"],
         "Share": [pos_share * 100, neu_share * 100, neg_share * 100],
         "Count": [pos_n, neu_n, neg_n]}
    )
    split_chart = (
        alt.Chart(sentiment_split)
        .mark_bar()
        .encode(
            x=alt.X("Share:Q", title=None, scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "Sentiment:N",
                scale=alt.Scale(
                    domain=["Positive", "Neutral", "Negative"],
                    range=["#1EBDA4", "#C9C2B6", "#E4572E"],
                ),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            order=alt.Order("Sentiment:N"),
            tooltip=[
                alt.Tooltip("Sentiment:N"),
                alt.Tooltip("Share:Q", title="Share (%)", format=".0f"),
                alt.Tooltip("Count:Q", title="Mentions", format=","),
            ],
        )
        .properties(height=70)
    )
    st.altair_chart(split_chart, use_container_width=True)

    scope_label = f"{crumbs} | tiers: {', '.join(selected_tiers) or 'none'}"
    if available():
        if st.button("✨ Generate AI insight summary"):
            with st.spinner("Asking the model…"):
                try:
                    st.markdown(summarize(filtered, scope_label))
                except LLMError as exc:
                    st.error(str(exc))
    else:
        st.caption(
            "✨ AI insight summary is available once an OpenAI API key is "
            "configured (set the OPENAI_API_KEY environment variable)."
        )

    st.markdown("---")

    # Finding 2 — side-by-side low vs high discount comparison so users don't have
    # to switch filters and hold one result in memory. Spans all tiers in the
    # category regardless of the tier multiselect.
    st.subheader(
        "Low vs high discount — side by side",
        help="Compares low-discount tiers (0–30%) against high-discount tiers "
             "(30%+) for the selected category. This view ignores the discount-tier "
             "filter on the left so the two sides are always comparable.",
    )
    st.caption(
        f"Low = {', '.join(LOW_DISCOUNT_TIERS)} &nbsp;·&nbsp; "
        f"High = {', '.join(HIGH_DISCOUNT_TIERS)} &nbsp;·&nbsp; **{crumbs}**"
    )
    low = tier_group_summary(cat_filtered, LOW_DISCOUNT_TIERS)
    high = tier_group_summary(cat_filtered, HIGH_DISCOUNT_TIERS)

    def _render_band(col, label: str, s: dict | None) -> None:
        with col:
            st.markdown(f"##### {label} discount")
            if s is None:
                st.info("No data in these tiers for the current category.")
                return
            st.markdown(
                f"🟢 **{s['pos_share']*100:.0f}%** positive ({s['pos']:,}) &nbsp;·&nbsp; "
                f"⚪ **{s['neu_share']*100:.0f}%** neutral ({s['neu']:,}) &nbsp;·&nbsp; "
                f"🔴 **{s['neg_share']*100:.0f}%** negative ({s['neg']:,})"
            )
            st.metric("Top negative aspect", s["top_neg_aspect"],
                      f"{s['top_neg_aspect_share']*100:.0f}% negative",
                      delta_color="inverse")
            st.caption(f"**{s['total']:,}** aspect mentions · **{s['products']:,}** products")

    cl, ch = st.columns(2)
    _render_band(cl, "Low", low)
    _render_band(ch, "High", high)

    if low and high:
        gap = (high["neg_share"] - low["neg_share"]) * 100
        if gap > 1:
            st.success(f"📉 Heavily-discounted products show **{gap:.0f} pts more negative** "
                       f"sentiment ({high['neg_share']*100:.0f}% vs {low['neg_share']*100:.0f}%).")
        elif gap < -1:
            st.success(f"📈 Heavily-discounted products show **{abs(gap):.0f} pts less negative** "
                       f"sentiment ({high['neg_share']*100:.0f}% vs {low['neg_share']*100:.0f}%).")
        else:
            st.info(f"Negative sentiment is about the same across discount levels "
                    f"({high['neg_share']*100:.0f}% vs {low['neg_share']*100:.0f}%).")

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
        for label, col in (("Positive", "positive"), ("Neutral", "neutral"),
                           ("Negative", "negative")):
            by_tier[label] = (by_tier[col] / by_tier["total"].clip(lower=1)) * 100
        tier_long = by_tier.melt(
            id_vars=["discount_group", "total"],
            value_vars=["Positive", "Neutral", "Negative"],
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
                        domain=["Positive", "Neutral", "Negative"],
                        range=["#1EBDA4", "#C9C2B6", "#E4572E"],
                    ),
                    title="Sentiment",
                ),
                tooltip=[
                    alt.Tooltip("discount_group:N", title="Discount tier"),
                    alt.Tooltip("Sentiment:N", title="Sentiment"),
                    alt.Tooltip("Share:Q", title="Share of mentions (%)",
                                format=".0f"),
                    alt.Tooltip("total:Q", title="Total mentions", format=","),
                ],
            )
        )
        st.altair_chart(chart, use_container_width=True)

    with right:
        st.subheader(
            "Aspect-based insights",
            help="Per aspect: total mentions, then the share that are positive / "
                 "neutral / negative (each row's three shares sum to 100%).",
        )
        display_aspects = by_aspect.sort_values("total", ascending=False)[
            ["aspect", "total", "positive_share", "neutral_share", "negative_share"]
        ].rename(columns={
            "total": "mentions", "positive_share": "positive",
            "neutral_share": "neutral", "negative_share": "negative",
        })
        st.dataframe(
            display_aspects.style.format(
                {"mentions": "{:,}", "positive": "{:.0%}",
                 "neutral": "{:.0%}", "negative": "{:.0%}"}
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("---")

    left2, right2 = st.columns(2)

    with left2:
        st.subheader(
            "Top complaints",
            help="Aspects ranked by their number of negative mentions; the % is that "
                 "aspect's negative share of its own mentions.",
        )
        st.caption("Aspects with the most negative mentions in the current filter")
        complaints = (
            filtered.groupby("aspect", observed=True)[["negative", "total"]]
            .sum()
            .reset_index()
        )
        complaints["negative share"] = complaints["negative"] / complaints["total"].clip(lower=1)
        complaints = (
            complaints.sort_values("negative", ascending=False)
            .head(5)[["aspect", "negative", "negative share"]]
            .rename(columns={"negative": "negative mentions"})
        )
        st.dataframe(
            complaints.style.format(
                {"negative mentions": "{:,}", "negative share": "{:.0%}"}
            ),
            hide_index=True, use_container_width=True,
        )

    with right2:
        st.subheader(
            "Top discount drivers",
            help="Among heavily-discounted products only, the aspects with the "
                 "highest negative share — i.e. what people complain about most when "
                 "the discount is large. Count shown alongside the %.",
        )
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
            drivers = drivers.rename(columns={"negative": "negative mentions"})
            st.dataframe(
                drivers[["aspect", "negative mentions", "negative share"]].style.format(
                    {"negative mentions": "{:,}", "negative share": "{:.0%}"}
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
