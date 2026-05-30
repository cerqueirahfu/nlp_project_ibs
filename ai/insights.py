"""Prompt building + AI calls for PricePulse, on top of `ai.llm_client`.

Both the Dashboard summary button and the chat page funnel through here so the
prompts (and the compact, token-cheap data context) live in one place.

We never send raw review text to the model — only the pre-aggregated counts
already computed for the dashboard. That keeps token use (and cost) tiny and
keeps the model grounded in the same numbers the user sees on screen.
"""
from __future__ import annotations

import pandas as pd

from ai.llm_client import complete

SYSTEM = (
    "You are PricePulse, an analytics assistant for an Amazon-electronics review "
    "study. The research question is whether customers complain differently when a "
    "product is heavily discounted. You are given pre-aggregated sentiment counts by "
    "aspect and by discount tier. Answer ONLY from the numbers provided — never invent "
    "products, figures, or reviews. Be concise and concrete; cite the actual percentages."
)


def build_context(filtered: pd.DataFrame, scope_label: str) -> str:
    """Render the filtered aggregates as a compact text block for the model."""
    if filtered.empty:
        return "No data in the current filter."

    by_aspect = (
        filtered.groupby("aspect", observed=True)[["positive", "negative", "neutral", "total"]]
        .sum()
        .reset_index()
        .sort_values("total", ascending=False)
    )
    by_tier = (
        filtered.groupby("discount_group", observed=True)[["positive", "negative", "total"]]
        .sum()
        .reset_index()
    )

    def pct(n: float, d: float) -> str:
        return f"{(n / d * 100):.0f}%" if d else "n/a"

    aspect_lines = [
        f"- {r.aspect}: {int(r.total)} mentions, "
        f"{pct(r.positive, r.total)} positive, {pct(r.negative, r.total)} negative"
        for r in by_aspect.itertuples()
    ]
    tier_lines = [
        f"- {r.discount_group}: {int(r.total)} mentions, "
        f"{pct(r.positive, r.total)} positive, {pct(r.negative, r.total)} negative"
        for r in by_tier.itertuples()
    ]

    total = int(filtered["total"].sum())
    products = int(filtered["product_id"].nunique())
    return (
        f"Scope: {scope_label}\n"
        f"Products in scope: {products}; total aspect mentions: {total}\n\n"
        "Sentiment by aspect:\n" + "\n".join(aspect_lines) + "\n\n"
        "Sentiment by discount tier:\n" + "\n".join(tier_lines)
    )


def summarize(filtered: pd.DataFrame, scope_label: str) -> str:
    """Produce a short executive summary of the current filter."""
    context = build_context(filtered, scope_label)
    prompt = (
        f"{context}\n\n"
        "Write a brief executive summary for this selection as exactly three "
        "markdown bullets:\n"
        "1. The biggest complaint (most-negative aspect) and its negative %.\n"
        "2. The strongest positive aspect and its positive %.\n"
        "3. One takeaway on whether higher discount tiers show more negativity "
        "than lower ones, referencing the tier numbers."
    )
    return complete(
        [{"role": "user", "content": prompt}],
        system=SYSTEM,
        max_tokens=350,
    )


REVIEW_SYSTEM = (
    "You summarize Amazon customer reviews for a single product. The input is "
    "several reviews concatenated together and may contain stray commas, image "
    "URLs, or sentence fragments — ignore that noise. Summarize ONLY what "
    "reviewers actually say; never invent features, praise, or complaints."
)


def summarize_reviews(product_name: str, titles: str, content: str) -> str:
    """Turn the raw concatenated review blob into clean Pros / Cons / Takeaway.

    The blob is capped to keep token use bounded; the model is told to ignore
    the URL/comma noise inherent in this dataset.
    """
    blob = f"Review titles: {titles}\n\nReview text: {content}".strip()[:6000]
    prompt = (
        f"Product: {product_name}\n\n{blob}\n\n"
        "Summarize the customer reviews as markdown with exactly these parts:\n"
        "**👍 Pros** — up to 4 short bullets of what customers liked.\n"
        "**👎 Cons** — up to 4 short bullets of complaints (write "
        "'No clear complaints mentioned.' if there are none).\n"
        "**Takeaway** — one sentence overall verdict."
    )
    return complete(
        [{"role": "user", "content": prompt}],
        system=REVIEW_SYSTEM,
        max_tokens=400,
    )


def chat(history: list[dict], filtered: pd.DataFrame, scope_label: str) -> str:
    """Answer a chat turn grounded in the current filter's aggregates.

    `history` is the running [{"role", "content"}] list; the latest user turn is
    already its last element. The data context is injected via the system prompt.
    """
    context = build_context(filtered, scope_label)
    system = f"{SYSTEM}\n\nData for the current selection:\n{context}"
    return complete(history, system=system, max_tokens=500)
