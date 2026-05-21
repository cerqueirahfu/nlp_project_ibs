import pandas as pd

ASPECT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "battery": ("battery", "charge", "charging", "power", "backup"),
    "quality": ("quality", "build", "material", "cheap", "premium"),
    "performance": ("performance", "speed", "fast", "slow", "lag", "smooth"),
    "durability": ("durable", "durability", "broke", "broken",
                   "stopped working", "lasted"),
    "delivery": ("delivery", "shipping", "arrived", "package", "packaging"),
    "value": ("price", "worth", "value", "money", "cost",
              "expensive", "cheap"),
    "sound": ("sound", "bass", "volume", "audio", "noise"),
}

POLARITIES = ("positive", "negative", "neutral")


def map_span_to_aspect(span: str) -> str | None:
    s = span.lower()
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(k in s for k in keywords):
            return aspect
    return None


def attach_predictions(
    sentences_df: pd.DataFrame,
    predictions: list[list[dict]],
) -> pd.DataFrame:
    rows: list[dict] = []
    for (_, srow), spans in zip(sentences_df.iterrows(), predictions):
        for span in spans:
            aspect = map_span_to_aspect(span.get("span", ""))
            if aspect is None:
                continue
            rows.append({
                **srow.to_dict(),
                "span": span["span"],
                "polarity": span["polarity"],
                "aspect": aspect,
            })
    return pd.DataFrame(rows)


def aspect_sentiment_by_group(
    spans_df: pd.DataFrame,
    group_col: str = "discount_group",
) -> pd.DataFrame:
    if spans_df.empty:
        return spans_df
    counts = (
        spans_df.groupby(["aspect", group_col, "polarity"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in POLARITIES:
        if col not in counts.columns:
            counts[col] = 0
    counts["total"] = counts[list(POLARITIES)].sum(axis=1)
    counts["positive_share"] = counts["positive"] / counts["total"]
    counts["negative_share"] = counts["negative"] / counts["total"]
    return counts


def top_complaints(spans_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if spans_df.empty:
        return spans_df
    negatives = spans_df[spans_df["polarity"] == "negative"]
    return (
        negatives.groupby("aspect")
        .size()
        .sort_values(ascending=False)
        .head(n)
        .reset_index(name="count")
    )
