import pandas as pd

CARRY_COLS = (
    "product_id",
    "product_name",
    "main_category",
    "discount_percentage",
    "discount_group",
    "rating",
)


def explode_sentences(
    df: pd.DataFrame,
    sentences_col: str = "review_sentences",
) -> pd.DataFrame:
    keep = [c for c in CARRY_COLS if c in df.columns]
    out = (
        df[keep + [sentences_col]]
        .explode(sentences_col)
        .dropna(subset=[sentences_col])
        .rename(columns={sentences_col: "sentence"})
        .reset_index(drop=True)
    )
    out["sentence"] = out["sentence"].astype(str).str.strip()
    out = out[out["sentence"].str.len() > 0].reset_index(drop=True)
    out["sentence_idx"] = out.groupby("product_id").cumcount()
    return out
