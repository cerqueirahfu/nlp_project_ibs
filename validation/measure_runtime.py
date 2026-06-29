"""Runtime latency + cost measurement for the PricePulse user-facing pipeline.

The user-facing pipeline at runtime is NOT the SetFitABSA model (that runs
offline on the laptop and is baked into results.parquet). What a user waits for
per query is:

    [cached parquet in memory] -> filter rows -> build compact text context
        -> OpenAI gpt-4o-mini call -> render

We measure the three real query types the dashboard exposes:
    1. summary  -> ai.insights.summarize()        (dashboard exec summary)
    2. product  -> ai.insights.summarize_reviews() (product Pros/Cons)
    3. chat     -> ai.insights.chat()              (dataset chat turn)

Local stages (filter + context/blob build) are TIMED here over many iterations.
Input tokens are COUNTED exactly with tiktoken. The OpenAI network call is not
made (no spend); its latency is MODELED from gpt-4o-mini's typical behaviour
(see LLM_TTFT_S / LLM_TOK_PER_S) and clearly labelled as an estimate.

Run from the project root:
    nlpenv/Scripts/python.exe validation/measure_runtime.py
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import tiktoken

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.insights import (  # noqa: E402
    REVIEW_SYSTEM,
    SYSTEM,
    build_context,
)

# --- gpt-4o-mini constants -------------------------------------------------
MODEL = "gpt-4o-mini"
PRICE_IN_PER_1M = 0.15   # USD per 1M input tokens  (published gpt-4o-mini)
PRICE_OUT_PER_1M = 0.60  # USD per 1M output tokens
ENC = tiktoken.get_encoding("o200k_base")  # gpt-4o-mini tokenizer

# Modeled API latency (NOT measured here): first-token latency + decode rate.
# Conservative mid-range figures for gpt-4o-mini.
LLM_TTFT_S = 0.5
LLM_TOK_PER_S = 80.0

N_ITERS = 50  # local-stage timing repetitions

# Realistic completion sizes vs. the max_tokens caps in ai/insights.py.
# (typical_out, cap_out) per query type.
OUT_TOKENS = {
    "summary": (220, 350),
    "product": (260, 400),
    "chat": (200, 500),
}


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


def time_stage(fn, *args, n=N_ITERS) -> tuple[float, float]:
    """Return (mean_ms, max_ms) for fn over n runs."""
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*args)
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(samples), max(samples)


# --- prompt builders mirroring ai/insights.py exactly ----------------------
def summary_prompt(ctx: str) -> str:
    return (
        f"{ctx}\n\nWrite a brief executive summary for this selection as exactly three "
        "markdown bullets:\n1. The biggest complaint (most-negative aspect) and its negative %.\n"
        "2. The strongest positive aspect and its positive %.\n"
        "3. One takeaway on whether higher discount tiers show more negativity "
        "than lower ones, referencing the tier numbers."
    )


def product_prompt(name: str, titles: str, content: str) -> str:
    blob = f"Review titles: {titles}\n\nReview text: {content}".strip()[:6000]
    return (
        f"Product: {name}\n\n{blob}\n\n"
        "Summarize the customer reviews as markdown with exactly these parts:\n"
        "**👍 Pros** — up to 4 short bullets of what customers liked.\n"
        "**👎 Cons** — up to 4 short bullets of complaints (write "
        "'No clear complaints mentioned.' if there are none).\n"
        "**Takeaway** — one sentence overall verdict."
    )


def model_llm_latency(in_tok: int, out_tok: int) -> float:
    """Seconds: first-token latency + decode time. Input cost is in TTFT."""
    return LLM_TTFT_S + out_tok / LLM_TOK_PER_S


def usd(in_tok: int, out_tok: int) -> float:
    return in_tok / 1e6 * PRICE_IN_PER_1M + out_tok / 1e6 * PRICE_OUT_PER_1M


def main() -> None:
    results = pd.read_parquet(ROOT / "results.parquet")
    products = pd.read_parquet(ROOT / "products.parquet")

    # Parquet load (cached once/hour via st.cache_data ttl=3600; not per-query).
    t0 = time.perf_counter()
    pd.read_parquet(ROOT / "results.parquet")
    parquet_ms = (time.perf_counter() - t0) * 1000

    # Representative selections -------------------------------------------
    # summary/chat: filter to the largest main_category (typical dashboard view).
    top_cat = results.groupby("main_category")["total"].sum().idxmax()
    filt = results[results["main_category"] == top_cat]
    scope = f"Category: {top_cat}"

    # product: pick a long-review product (worst case) + a median one.
    products = products.copy()
    products["_len"] = products["review_content"].fillna("").str.len()
    long_row = products.loc[products["_len"].idxmax()]
    med_len = products["_len"].median()
    med_row = products.iloc[(products["_len"] - med_len).abs().argsort().iloc[0]]

    print(f"Parquet load (once/hour, not per-query): {parquet_ms:.0f} ms (local disk; "
          "S3 adds network on cold start)\n")
    print(f"summary/chat scope: {scope}  ({len(filt)} aspect rows)\n")

    rows = []

    # ---- 1. summary ----
    _, ctx_max = None, None
    ctx_mean_ms, ctx_max_ms = time_stage(build_context, filt, scope)
    ctx = build_context(filt, scope)
    prompt = summary_prompt(ctx)
    in_tok = n_tokens(SYSTEM) + n_tokens(prompt)
    out_typ, out_cap = OUT_TOKENS["summary"]
    rows.append(("summary (dashboard)", ctx_mean_ms, ctx_max_ms, in_tok, out_typ, out_cap))

    # ---- 2. product (median + worst-case long review) ----
    for tag, row in (("product (median review)", med_row), ("product (longest review)", long_row)):
        name = str(row["product_name"])
        titles = str(row.get("review_title") or "")
        content = str(row.get("review_content") or "")
        mean_ms, max_ms = time_stage(product_prompt, name, titles, content)
        prompt = product_prompt(name, titles, content)
        in_tok = n_tokens(REVIEW_SYSTEM) + n_tokens(prompt)
        out_typ, out_cap = OUT_TOKENS["product"]
        rows.append((tag, mean_ms, max_ms, in_tok, out_typ, out_cap))

    # ---- 3. chat ----
    user_turn = "Which aspect has the most complaints, and does it get worse at higher discounts?"
    ctx_mean_ms, ctx_max_ms = time_stage(build_context, filt, scope)
    ctx = build_context(filt, scope)
    system = f"{SYSTEM}\n\nData for the current selection:\n{ctx}"
    in_tok = n_tokens(system) + n_tokens(user_turn)
    out_typ, out_cap = OUT_TOKENS["chat"]
    rows.append(("chat (1 turn)", ctx_mean_ms, ctx_max_ms, in_tok, out_typ, out_cap))

    # ---- report ----
    hdr = f"{'Query type':<26}{'prep ms':>9}{'prep max':>10}{'in tok':>8}{'out tok':>8}" \
          f"{'E2E s (typ)':>13}{'E2E s (worst)':>15}{'cost/query $':>14}"
    print(hdr)
    print("-" * len(hdr))
    cost_samples = []
    for tag, mean_ms, max_ms, in_tok, out_typ, out_cap in rows:
        e2e_typ = mean_ms / 1000 + model_llm_latency(in_tok, out_typ)
        e2e_worst = max_ms / 1000 + model_llm_latency(in_tok, out_cap)
        cost = usd(in_tok, out_typ)
        cost_samples.append(cost)
        print(f"{tag:<26}{mean_ms:>9.1f}{max_ms:>10.1f}{in_tok:>8}{out_typ:>8}"
              f"{e2e_typ:>13.2f}{e2e_worst:>15.2f}{cost:>14.5f}")

    # Blended average query (equal mix of the three real query types).
    blended = statistics.mean([cost_samples[0], statistics.mean(cost_samples[1:3]), cost_samples[3]])
    print("\n--- Cost projection (blended avg query) ---")
    print(f"Blended cost/query: ${blended:.5f}")
    for scale in (100, 1000, 10000):
        print(f"  {scale:>6} queries/mo -> ${blended * scale:.2f}/mo")

    print("\nLLM latency is MODELED: TTFT="
          f"{LLM_TTFT_S}s + output/{LLM_TOK_PER_S:.0f} tok per s. "
          "Local prep stages are measured.")


if __name__ == "__main__":
    main()
