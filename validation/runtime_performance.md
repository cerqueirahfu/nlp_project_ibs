# Runtime Performance — Latency & Cost

**What "the pipeline" means here.** The heavy SetFitABSA inference runs **offline**
on the laptop and is baked into `results.parquet` — it never runs per user query.
What a user actually waits for at runtime is:

```
[results.parquet cached in RAM] -> filter rows -> build compact text context
    -> OpenAI gpt-4o-mini call -> render answer
```

So latency and cost are measured on the three real OpenAI-backed query types the
dashboard exposes: **dashboard summary**, **product Pros/Cons**, and **dataset chat**.

**Method.** Local stages (filter + context/blob build) are *measured* over 50
iterations each. Input tokens are *counted exactly* with `tiktoken` (`o200k_base`)
from the real prompts in [ai/insights.py](../ai/insights.py) using real data from
`results.parquet` / `products.parquet`. The OpenAI network call was **not** made
(no spend); its latency is *modeled* from gpt-4o-mini's typical behaviour
(first-token ≈ 0.5 s + output at ≈ 80 tok/s) and labeled as such. Cost uses
published gpt-4o-mini pricing: **$0.15 / 1M input, $0.60 / 1M output tokens**.
Reproduce with `nlpenv/Scripts/python.exe validation/measure_runtime.py`.

## Per-query-type breakdown

| Query type | Local prep (mean / max) | In tok | Out tok | E2E latency (typ / worst) | Cost/query |
|---|---:|---:|---:|---:|---:|
| Dashboard summary | 3.5 / 6.0 ms | 373 | 220 | 3.25 s / 4.88 s | $0.00019 |
| Product Pros/Cons (median review) | <1 ms | 414 | 260 | 3.75 s / 5.50 s | $0.00022 |
| Product Pros/Cons (longest review) | <1 ms | 1 554 | 260 | 3.75 s / 5.50 s | $0.00039 |
| Dataset chat (1 turn) | 3.3 / 5.1 ms | 333 | 200 | 3.00 s / 6.76 s | $0.00017 |

Worst-case columns use the `max_tokens` caps from `ai/insights.py` (350/400/500)
as the largest completion the model can return.

## Deliverable — latency & cost at three usage scales

Blended over an equal mix of the three query types: **~$0.00022 / query**. Latency
is **flat across scales** — each query is independent and the work is I/O-wait on
the OpenAI API, not local compute, so volume doesn't change per-query time (it only
adds queueing under heavy concurrency, irrelevant at these volumes).

| Scale | Avg. latency | Worst case | Cost/query | Est. monthly cost |
|---|---:|---:|---:|---:|
| 100 queries/mo | 3.4 s | 6.8 s | $0.00022 | **$0.02** |
| 1 000 queries/mo | 3.4 s | 6.8 s | $0.00022 | **$0.22** |
| 10 000 queries/mo | 3.4 s | 6.8 s | $0.00022 | **$2.21** |

Compute is **$0 on top of this**: EC2 `t2.micro` and S3 both sit in the AWS free
tier (12 months / 5 GB), and there is no per-query model inference on the server.
At 10 000 queries/month the entire bill is ~$2.21 of OpenAI credit.

## Bottleneck

**The OpenAI gpt-4o-mini call dominates: ≈ 99.8% of end-to-end latency.** Every
local stage combined is single-digit milliseconds — filtering and context-building
run in 3–7 ms, and the `results.parquet` read (3 ms locally; +network on a cold
S3 fetch) is cached for an hour and amortizes to ~0 per query. Worst-case latency
(~6.8 s) comes from the **dataset chat**, whose 500-token cap is the longest the
model may decode; the longest input is the product Pros/Cons path, but its 6 000-char
blob cap keeps input bounded at ~1 554 tokens, so cost never runs away.

**Takeaway:** the product is comfortably viable — ~3.4 s typical latency and a
fraction of a cent per query. The only knob that moves both latency and cost is the
`max_tokens` cap; lowering the chat cap from 500 would shave the worst case, and
streaming the response would cut *perceived* latency since first tokens arrive in
~0.5 s.
