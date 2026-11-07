# Search benchmark: Qwen/Qwen3-Embedding-0.6B

- Machine: AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD, Windows-11-10.0.26300-SP0, torch threads=12, device=cpu
- Index: `cache/qwen3`: 640 patents, 11,218 chunks, dim 1024, mode `hybrid`
- One-time costs: index load 559 ms, model load 18923 ms

## A. Real index: hybrid filters, pre- vs post-filtering

Query embedding (Qwen/Qwen3-Embedding-0.6B, cpu): median **129.482 ms**, max 215.2 ms per query (same for every scenario, so excluded below).

| scenario | strategy | candidates | filter | dense | bm25 | fuse | explain | **search total** |
|---|---|---|---|---|---|---|---|---|
| no filters | pre | 640 | 0.004 | 0.319 | 1.136 | 0.233 | 0.954 | **2.784** |
| cpc=B60B (wheels) | pre | 298 | 0.037 | 0.593 | 0.744 | 0.11 | 0.871 | **2.431** |
| cpc=B60B (wheels) | post | 640 | 0.003 | 0.342 | 1.145 | 0.325 | 0.734 | **2.604** |
| cpc=B60C (tyres) | pre | 318 | 0.049 | 0.579 | 0.716 | 0.11 | 0.726 | **2.212** |
| cpc=B60C (tyres) | post | 640 | 0.004 | 0.386 | 1.379 | 0.399 | 0.823 | **3.027** |
| cpc=B60C 23/ (pressure) | pre | 28 | 0.022 | 0.011 | 0.141 | 0.033 | 0.64 | **0.857** |
| cpc=B60C 23/ (pressure) | post | 640 | 0.003 | 0.416 | 1.453 | 0.401 | 0.829 | **3.163** |
| kw=sensor (title/abstract) | pre | 29 | 0.046 | 0.014 | 0.154 | 0.042 | 0.793 | **1.107** |
| kw=sensor (title/abstract) | post | 640 | 0.003 | 0.289 | 0.984 | 0.311 | 0.788 | **2.433** |
| cpc=B60C + kw=tread | pre | 150 | 0.045 | 0.042 | 0.284 | 0.056 | 0.636 | **1.075** |
| cpc=B60C + kw=tread | post | 640 | 0.003 | 0.284 | 1.019 | 0.358 | 0.75 | **2.496** |
| week >= 2025-01-01 | pre | 180 | 0.042 | 0.071 | 0.475 | 0.086 | 0.841 | **1.52** |
| week >= 2025-01-01 | post | 640 | 0.004 | 0.44 | 1.676 | 0.443 | 0.974 | **3.668** |
| exact title 'PNEUMATIC TIRE' (no query) | lookup | 33 | – | – | – | – | – | **0.029** |

All times are medians in milliseconds over 8 queries × 5 repeats. `candidates` = patents scored.

## B. Synthetic scale test: where filtering strategy matters

| patents (N) | filter | passing | post (score all) | pre (gather) | pre+partition (slice) |
|---|---|---|---|---|---|
| 640 | ~10% | 64 | 0.471 | 0.058 | 0.024 |
| 640 | ~1% | 6 | 0.323 | 0.02 | 0.006 |
| 64,000 | ~10% | 6,400 | 5.447 | 10.251 | 1.106 |
| 64,000 | ~1% | 640 | 4.202 | 1.585 | 0.448 |
| 256,000 | ~10% | 25,600 | 16.176 | 29.893 | 2.123 |
| 256,000 | ~1% | 2,560 | 15.504 | 3.326 | 0.651 |

Dense scoring + top-10 only, 1024-d float32 vectors, medians over 5 runs (ms). Rows beyond 640 are tiled copies of the real vectors.
