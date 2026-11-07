# Search benchmark: sentence-transformers/all-MiniLM-L6-v2

- Machine: AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD, Windows-11-10.0.26300-SP0, torch threads=12, device=cpu
- Index: `cache/minilm`: 640 patents, 11,218 chunks, dim 384, mode `hybrid`
- One-time costs: index load 523 ms, model load 15803 ms

## A. Real index: hybrid filters, pre- vs post-filtering

Query embedding (sentence-transformers/all-MiniLM-L6-v2, cpu): median **9.315 ms**, max 14.8 ms per query (same for every scenario, so excluded below).

| scenario | strategy | candidates | filter | dense | bm25 | fuse | explain | **search total** |
|---|---|---|---|---|---|---|---|---|
| no filters | pre | 640 | 0.004 | 0.029 | 1.116 | 0.182 | 0.8 | **2.137** |
| cpc=B60B (wheels) | pre | 298 | 0.024 | 0.034 | 0.458 | 0.085 | 0.593 | **1.237** |
| cpc=B60B (wheels) | post | 640 | 0.003 | 0.026 | 0.992 | 0.238 | 0.688 | **2.002** |
| cpc=B60C (tyres) | pre | 318 | 0.037 | 0.041 | 0.625 | 0.127 | 0.727 | **1.651** |
| cpc=B60C (tyres) | post | 640 | 0.004 | 0.038 | 1.284 | 0.34 | 0.844 | **2.559** |
| cpc=B60C 23/ (pressure) | pre | 28 | 0.029 | 0.01 | 0.191 | 0.044 | 0.812 | **1.11** |
| cpc=B60C 23/ (pressure) | post | 640 | 0.006 | 0.051 | 1.78 | 0.428 | 0.968 | **3.441** |
| kw=sensor (title/abstract) | pre | 29 | 0.063 | 0.011 | 0.209 | 0.064 | 0.914 | **1.452** |
| kw=sensor (title/abstract) | post | 640 | 0.005 | 0.043 | 1.467 | 0.375 | 0.885 | **2.913** |
| cpc=B60C + kw=tread | pre | 150 | 0.118 | 0.03 | 0.468 | 0.103 | 0.889 | **1.82** |
| cpc=B60C + kw=tread | post | 640 | 0.008 | 0.043 | 1.609 | 0.455 | 0.927 | **3.445** |
| week >= 2025-01-01 | pre | 180 | 0.039 | 0.026 | 0.408 | 0.078 | 0.708 | **1.376** |
| week >= 2025-01-01 | post | 640 | 0.004 | 0.038 | 1.249 | 0.289 | 0.833 | **2.493** |
| exact title 'PNEUMATIC TIRE' (no query) | lookup | 33 | – | – | – | – | – | **0.02** |

All times are medians in milliseconds over 8 queries × 5 repeats. `candidates` = patents scored.
