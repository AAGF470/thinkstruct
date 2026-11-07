# Vector compression vs retrieval quality (Qwen/Qwen3-Embedding-0.6B, 100 claim queries, dense only)

| dims | storage | bytes/vector | 10M patents | P@10 | nDCG@10 | R@30 |
|---|---|---|---|---|---|---|
| 1024 | float32 | 4096 | 40.96 GB | 0.509 | 0.540 | 0.322 |
| 1024 | int8 | 1028 | 10.28 GB | 0.509 | 0.540 | 0.322 |
| 1024 | binary | 128 | 1.28 GB | 0.476 | 0.514 | 0.294 |
| 512 | float32 | 2048 | 20.48 GB | 0.505 | 0.539 | 0.307 |
| 512 | int8 | 516 | 5.16 GB | 0.505 | 0.538 | 0.308 |
| 512 | binary | 64 | 0.64 GB | 0.459 | 0.495 | 0.276 |
| 256 | float32 | 1024 | 10.24 GB | 0.470 | 0.513 | 0.297 |
| 256 | int8 | 260 | 2.60 GB | 0.471 | 0.514 | 0.296 |
| 256 | binary | 32 | 0.32 GB | 0.399 | 0.424 | 0.234 |
| 128 | float32 | 512 | 5.12 GB | 0.451 | 0.487 | 0.276 |
| 128 | int8 | 132 | 1.32 GB | 0.452 | 0.488 | 0.278 |
| 128 | binary | 16 | 0.16 GB | 0.359 | 0.393 | 0.200 |

`10M patents` = raw vector bytes for one vector per patent (ANN graph overhead not included). int8 adds a 4-byte per-vector scale.
