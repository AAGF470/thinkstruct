# Hybrid fusion: BM25 weight sweep (Qwen/Qwen3-Embedding-0.6B, `deploy/index/qwen3`)

Hybrid score = 1/(60+dense_rank) + w × 1/(60+bm25_rank). w = 0 is dense only.

| BM25 weight w | A: known items hit@1 | A: hit@3 | A: MRR | B: claims P@10 | B: nDCG@10 | B: MRR@30 |
|---|---|---|---|---|---|---|
| 0.0 | 10/12 | 12/12 | 0.92 | 0.505 | 0.539 | 0.727 |
| 0.1 | 8/12 | 11/12 | 0.78 | 0.514 | 0.548 | 0.739 |
| 0.25 | 8/12 | 9/12 | 0.76 | 0.516 | 0.548 | 0.726 |
| 0.5 | 7/12 | 9/12 | 0.69 | 0.501 | 0.535 | 0.719 |
| 0.75 | 5/12 | 9/12 | 0.60 | 0.492 | 0.529 | 0.728 |
| 1.0 | 4/12 | 9/12 | 0.56 | 0.484 | 0.520 | 0.728 |
