# Reranking evaluation (cache/qwen3, 100 queries)

- Machine: AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD, device=cpu, torch threads=12
- Embedding model: Qwen/Qwen3-Embedding-0.6B. Rerankers: `minilm` = cross-encoder/ms-marco-MiniLM-L6-v2
- Query = first claim of a sampled patent; relevant = other patents in the same CPC main group (silver labels, min group size 5). Stage 1 keeps 100 candidates, the reranker keeps 30; reranker input capped at 512 tokens. Hybrid BM25 weight = 0.1.
- Random baseline P@10 = 0.066

| system | P@10 | nDCG@10 | MRR@30 | R@30 | rerank ms/query |
|---|---|---|---|---|---|
| bm25 | 0.376 | 0.418 | 0.668 | 0.247 | – |
| dense | 0.509 | 0.540 | 0.715 | 0.322 | – |
| hybrid | 0.512 | 0.548 | 0.743 | 0.323 | – |
| hybrid+minilm | 0.446 | 0.474 | 0.668 | 0.287 | 2950 |
| hybrid+minilm +rrf | 0.502 | 0.531 | 0.723 | 0.337 | 2950 |
| hybrid+minilm q≤64 | 0.446 | 0.470 | 0.649 | 0.302 | 2388 |
| hybrid+minilm q≤64 +rrf | 0.498 | 0.525 | 0.702 | 0.331 | 2388 |

Per-query nDCG@10 against `hybrid` (better / worse / tied):

- `hybrid+minilm`: 28 / 63 / 9
- `hybrid+minilm +rrf`: 35 / 47 / 18
- `hybrid+minilm q≤64`: 30 / 61 / 9
- `hybrid+minilm q≤64 +rrf`: 38 / 45 / 17

## Showcase: stage 1 vs reranked (top 5)

**tire with a sensor that monitors air pressure**

| # | stage 1 (hybrid) | hybrid+minilm | hybrid+minilm q≤64 +rrf |
|---|---|---|---|
| 1 | SYSTEMS AND METHODS FOR TIRE TREADWEAR SENSING (B60C 11/24) | FLAT TIRE AIRBAG SYSTEM (B60C 17/02, was #8) | SYSTEM AND METHOD FOR TIRE LEAK DETECTION (B60C 23/00, was #2) |
| 2 | SYSTEM AND METHOD FOR TIRE LEAK DETECTION (B60C 23/00) | TIRE PRESSURE DETECTOR WITH MUTLTIPLE FREQUENCIES (B60C 23/04, was #14) | FLAT TIRE AIRBAG SYSTEM (B60C 17/02, was #8) |
| 3 | TIRE MANAGEMENT DEVICE, PROGRAM, AND TIRE MANAGEMENT ME (B60C 23/04) | METHOD FOR OPERATING A MONITORING DEVICE FOR AT LEAST O (B60C 23/04, was #10) | Tire Pressure Sensor Information Processing Method and  (B60C 23/04, was #5) |
| 4 | TIRE PRESSURE ALARM METHOD AND APPARATUS, COMPUTER DEVI (B60C 23/06) | MODULAR TIRE PRESSURE MONITORING SYSTEM (TPMS) SENSOR W (B60C 23/04, was #15) | METHOD FOR OPERATING A MONITORING DEVICE FOR AT LEAST O (B60C 23/04, was #10) |
| 5 | Tire Pressure Sensor Information Processing Method and  (B60C 23/04) | Tire Pressure Sensor Information Processing Method and  (B60C 23/04, was #5) | TIRE INFLATION SYSTEM AND COMMERCIAL VEHICLE WITH A TIR (B60C 23/00, was #6) |

**spoke connecting the hub to the rim of a bicycle wheel**

| # | stage 1 (hybrid) | hybrid+minilm | hybrid+minilm q≤64 +rrf |
|---|---|---|---|
| 1 | BICYCLE WHEEL ASSEMBLY STRUCTURE (B60B 1/04) | Trochoid, Polar or Conic Section Spoke for Wheels (B60B 9/26, was #83) | BICYCLE HUB AND SPOKE ARRANGEMENT (B60B 1/04, was #2) |
| 2 | BICYCLE HUB AND SPOKE ARRANGEMENT (B60B 1/04) | BICYCLE HUB AND SPOKE ARRANGEMENT (B60B 1/04, was #2) | BICYCLE WHEEL ASSEMBLY STRUCTURE (B60B 1/04, was #1) |
| 3 | INTERNAL CONNECTION OF A BICYCLE REAR HUB (B60B 27/02) | RIM OF A BICYCLE WHEEL AND RELATED BICYCLE WHEEL (B60B 1/04, was #21) | FACE TO RIM REINFORCING CONNECTOR OF A COMPOSITE WHEEL (B60B 3/10, was #4) |
| 4 | FACE TO RIM REINFORCING CONNECTOR OF A COMPOSITE WHEEL (B60B 3/10) | BICYCLE WHEEL ASSEMBLY STRUCTURE (B60B 1/04, was #1) | INTERNAL CONNECTION OF A BICYCLE REAR HUB (B60B 27/02, was #3) |
| 5 | INTERNAL CONNECTION OF A BICYCLE REAR HUB (B60B 27/02) | WHEEL (B60B 5/02, was #24) | BICYCLE HUB SYSTEM, METHOD AND DEVICE INCLUDING AN INTE (B60B 27/02, was #7) |

**tread pattern that reduces road noise**

| # | stage 1 (hybrid) | hybrid+minilm | hybrid+minilm q≤64 +rrf |
|---|---|---|---|
| 1 | LOW NOISE TIRE (B60C 11/13) | LOW NOISE TIRE (B60C 11/13, was #1) | LOW NOISE TIRE (B60C 11/13, was #1) |
| 2 | Tire Having a Compromise in Terms of Performance Betwee (B60C 11/03) | DISCRETE SOUND ABSORBERS FOR TIRE CAVITY NOISE REDUCTIO (B60C 19/00, was #6) | Tire Having a Compromise in Terms of Performance Betwee (B60C 11/03, was #2) |
| 3 | Mixed-Use Heavy-Duty Vehicle Tire with Low Road Noise (B60C 11/11) | Mixed-Use Heavy-Duty Vehicle Tire with Low Road Noise (B60C 11/11, was #3) | Mixed-Use Heavy-Duty Vehicle Tire with Low Road Noise (B60C 11/11, was #3) |
| 4 | TIRE WITH VARIABLE NUMBER OF LATERAL BLOCKS PER PITCH (B60C 11/03) | Tire Having a Compromise in Terms of Performance Betwee (B60C 11/03, was #2) | DISCRETE SOUND ABSORBERS FOR TIRE CAVITY NOISE REDUCTIO (B60C 19/00, was #6) |
| 5 | TIRE (B60C 11/03) | Heavy Goods Vehicle Tire with Complex Tread Pattern (B60C 11/03, was #13) | TIRE WITH VARIABLE NUMBER OF LATERAL BLOCKS PER PITCH (B60C 11/03, was #4) |

