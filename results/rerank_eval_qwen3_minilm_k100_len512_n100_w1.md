# Reranking evaluation (cache/qwen3, 100 queries)

- Machine: AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD, device=cpu, torch threads=12
- Embedding model: Qwen/Qwen3-Embedding-0.6B. Rerankers: `minilm` = cross-encoder/ms-marco-MiniLM-L6-v2
- Query = first claim of a sampled patent; relevant = other patents in the same CPC main group (silver labels, min group size 5). Stage 1 keeps 100 candidates, the reranker keeps 30; reranker input capped at 512 tokens.
- Random baseline P@10 = 0.066

| system | P@10 | nDCG@10 | MRR@30 | R@30 | rerank ms/query |
|---|---|---|---|---|---|
| bm25 | 0.376 | 0.418 | 0.668 | 0.247 | – |
| dense | 0.509 | 0.540 | 0.715 | 0.322 | – |
| hybrid | 0.489 | 0.528 | 0.744 | 0.314 | – |
| hybrid+minilm | 0.433 | 0.460 | 0.655 | 0.278 | 4496 |

`hybrid+minilm` vs `hybrid` on nDCG@10, per query: 30 better, 59 worse, 11 tied.

## Showcase: stage 1 vs reranked (top 5)

**tire with a sensor that monitors air pressure**

| # | stage 1 (hybrid) | reranked: minilm |
|---|---|---|
| 1 | SYSTEMS AND METHODS FOR TIRE TREADWEAR SENSING (B60C 11/24) | FLAT TIRE AIRBAG SYSTEM (B60C 17/02, was #11) |
| 2 | SYSTEM AND METHOD FOR TIRE LEAK DETECTION (B60C 23/00) | TIRE PRESSURE DETECTOR WITH MUTLTIPLE FREQUENCIES (B60C 23/04, was #12) |
| 3 | TIRE INFLATION SYSTEM AND COMMERCIAL VEHICLE WITH A TIR (B60C 23/00) | METHOD FOR OPERATING A MONITORING DEVICE FOR AT LEAST O (B60C 23/04, was #5) |
| 4 | Tire Pressure Sensor Information Processing Method and  (B60C 23/04) | MODULAR TIRE PRESSURE MONITORING SYSTEM (TPMS) SENSOR W (B60C 23/04, was #6) |
| 5 | METHOD FOR OPERATING A MONITORING DEVICE FOR AT LEAST O (B60C 23/04) | Tire Pressure Sensor Information Processing Method and  (B60C 23/04, was #4) |

**spoke connecting the hub to the rim of a bicycle wheel**

| # | stage 1 (hybrid) | reranked: minilm |
|---|---|---|
| 1 | BICYCLE WHEEL ASSEMBLY STRUCTURE (B60B 1/04) | Trochoid, Polar or Conic Section Spoke for Wheels (B60B 9/26, was #58) |
| 2 | BICYCLE HUB AND SPOKE ARRANGEMENT (B60B 1/04) | BICYCLE HUB AND SPOKE ARRANGEMENT (B60B 1/04, was #2) |
| 3 | INTERNAL CONNECTION OF A BICYCLE REAR HUB (B60B 27/02) | RIM OF A BICYCLE WHEEL AND RELATED BICYCLE WHEEL (B60B 1/04, was #7) |
| 4 | INTERNAL CONNECTION OF A BICYCLE REAR HUB (B60B 27/02) | BICYCLE WHEEL ASSEMBLY STRUCTURE (B60B 1/04, was #1) |
| 5 | BICYCLE HUB SYSTEM, METHOD AND DEVICE INCLUDING AN INTE (B60B 27/02) | WHEEL (B60B 5/02, was #17) |

**tread pattern that reduces road noise**

| # | stage 1 (hybrid) | reranked: minilm |
|---|---|---|
| 1 | LOW NOISE TIRE (B60C 11/13) | LOW NOISE TIRE (B60C 11/13, was #1) |
| 2 | Tire Having a Compromise in Terms of Performance Betwee (B60C 11/03) | DISCRETE SOUND ABSORBERS FOR TIRE CAVITY NOISE REDUCTIO (B60C 19/00, was #8) |
| 3 | Mixed-Use Heavy-Duty Vehicle Tire with Low Road Noise (B60C 11/11) | Mixed-Use Heavy-Duty Vehicle Tire with Low Road Noise (B60C 11/11, was #3) |
| 4 | TIRE WITH VARIABLE NUMBER OF LATERAL BLOCKS PER PITCH (B60C 11/03) | Tire Having a Compromise in Terms of Performance Betwee (B60C 11/03, was #2) |
| 5 | PNEUMATIC TIRE WITH A TEXTURED SURFACE, METHOD OF MANUF (B60C 13/00) | Heavy Goods Vehicle Tire with Complex Tread Pattern (B60C 11/03, was #10) |

