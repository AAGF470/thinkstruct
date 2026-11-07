"""Sweep the BM25 weight in hybrid fusion on two query sets.

    python eval_hybrid_weight.py                       # writes results/hybrid_weight_sweep.md

  A. known items: the 12 hand-written paraphrases in smoke_test.py (target = 1 patent)
  B. claim queries: the 100 first-claim queries + CPC-group silver labels of eval_rerank.py
weight 0 = dense only; 1.0 = the original equal-weight hybrid.
"""

from __future__ import annotations

import os
import statistics as st

import numpy as np

from data import is_canceled
from eval_rerank import group_of, metrics
from search import SearchEngine
from smoke_test import KNOWN

WEIGHTS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


def main(index="deploy/index/qwen3", n=100, seed=0, min_group=5):
    eng = SearchEngine(index, device="cpu")
    lines = [f"# Hybrid fusion: BM25 weight sweep ({eng.manifest['model_id']}, `{index}`)\n",
             "Hybrid score = 1/(60+dense_rank) + w × 1/(60+bm25_rank). w = 0 is dense only.\n",
             "| BM25 weight w | A: known items hit@1 | A: hit@3 | A: MRR | B: claims P@10 | B: nDCG@10 | B: MRR@30 |",
             "|---|---|---|---|---|---|---|"]

    groups = [group_of(p) for p in eng.docs]
    size = {g: groups.count(g) for g in set(groups)}
    eligible = []
    for i, p in enumerate(eng.docs):
        if size[groups[i]] >= min_group:
            c = next((c for c in p["claims"] if not is_canceled(c) and len(c.split()) >= 15), None)
            if c:
                eligible.append((i, c))
    perm = np.random.default_rng(seed).permutation(len(eligible))
    sample = [eligible[j] for j in perm[:n]]  # identical to eval_rerank.py

    for w in WEIGHTS:
        rr = []
        for doc, q in KNOWN:
            res = eng.search(q, k=50, bm25_weight=w)["results"]
            rr.append(next((r["rank"] for r in res if r["doc_number"] == doc), None))
        per = []
        for i, claim in sample:
            me = eng.docs[i]["doc_number"]
            res = [r for r in eng.search(claim, k=31, bm25_weight=w)["results"]
                   if r["doc_number"] != me][:30]
            rel = {eng.docs[j]["doc_number"] for j in range(eng.n) if groups[j] == groups[i] and j != i}
            per.append(metrics([r["doc_number"] for r in res], rel, len(rel)))
        lines.append(
            f"| {w} | {sum(r == 1 for r in rr)}/{len(rr)} | {sum(bool(r) and r <= 3 for r in rr)}/{len(rr)} | "
            f"{st.mean(1 / r if r else 0 for r in rr):.2f} | {st.mean(x['P@10'] for x in per):.3f} | "
            f"{st.mean(x['nDCG@10'] for x in per):.3f} | {st.mean(x['MRR@30'] for x in per):.3f} |")
        print(lines[-1])

    text = "\n".join(lines) + "\n"
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "hybrid_weight_sweep.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
