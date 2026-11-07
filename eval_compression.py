"""How much can the patent vectors be shrunk before retrieval quality drops?

Sizes the 10M-patent design (DESIGN.md). Uses the same 100 claim queries and
silver labels (same CPC main group) as eval_rerank.py, dense ranking only.

  dims:  Qwen3-Embedding is Matryoshka-trained, so the first d dimensions of a
         vector are themselves a usable embedding: truncate + re-normalise.
  dtype: float32 | int8 (per-vector scale) | binary (sign bits, Hamming distance)

    python eval_compression.py
"""

from __future__ import annotations

import os
import statistics as st

import numpy as np

from data import is_canceled
from eval_rerank import group_of, metrics
from search import SearchEngine


def l2n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-12)


def quantize(P, q, dtype):
    """Return a scoring function's inputs for each storage format."""
    if dtype == "float32":
        return P @ q
    if dtype == "int8":
        Ps = np.abs(P).max(axis=1, keepdims=True) / 127.0
        Pq = np.round(P / Ps).astype(np.int8)
        return (Pq.astype(np.float32) * Ps) @ q   # what the DB reconstructs
    if dtype == "binary":
        Pb, qb = P > 0, q > 0
        return -(Pb != qb).sum(axis=1)             # negative Hamming distance
    raise ValueError(dtype)


def main(index="cache/qwen3", n=100, seed=0, min_group=5):
    eng = SearchEngine(index, device="cpu")
    groups = [group_of(p) for p in eng.docs]
    size = {g: groups.count(g) for g in set(groups)}
    eligible = []
    for i, p in enumerate(eng.docs):
        if size[groups[i]] >= min_group:
            c = next((c for c in p["claims"] if not is_canceled(c) and len(c.split()) >= 15), None)
            if c:
                eligible.append((i, c))
    perm = np.random.default_rng(seed).permutation(len(eligible))
    sample = [eligible[j] for j in perm[:n]]           # identical to eval_rerank.py
    Q = np.stack([eng.embed_query(c) for _, c in sample])

    dims_list = [1024, 512, 256, 128] if eng.P.shape[1] == 1024 else [eng.P.shape[1]]
    lines = [f"# Vector compression vs retrieval quality ({eng.manifest['model_id']}, "
             f"{len(sample)} claim queries, dense only)\n",
             "| dims | storage | bytes/vector | 10M patents | P@10 | nDCG@10 | R@30 |",
             "|---|---|---|---|---|---|---|"]
    for d in dims_list:
        P, Qd = l2n(eng.P[:, :d]), l2n(Q[:, :d])
        for dtype in ("float32", "int8", "binary"):
            per = []
            for (i, _), q in zip(sample, Qd):
                s = quantize(P, q, dtype).astype(np.float64)
                s[i] = -np.inf                         # exclude the query patent
                top = np.argsort(-s, kind="stable")[:30]
                rel = {eng.docs[j]["doc_number"] for j in range(eng.n)
                       if groups[j] == groups[i] and j != i}
                per.append(metrics([eng.docs[j]["doc_number"] for j in top], rel, len(rel)))
            b = {"float32": 4 * d, "int8": d + 4, "binary": d // 8}[dtype]
            tot = b * 10_000_000 / 1e9
            lines.append(f"| {d} | {dtype} | {b} | {tot:.2f} GB | "
                         f"{st.mean(x['P@10'] for x in per):.3f} | "
                         f"{st.mean(x['nDCG@10'] for x in per):.3f} | "
                         f"{st.mean(x['R@30'] for x in per):.3f} |")
    lines.append("\n`10M patents` = raw vector bytes for one vector per patent (ANN graph overhead "
                 "not included). int8 adds a 4-byte per-vector scale.")
    text = "\n".join(lines) + "\n"
    print(text)
    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"compression_{eng.manifest['model_alias']}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
