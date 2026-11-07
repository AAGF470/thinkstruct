"""Evaluate stage-1 retrieval vs two-phase reranking (Part 3).

    python eval_rerank.py                                  # qwen3 index, minilm reranker, 100 queries
    python eval_rerank.py --rerankers minilm bge --device cuda
    python eval_rerank.py --rerankers bge --n 20           # smaller sample (same seed = same first 20)

No human relevance labels exist for this data, so we use SILVER labels:
  query     = the first (non-canceled) claim of a sampled patent  -> a claim-mapping task
  relevant  = any OTHER patent in the same CPC main group (e.g. "B60C 23/" tire-pressure)
  excluded  = the query patent itself
Only patents whose group has >= --min-group members are sampled, so every query
has several relevant targets.

Caveats (stated in the README): group membership is a coarse proxy. It misses
relevant patents classified elsewhere and counts every same-group patent as
equally relevant. It measures topical precision, not true prior-art relevance.
It is good enough to compare systems against each other, not as an absolute score.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics as st
import time

import numpy as np

from data import is_canceled
from rerank import Reranker
from search import SearchEngine

SHOWCASE = [
    "tire with a sensor that monitors air pressure",
    "spoke connecting the hub to the rim of a bicycle wheel",
    "tread pattern that reduces road noise",
]


def group_of(p: dict) -> str:
    d = p["cpc"]["display"]
    return d.split("/")[0] + "/" if "/" in d else d


def metrics(ranked: list[str], rel: set[str], n_rel: int) -> dict:
    hits = [d in rel for d in ranked]
    p10 = sum(hits[:10]) / 10
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits[:10]))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(10, n_rel)))
    first = next((i for i, h in enumerate(hits[:30]) if h), None)
    return {"P@10": p10, "nDCG@10": dcg / idcg if idcg else 0.0,
            "MRR@30": 1 / (first + 1) if first is not None else 0.0,
            "R@30": sum(hits[:30]) / n_rel if n_rel else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="cache/qwen3")
    ap.add_argument("--rerankers", nargs="+", default=["minilm"])
    ap.add_argument("--n", type=int, default=100, help="number of sampled queries")
    ap.add_argument("--k1", type=int, default=100, help="stage-1 candidates")
    ap.add_argument("--k2", type=int, default=30, help="kept after rerank")
    ap.add_argument("--rr-max-length", type=int, default=512,
                    help="reranker input tokens (query + candidate); 256 = Raya-friendly")
    ap.add_argument("--qcaps", type=int, nargs="+", default=[0, 128],
                    help="query token caps to test (0 = uncapped); each is one scoring pass")
    ap.add_argument("--fuses", nargs="+", default=["replace", "rrf"], choices=["replace", "rrf"])
    ap.add_argument("--bm25-weight", type=float, default=None,
                    help="hybrid BM25 weight (default: search.BM25_WEIGHT). The saved "
                         "results/*_n100.md and *_fix.md files were produced with 1.0")
    ap.add_argument("--min-group", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (4 = Raya)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    if a.threads:
        torch.set_num_threads(a.threads)

    from search import BM25_WEIGHT
    bw = BM25_WEIGHT if a.bm25_weight is None else a.bm25_weight
    eng = SearchEngine(a.index, device=a.device)
    eng.model()
    groups = [group_of(p) for p in eng.docs]
    size = {g: groups.count(g) for g in set(groups)}

    # ---- sample queries (deterministic permutation; --n takes a prefix of it)
    eligible = []
    for i, p in enumerate(eng.docs):
        if size[groups[i]] < a.min_group:
            continue
        claim = next((c for c in p["claims"] if not is_canceled(c) and len(c.split()) >= 15), None)
        if claim:
            eligible.append((i, claim))
    perm = np.random.default_rng(a.seed).permutation(len(eligible))
    sample = [eligible[j] for j in perm[: a.n]]
    print(f"{len(eligible)} eligible query patents in {sum(s >= a.min_group for s in size.values())} "
          f"groups; using {len(sample)}")

    rerankers = {name: Reranker(name, device=a.device, max_length=a.rr_max_length)
                 for name in a.rerankers}
    def label(name, qc, fu):
        return f"hybrid+{name}" + (f" q≤{qc}" if qc else "") + (" +rrf" if fu == "rrf" else "")

    variants = [(n, qc, fu) for n in rerankers for qc in a.qcaps for fu in a.fuses]
    systems = ["bm25", "dense", "hybrid"] + [label(*v) for v in variants]
    per_sys = {s: [] for s in systems}
    rr_ms = {(n, qc): [] for n in rerankers for qc in a.qcaps}
    rand_p10 = []

    for qi, (i, claim) in enumerate(sample, 1):
        me = eng.docs[i]["doc_number"]
        rel = {eng.docs[j]["doc_number"] for j in range(eng.n) if groups[j] == groups[i] and j != i}
        rand_p10.append(len(rel) / (eng.n - 1))
        stage1 = {}
        for mode in ("bm25", "dense", "hybrid"):
            res = eng.search(claim, k=a.k1 + 1, mode=mode, bm25_weight=bw)["results"]
            stage1[mode] = [r for r in res if r["doc_number"] != me][: a.k1]
            per_sys[mode].append(metrics([r["doc_number"] for r in stage1[mode]], rel, len(rel)))
        for name, rr in rerankers.items():
            for qc in a.qcaps:  # one scoring pass per query cap; fusions reuse the scores
                t = time.perf_counter()
                s = rr.score(claim, stage1["hybrid"], query_max_tokens=qc or None)
                rr_ms[(name, qc)].append((time.perf_counter() - t) * 1000)
                for fu in a.fuses:
                    out = rr.apply(stage1["hybrid"], s, top_n=a.k2, fuse=fu)
                    per_sys[label(name, qc, fu)].append(
                        metrics([r["doc_number"] for r in out], rel, len(rel)))
        if qi % 10 == 0:
            print(f"  {qi}/{len(sample)} queries")

    # ---- report
    def mean(sysname, m):
        return st.mean(x[m] for x in per_sys[sysname])

    lines = [f"# Reranking evaluation ({a.index}, {len(sample)} queries)\n",
             f"- Machine: {platform.processor() or platform.machine()}, device={a.device}, "
             f"torch threads={torch.get_num_threads()}",
             f"- Embedding model: {eng.manifest['model_id']}. Rerankers: "
             + ", ".join(f"`{n}` = {r.model_id}" for n, r in rerankers.items()),
             f"- Query = first claim of a sampled patent; relevant = other patents in the same CPC "
             f"main group (silver labels, min group size {a.min_group}). Stage 1 keeps {a.k1} "
             f"candidates, the reranker keeps {a.k2}; reranker input capped at "
             f"{a.rr_max_length} tokens. Hybrid BM25 weight = {bw}.",
             f"- Random baseline P@10 = {st.mean(rand_p10):.3f}\n",
             "| system | P@10 | nDCG@10 | MRR@30 | R@30 | rerank ms/query |",
             "|---|---|---|---|---|---|"]
    lat_of = {label(*v): st.median(rr_ms[(v[0], v[1])]) for v in variants}
    for s in systems:
        lat = f"{lat_of[s]:.0f}" if s in lat_of else "–"
        lines.append(f"| {s} | {mean(s, 'P@10'):.3f} | {mean(s, 'nDCG@10'):.3f} | "
                     f"{mean(s, 'MRR@30'):.3f} | {mean(s, 'R@30'):.3f} | {lat} |")

    lines.append("\nPer-query nDCG@10 against `hybrid` (better / worse / tied):\n")
    base = [x["nDCG@10"] for x in per_sys["hybrid"]]
    for v in variants:
        b_ = [x["nDCG@10"] for x in per_sys[label(*v)]]
        w = sum(y > x + 1e-9 for x, y in zip(base, b_))
        l_ = sum(y < x - 1e-9 for x, y in zip(base, b_))
        lines.append(f"- `{label(*v)}`: {w} / {l_} / {len(base) - w - l_}")

    # ---- showcase: natural-language queries, before/after
    show = [(n, 0, "replace") for n in rerankers] + [(n, max(a.qcaps), "rrf") for n in rerankers
                                                     if "rrf" in a.fuses]
    lines.append("\n## Showcase: stage 1 vs reranked (top 5)\n")
    for q in SHOWCASE:
        s1 = eng.search(q, k=a.k1, bm25_weight=bw)["results"]
        lines.append(f"**{q}**\n")
        lines.append("| # | stage 1 (hybrid) | " + " | ".join(label(*v) for v in show) + " |")
        lines.append("|---|---|" + "---|" * len(show))
        outs = {v: rerankers[v[0]].rerank(q, s1, top_n=5, fuse=v[2], query_max_tokens=v[1] or None)
                for v in show}
        for k in range(5):
            cells = [f"{s1[k]['title'][:55]} ({s1[k]['cpc']})"]
            for n in show:
                r = outs[n][k]
                cells.append(f"{r['title'][:55]} ({r['cpc']}, was #{r['stage1_rank']})")
            lines.append(f"| {k + 1} | " + " | ".join(cells) + " |")
        lines.append("")

    text = "\n".join(lines) + "\n"
    print(text)
    out = a.out or os.path.join("results", f"rerank_eval_{eng.manifest['model_alias']}_"
                                f"{'-'.join(rerankers)}_k{a.k1}_len{a.rr_max_length}"
                                f"_n{len(sample)}_w{bw:g}.md")
    os.makedirs("results", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    with open(out.replace(".md", ".json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "per_query": per_sys,
                   "rerank_ms": {f"{n}|q{qc}": v for (n, qc), v in rr_ms.items()}}, f)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
