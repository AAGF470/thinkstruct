"""Benchmark search with and without hybrid filters (Part 1 deliverable).

    python bench.py                       # qwen3 index, writes results/bench_<model>.md
    python bench.py --index cache/minilm
    python bench.py --scale 64000 256000  # also run the synthetic scale test at these sizes

Part A: the real index (640 patents). Each scenario is run for every query,
        pre-filter vs post-filter, timed per stage. Query embeddings are computed
        once and cached, so the per-scenario numbers isolate the search work. The
        embedding cost is reported on its own (it is the same for every scenario).
Part B: a synthetic scale test. The patent-vector matrix is tiled up to N rows to
        show how the three filtering strategies diverge when N is large:
          post      score all N rows, then drop rows that fail the filter
          pre       gather the passing rows (fancy index = copy), score only those
          pre+part  rows stored sorted by CPC subclass, so a subclass filter is a
                    contiguous slice (no copy) - what a partitioned index gives you
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics as st
import time

import numpy as np

from search import Filters, SearchEngine

QUERIES = [
    "tire with a sensor that monitors air pressure",
    "spoke connecting the hub to the rim of a bicycle wheel",
    "run-flat tire with reinforced sidewall",
    "wheel hub bearing assembly",
    "tread pattern that reduces road noise",
    "trailer hitch coupling for towing",
    "lightweight carbon fiber wheel rim",
    "valve for inflating a tire",
]

SCENARIOS = [
    ("no filters", Filters()),
    ("cpc=B60B (wheels)", Filters(cpc=["B60B"])),
    ("cpc=B60C (tyres)", Filters(cpc=["B60C"])),
    ("cpc=B60C 23/ (pressure)", Filters(cpc=["B60C 23/"])),
    ("kw=sensor (title/abstract)", Filters(keywords=["sensor"])),
    ("cpc=B60C + kw=tread", Filters(cpc=["B60C"], keywords=["tread"])),
    ("week >= 2025-01-01", Filters(week_from="2025-01-01")),
]


def _median(xs):
    return round(st.median(xs), 3) if xs else 0.0


def bench_real(eng: SearchEngine, repeats: int, mode: str) -> tuple[list[str], dict]:
    lines = []
    # 1. cold query-embedding cost (model already loaded, nothing cached)
    eng.model()
    eng.embed_query("warm up")
    embed_ms = []
    for q in QUERIES:
        t = time.perf_counter()
        eng.embed_query(q)
        embed_ms.append((time.perf_counter() - t) * 1000)
    lines.append(f"Query embedding ({eng.manifest['model_id']}, {eng.device}): median "
                 f"**{_median(embed_ms)} ms**, max {max(embed_ms):.1f} ms per query "
                 "(same for every scenario, so excluded below).\n")

    lines.append(f"| scenario | strategy | candidates | filter | dense | bm25 | fuse | explain | **search total** |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    summary = {}
    for name, f in SCENARIOS:
        for strategy in (["pre", "post"] if f.active() else ["pre"]):
            per_stage: dict[str, list[float]] = {}
            cands = []
            for q in QUERIES:
                for _ in range(repeats):
                    r = eng.search(q, f, k=10, mode=mode, strategy=strategy)
                    for k, v in r["timings_ms"].items():
                        per_stage.setdefault(k, []).append(v)
                cands.append(r["n_candidates"])
            # search total = everything except the (cached) embed call
            tot = [t - e for t, e in zip(per_stage["total"], per_stage["embed"])]
            row = {k: _median(v) for k, v in per_stage.items()}
            row["search_total"] = _median(tot)
            summary[(name, strategy)] = row
            n_c = int(st.median(cands)) if strategy == "pre" else eng.n
            lines.append(f"| {name} | {strategy} | {n_c} | {row['filter']} | {row.get('dense', 0)} | "
                         f"{row.get('bm25', 0)} | {row['fuse']} | {row['explain']} | **{row['search_total']}** |")

    # exact title lookup: no model at all
    lt = []
    for _ in range(repeats * len(QUERIES)):
        lt.append(eng.search(None, Filters(title="PNEUMATIC TIRE"), k=10)["timings_ms"]["total"])
    lines.append(f"| exact title 'PNEUMATIC TIRE' (no query) | lookup | 33 | – | – | – | – | – | **{_median(lt)}** |")
    lines.append("\nAll times are medians in milliseconds over "
                 f"{len(QUERIES)} queries × {repeats} repeats. `candidates` = patents scored.")
    return lines, {"embed_ms": _median(embed_ms), "rows": summary}


def bench_scale(eng: SearchEngine, sizes: list[int], repeats: int) -> list[str]:
    """Dense step only, at synthetic sizes, for a ~10% and a ~1% selective filter."""
    q = eng.embed_query(QUERIES[0])
    P0 = eng.P
    subclass = np.array([p["cpc"]["subclass"] for p in eng.docs])
    lines = ["| patents (N) | filter | passing | post (score all) | pre (gather) | pre+partition (slice) |",
             "|---|---|---|---|---|---|"]
    for n in sizes:
        reps = int(np.ceil(n / len(P0)))
        P = np.tile(P0, (reps, 1))[:n]
        sub = np.tile(subclass, reps)[:n]
        order = np.argsort(sub, kind="stable")  # partitioned layout
        P_sorted, sub_sorted = np.ascontiguousarray(P[order]), sub[order]
        rng = np.random.default_rng(0)
        for label, sel in (("~10%", 0.10), ("~1%", 0.01)):
            # synthetic filter: one contiguous block of the sorted layout of that size
            m = max(1, int(n * sel))
            start = int(rng.integers(0, n - m + 1))
            idx_sorted = np.arange(start, start + m)
            idx_orig = np.sort(order[idx_sorted])        # same patents in original order
            mask = np.zeros(n, dtype=bool)
            mask[idx_orig] = True
            t_post, t_pre, t_part = [], [], []
            for _ in range(repeats):
                t = time.perf_counter()
                s = P @ q
                s = s[mask]
                np.argpartition(-s, 10)[:10] if len(s) > 10 else s
                t_post.append((time.perf_counter() - t) * 1000)

                t = time.perf_counter()
                s = P[idx_orig] @ q
                np.argpartition(-s, 10)[:10] if len(s) > 10 else s
                t_pre.append((time.perf_counter() - t) * 1000)

                t = time.perf_counter()
                s = P_sorted[start:start + m] @ q
                np.argpartition(-s, 10)[:10] if len(s) > 10 else s
                t_part.append((time.perf_counter() - t) * 1000)
            lines.append(f"| {n:,} | {label} | {m:,} | {_median(t_post)} | {_median(t_pre)} | "
                         f"{_median(t_part)} |")
        del P, P_sorted
    lines.append(f"\nDense scoring + top-10 only, {P0.shape[1]}-d float32 vectors, medians over "
                 f"{repeats} runs (ms). Rows beyond 640 are tiled copies of the real vectors.")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="cache/qwen3")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--scale", type=int, nargs="*", default=[640, 64_000, 256_000])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    t = time.perf_counter()
    eng = SearchEngine(a.index, device="cpu")
    startup = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    eng.model()
    model_load = (time.perf_counter() - t) * 1000

    import torch
    lines = [f"# Search benchmark: {eng.manifest['model_id']}\n",
             f"- Machine: {platform.processor() or platform.machine()}, {platform.platform()}, "
             f"torch threads={torch.get_num_threads()}, device=cpu",
             f"- Index: `{a.index}`: {eng.n} patents, {len(eng.chunks):,} chunks, "
             f"dim {eng.manifest['dim']}, mode `{a.mode}`",
             f"- One-time costs: index load {startup:.0f} ms, model load {model_load:.0f} ms\n",
             "## A. Real index: hybrid filters, pre- vs post-filtering\n"]
    real, _ = bench_real(eng, a.repeats, a.mode)
    lines += real
    if a.scale:
        lines += ["\n## B. Synthetic scale test: where filtering strategy matters\n"]
        lines += bench_scale(eng, a.scale, max(3, a.repeats))

    text = "\n".join(lines) + "\n"
    print(text)
    out = a.out or os.path.join("results", f"bench_{eng.manifest['model_alias']}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
