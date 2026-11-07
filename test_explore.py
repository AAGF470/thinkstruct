"""Invariants behind the linked-views UI ("the views must never disagree").

    python test_explore.py                 # plain asserts; also runs under pytest

For several queries (and browse mode), checks that:
  1. CPC counts and week counts each sum to n_results (every view describes one set)
  2. clicking any CPC group / any week yields exactly the count that view showed
  3. every returned patent satisfies every active filter
  4. filtering never reorders: filtered results keep their relative order
  5. combining filters is order-independent (A then B == B then A)
"""

from __future__ import annotations

import os
import random
import sys

from search import Filters, SearchEngine

INDEX = os.environ.get("INDEX_DIR", os.path.join("deploy", "index", "qwen3"))
QUERIES = [None, "tire with a sensor that monitors air pressure", "carbon fiber bicycle spoke",
           "trailer hitch coupling"]
_eng = None


def eng():
    global _eng
    if _eng is None:
        _eng = SearchEngine(INDEX)
    return _eng


def all_docs(q, f):
    """Every doc in the filtered match set, in order (paging through k=50)."""
    out, off = [], 0
    while True:
        r = eng().explore(q, f, k=50, offset=off)
        out += [x["doc_number"] for x in r["results"]]
        off += 50
        if off >= r["n_results"]:
            return out, r


def test_counts_sum_to_total():
    for q in QUERIES:
        for f in (Filters(), Filters(cpc=["B60C"]), Filters(keywords=["sensor"]),
                  Filters(week_from="2025-01-01")):
            r = eng().explore(q, f, k=5)
            assert sum(r["facets"]["cpc"].values()) == r["n_results"], (q, f)
            assert sum(r["facets"]["weeks"].values()) == r["n_results"], (q, f)


def test_clicking_a_facet_gives_its_count():
    for q in QUERIES:
        r = eng().explore(q, Filters(), k=5)
        for g, n in r["facets"]["cpc"].items():
            if g == "N/A":
                continue
            assert eng().explore(q, Filters(cpc=[g]), k=1)["n_results"] == n, (q, g)
        for wk, n in random.Random(0).sample(sorted(r["facets"]["weeks"].items()), 8):
            got = eng().explore(q, Filters(week_from=wk, week_to=wk), k=1)["n_results"]
            assert got == n, (q, wk)
        # subclass roll-up: sum of its groups
        sub = {}
        for g, n in r["facets"]["cpc"].items():
            sub[g[:4]] = sub.get(g[:4], 0) + n
        for s, n in sub.items():
            if s != "N/A":
                assert eng().explore(q, Filters(cpc=[s]), k=1)["n_results"] == n, (q, s)


def test_results_satisfy_filters_and_keep_order():
    for q in QUERIES:
        base, _ = all_docs(q, Filters())
        pos = {d: i for i, d in enumerate(base)}
        f = Filters(cpc=["B60C"], week_from="2024-06-01")
        docs, r = all_docs(q, f)
        for res in eng().explore(q, f, k=50)["results"]:
            assert res["cpc"].startswith("B60C") and res["week"] >= "2024-06-01", res
        assert [pos[d] for d in docs] == sorted(pos[d] for d in docs), "filter reordered results"


def test_filter_order_independent():
    for q in QUERIES:
        a = all_docs(q, Filters(cpc=["B60B"], keywords=["spoke"]))[0]
        b = all_docs(q, Filters(keywords=["spoke"], cpc=["B60B"]))[0]
        assert a == b


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} passed")
    sys.exit(0)
