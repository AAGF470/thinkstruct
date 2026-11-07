"""Hybrid patent search over an index cache built by build_index.py.

    python search.py "tire with a sensor that monitors air pressure"
    python search.py "spoke connecting hub to rim" --cpc B60B --kw spoke -k 5
    python search.py --title "PNEUMATIC TIRE"                 # exact title, no model needed
    python search.py --like 20240051333                       # patent as the query
    python search.py --like-claim 20240051333:1               # one claim as the query
    python search.py -i --index cache/minilm                  # interactive, model stays loaded

Pipeline per query (see ARCHITECTURE.md §3):
  1. FILTER   metadata constraints -> candidate patents (inverted index / vector ops, no AI)
  2. EMBED    the query once with the index's model + query prompt (skipped for --like*)
  3. DENSE    cosine score of each candidate's title+abstract+claims vector
  4. BM25     keyword score over the same candidates
  5. FUSE     Reciprocal Rank Fusion of 3 and 4  (mode=hybrid, the default)
  6. EXPLAIN  for the top k only: which claim matched best, which query terms hit

strategy="pre" (default) filters BEFORE scoring, so constraints shrink the work.
strategy="post" scores everything and filters afterwards; kept for benchmarking.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from build_index import load_index, load_model
from data import is_canceled, parse_cpc, tokenize

RRF_K = 60  # standard Reciprocal Rank Fusion constant
# Weight of the BM25 ranking in hybrid fusion (dense = 1.0). Equal weights hurt:
# BM25 misses paraphrased queries and drags dense #1 hits down to #15-#23.
# Chosen by eval_hybrid_weight.py (results/hybrid_weight_sweep.md): 0.1 is best on the
# 100 claim queries (nDCG@10 0.548 vs 0.520 at 1.0) and keeps paraphrase MRR at 0.78
# (dense-only 0.92, equal weights 0.56). Use mode="dense" for pure semantic search.
BM25_WEIGHT = 0.1


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 3)


def _norm_title(s: str) -> str:
    return " ".join(s.lower().split())


@dataclass
class Filters:
    cpc: list[str] = field(default_factory=list)       # OR of prefixes: "B60B", "B60C 11/"
    keywords: list[str] = field(default_factory=list)  # AND: every keyword must appear
    kw_field: str = "any"                              # where keywords must appear: title | abstract | any
    title: str | None = None                           # exact title, case/space-insensitive
    week_from: str | None = None                       # ISO dates, inclusive
    week_to: str | None = None

    def active(self) -> bool:
        return bool(self.cpc or self.keywords or self.title or self.week_from or self.week_to)

    def describe(self) -> str:
        parts = [f"cpc={'|'.join(self.cpc)}"] if self.cpc else []
        parts += [f"kw[{self.kw_field}]={k!r}" for k in self.keywords]
        if self.title:
            parts.append(f"title={self.title!r}")
        if self.week_from or self.week_to:
            parts.append(f"week={self.week_from or '…'}..{self.week_to or '…'}")
        return ", ".join(parts) or "none"


def _rrf(scores: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """1 / (K + rank). Items marked invalid (e.g. BM25 score 0) get no credit."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(len(scores))
    out = 1.0 / (RRF_K + ranks + 1)
    if valid is not None:
        out[~valid] = 0.0
    return out


class SearchEngine:
    def __init__(self, cache_dir: str, device: str = "cpu"):
        t0 = time.perf_counter()
        ix = load_index(cache_dir)
        self.cache_dir, self.device, self._model = cache_dir, device, None
        self.manifest, self.E, self.chunks = ix["manifest"], ix["embeddings"], ix["chunks"]
        self.docs = list(ix["patents"].values())
        for p in self.docs:  # re-parse with the current parser (older caches stored an older parse)
            p["cpc"] = parse_cpc(p.get("classification"))
        self.doc_idx = {p["doc_number"]: i for i, p in enumerate(self.docs)}
        n = self.n = len(self.docs)

        # chunk rows grouped by patent: 1 "patent" row + its claim rows
        self.patent_row = np.full(n, -1, dtype=np.int64)
        claim_rows: list[list[int]] = [[] for _ in range(n)]
        claim_nos: list[list[int]] = [[] for _ in range(n)]
        for c in self.chunks:
            i = self.doc_idx[c["doc_number"]]
            if c["kind"] == "patent":
                self.patent_row[i] = c["chunk_id"]
            elif c["kind"] == "claim":
                # older caches still contain "(canceled)" placeholder claims
                if is_canceled(self.docs[i]["claims"][c["claim_no"] - 1]):
                    continue
                claim_rows[i].append(c["chunk_id"])
                claim_nos[i].append(c["claim_no"])
        self.claim_rows = [np.asarray(r, dtype=np.int64) for r in claim_rows]
        self.claim_nos = [np.asarray(r, dtype=np.int64) for r in claim_nos]
        # Contiguous patent-vector matrix: makes the dense step one matmul.
        self.P = np.ascontiguousarray(self.E[self.patent_row])

        # metadata for filters
        # CPC prefix index: codes sorted once, so any prefix is a contiguous range
        # found by binary search (O(log n)) instead of a string scan (O(n)).
        self._cpc_index = {}
        for name, codes in (("raw", [p["cpc"]["raw"] for p in self.docs]),
                            ("display", [p["cpc"]["display"].upper() for p in self.docs])):
            order = np.argsort(codes, kind="stable")
            self._cpc_index[name] = (np.asarray(codes)[order], order)
        self.week = np.array([p["week"] for p in self.docs])
        self.by_title: dict[str, list[int]] = {}
        self.inv: dict[str, dict[str, set[int]]] = {"title": {}, "abstract": {}}
        corpus = []
        for i, p in enumerate(self.docs):
            self.by_title.setdefault(_norm_title(p["title"]), []).append(i)
            for fld in ("title", "abstract"):
                for tok in set(tokenize(p[fld])):
                    self.inv[fld].setdefault(tok, set()).add(i)
            corpus.append(tokenize(" ".join([p["title"], p["abstract"], " ".join(p["claims"])])))
        self.bm25 = BM25Okapi(corpus)
        self._qcache: dict[str, np.ndarray] = {}
        self.startup_ms = _ms(t0)

    # ------------------------------------------------------------------ model
    def model(self):
        if self._model is None:
            m = self.manifest
            self._model, spec = load_model(m["model_alias"], self.device, m["max_seq_length"])
            if spec["id"] != m["model_id"]:
                raise RuntimeError(f"index built with {m['model_id']}, model registry says {spec['id']}")
        return self._model

    def embed_query(self, text: str) -> np.ndarray:
        if text not in self._qcache:
            v = self.model().encode([text], prompt=self.manifest["query_prompt"] or None,
                                    normalize_embeddings=True, convert_to_numpy=True)[0]
            v = v.astype(np.float32)
            d = self.E.shape[1]
            if v.shape[0] > d:  # Matryoshka-truncated export: truncate + re-normalise
                v = v[:d] / max(float(np.linalg.norm(v[:d])), 1e-12)
            self._qcache[text] = v
        return self._qcache[text]

    def chunk_text(self, row: int) -> str:
        """Chunk text, rebuilt from patents.json when a slim export dropped it."""
        c = self.chunks[row]
        if c.get("text"):
            return c["text"]
        p = self.docs[self.doc_idx[c["doc_number"]]]
        if c["kind"] == "claim":
            return f"{p['title']}. {p['claims'][c['claim_no'] - 1]}"
        return f"{p['title']}\n{p['abstract']}"

    # ------------------------------------------------------------------ filters
    def _cpc_mask(self, prefix: str) -> np.ndarray:
        q = " ".join(prefix.upper().split())
        # "B60C 11/" (has space or slash) matches the parsed display form, so it does
        # not also catch "B60C 1/10"; a bare "B60C" matches the raw code.
        sorted_codes, order = self._cpc_index["display" if (" " in q or "/" in q) else "raw"]
        lo = np.searchsorted(sorted_codes, q, side="left")
        hi = np.searchsorted(sorted_codes, q + "￿", side="left")
        m = np.zeros(self.n, dtype=bool)
        m[order[lo:hi]] = True
        return m

    def filter_docs(self, f: Filters) -> np.ndarray:
        """Indices of patents passing every constraint."""
        mask = np.ones(self.n, dtype=bool)
        if f.title is not None:
            m = np.zeros(self.n, dtype=bool)
            m[self.by_title.get(_norm_title(f.title), [])] = True
            mask &= m
        if f.cpc:
            m = np.zeros(self.n, dtype=bool)
            for pre in f.cpc:
                m |= self._cpc_mask(pre)
            mask &= m
        fields = ["title", "abstract"] if f.kw_field == "any" else [f.kw_field]
        for kw in f.keywords:  # inverted-index lookups, never a scan of the text
            toks = tokenize(kw)
            if not toks:
                continue
            hit: set[int] = set()
            for fld in fields:
                hit |= set.intersection(*(self.inv[fld].get(t, set()) for t in toks))
            m = np.zeros(self.n, dtype=bool)
            m[list(hit)] = True
            mask &= m
        if f.week_from:
            mask &= self.week >= f.week_from
        if f.week_to:
            mask &= self.week <= f.week_to
        return np.flatnonzero(mask)

    # ------------------------------------------------------------------ search
    def _prepare_query(self, query, like, like_claim, mode, T):
        """-> (query vector or None, text for BM25 / highlighting, index of the source
        patent to exclude for --like / --like-claim, or None)."""
        if like or like_claim:
            doc, _, no = (like or like_claim).partition(":")
            if doc not in self.doc_idx:
                raise KeyError(f"unknown doc_number {doc}")
            self_doc = self.doc_idx[doc]
            if like_claim:
                hit = np.flatnonzero(self.claim_nos[self_doc] == int(no))
                if not len(hit):
                    raise KeyError(f"{doc} has no searchable claim {no} (missing or canceled)")
                row = int(self.claim_rows[self_doc][hit[0]])
            else:
                row = int(self.patent_row[self_doc])
            T["embed"] = 0.0
            return self.E[row], self.chunk_text(row), self_doc
        if mode != "bm25" and self._model is None:  # one-time cost, reported apart
            t = time.perf_counter()
            self.model()
            T["model_load"] = _ms(t)
        t = time.perf_counter()
        qvec = self.embed_query(query) if mode != "bm25" else None
        T["embed"] = _ms(t)
        return qvec, query, None

    def explore(self, query: str | None = None, filters: Filters | None = None, k: int = 20,
                offset: int = 0, match_n: int = 200, mode: str = "hybrid",
                like: str | None = None, like_claim: str | None = None,
                bm25_weight: float = BM25_WEIGHT) -> dict:
        """Linked-views query (the web UI): ONE result set + facets that always agree.

        Match set M = the query's top `match_n` patents over the WHOLE corpus (or every
        patent, newest first, when there is no query). Filters only ever select a
        subset of M, and the results page, CPC counts and week counts are all computed
        from that same subset. So a facet bar showing 80 yields exactly 80 results
        when clicked, and the counts in every view sum to `n_results`.
        """
        f = filters or Filters()
        T: dict[str, float] = {}
        t_all = time.perf_counter()
        has_q = bool(query or like or like_claim)
        qvec, qtoks = None, []
        fused = dense = bm = None

        t = time.perf_counter()
        if has_q:
            qvec, qtext, self_doc = self._prepare_query(query, like, like_claim, mode, T)
            qtoks = tokenize(qtext)
            t = time.perf_counter()
            if mode in ("dense", "hybrid"):
                dense = self.P @ qvec
            if mode in ("bm25", "hybrid"):
                bm = np.asarray(self.bm25.get_scores(qtoks))
            if mode == "hybrid":
                fused = _rrf(dense) + bm25_weight * _rrf(bm, valid=bm > 0)
            else:
                fused = dense if dense is not None else bm
            if self_doc is not None:
                fused = fused.copy()
                fused[self_doc] = -np.inf
            order = np.argsort(-fused, kind="stable")
            if self_doc is not None:
                order = order[order != self_doc]
            M = order[:match_n]
        else:  # browse: whole corpus, newest filings first
            M = np.lexsort((np.arange(self.n), [-int(w.replace("-", "") or 0) for w in self.week]))
        T["rank"] = _ms(t)

        t = time.perf_counter()
        if f.active():
            M_f = M[np.isin(M, self.filter_docs(f), assume_unique=True)]
        else:
            M_f = M
        T["filter"] = _ms(t)

        t = time.perf_counter()
        cpc_counts: dict[str, int] = {}
        week_counts: dict[str, int] = {}
        for i in M_f:
            p = self.docs[int(i)]
            disp = p["cpc"]["display"]
            g = disp.split("/")[0] + "/" if "/" in disp else "N/A"
            cpc_counts[g] = cpc_counts.get(g, 0) + 1
            week_counts[p["week"]] = week_counts.get(p["week"], 0) + 1
        T["facets"] = _ms(t)

        t = time.perf_counter()
        results = []
        for j, i in enumerate(M_f[offset:offset + k]):
            i = int(i)
            sc = {}
            if fused is not None:
                sc["score"] = round(float(fused[i]), 5)
            if dense is not None:
                sc["dense"] = round(float(dense[i]), 4)
            if bm is not None:
                sc["bm25"] = round(float(bm[i]), 3)
            results.append(self._result(i, offset + j + 1, qvec, sc, qtoks))
        T["explain"] = _ms(t)
        T["total"] = _ms(t_all)
        return {"results": results, "n_results": int(len(M_f)), "n_match": int(len(M)),
                "match_n": match_n if has_q else None, "offset": offset,
                "facets": {"cpc": cpc_counts, "weeks": week_counts},
                "mode": mode if has_q else "browse", "filters": f.describe(),
                "query_terms": sorted(set(qtoks)), "timings_ms": T,
                "model": self.manifest["model_id"]}

    def meta(self) -> dict:
        """Corpus-wide facts the UI needs once: the full week axis and CPC totals."""
        cpc: dict[str, int] = {}
        for p in self.docs:
            d = p["cpc"]["display"]
            g = d.split("/")[0] + "/" if "/" in d else "N/A"
            cpc[g] = cpc.get(g, 0) + 1
        return {"n_patents": self.n, "weeks": sorted(set(self.week.tolist())), "cpc": cpc,
                "model": self.manifest["model_id"], "dim": int(self.E.shape[1])}

    def search(self, query: str | None = None, filters: Filters | None = None, k: int = 10,
               mode: str = "hybrid", strategy: str = "pre", like: str | None = None,
               like_claim: str | None = None, bm25_weight: float = BM25_WEIGHT) -> dict:
        f = filters or Filters()
        T: dict[str, float] = {}
        t_all = time.perf_counter()

        t = time.perf_counter()
        cand = self.filter_docs(f) if (strategy == "pre" and f.active()) else np.arange(self.n)
        T["filter"] = _ms(t)

        # Metadata-only lookup (e.g. exact title): no model, no scoring.
        if query is None and like is None and like_claim is None:
            if not f.active():
                raise ValueError("give a query, --like/--like-claim, or at least one filter")
            hits = [self._result(i, rank, None, {}) for rank, i in enumerate(cand[:k], 1)]
            T["total"] = _ms(t_all)
            return {"results": hits, "n_candidates": int(len(cand)), "timings_ms": T,
                    "mode": "lookup", "strategy": strategy, "filters": f.describe()}

        qvec, qtext, self_doc = self._prepare_query(query, like, like_claim, mode, T)
        if self_doc is not None:
            cand = cand[cand != self_doc]
        qtoks = tokenize(qtext)

        dense = bm = None
        if mode in ("dense", "hybrid"):
            t = time.perf_counter()
            dense = self.P[cand] @ qvec if len(cand) < self.n else self.P @ qvec
            T["dense"] = _ms(t)
        if mode in ("bm25", "hybrid"):
            t = time.perf_counter()
            bm = np.asarray(self.bm25.get_batch_scores(qtoks, cand.tolist()))
            T["bm25"] = _ms(t)

        t = time.perf_counter()
        if mode == "hybrid":
            fused = _rrf(dense) + bm25_weight * _rrf(bm, valid=bm > 0)
        else:
            fused = dense if dense is not None else bm
        if strategy == "post" and f.active():  # score-then-filter baseline
            keep = np.isin(cand, self.filter_docs(f))
            fused, cand = fused[keep], cand[keep]
            dense = dense[keep] if dense is not None else None
            bm = bm[keep] if bm is not None else None
        top = np.argsort(-fused, kind="stable")[:k] if len(fused) else np.array([], dtype=int)
        T["fuse"] = _ms(t)

        t = time.perf_counter()
        results = []
        for rank, j in enumerate(top, 1):
            i = int(cand[j])
            scores = {"score": round(float(fused[j]), 5)}
            if dense is not None:
                scores["dense"] = round(float(dense[j]), 4)
            if bm is not None:
                scores["bm25"] = round(float(bm[j]), 3)
            results.append(self._result(i, rank, qvec, scores, qtoks))
        T["explain"] = _ms(t)
        T["total"] = _ms(t_all)
        return {"results": results, "n_candidates": int(len(cand)), "timings_ms": T,
                "mode": mode, "strategy": strategy, "filters": f.describe(),
                "model": self.manifest["model_id"]}

    def _result(self, i: int, rank: int, qvec, scores: dict, qtoks=()) -> dict:
        p = self.docs[i]
        out = {"rank": rank, "doc_number": p["doc_number"], "title": p["title"],
               "cpc": p["cpc"]["display"], "week": p["week"], **scores,
               "abstract": p["abstract"]}
        rows, nos = self.claim_rows[i], self.claim_nos[i]
        if len(rows):
            if qvec is not None:  # best-matching claim = the "why it matched"
                cs = self.E[rows] @ qvec
                b = int(np.argmax(cs))
                no = int(nos[b])
                out["matched_claim"] = {"claim_no": no, "score": round(float(cs[b]), 4),
                                        "text": p["claims"][no - 1]}
            elif qtoks:  # BM25-only: claim with the most query-term overlap
                q = set(qtoks)
                no = int(max(nos, key=lambda c: len(q & set(tokenize(p["claims"][c - 1])))))
                out["matched_claim"] = {"claim_no": no, "score": None, "text": p["claims"][no - 1]}
        if qtoks:
            ta = set(tokenize(p["title"] + " " + p["abstract"]))
            out["terms_in_title_abstract"] = sorted(set(qtoks) & ta)
        return out


# ---------------------------------------------------------------------- CLI
def _default_index() -> str:
    return "cache/qwen3" if os.path.exists("cache/qwen3/manifest.json") else "cache/minilm"


def _print(res: dict, width: int = 110) -> None:
    T = res["timings_ms"]
    stages = ", ".join(f"{k} {v:.1f}" for k, v in T.items() if k != "total")
    print(f"\n[{res.get('model', '')} | {res['mode']} | {res['strategy']}-filter | filters: "
          f"{res['filters']}] {res['n_candidates']} candidates | {T['total']:.1f} ms ({stages})")
    if not res["results"]:
        print("  no results")
    for r in res["results"]:
        sc = "  ".join(f"{k} {r[k]}" for k in ("score", "dense", "bm25") if k in r)
        print(f"{r['rank']:>3}. {r['doc_number']}  {r['cpc']:<12} {r['week']}  {sc}")
        print(f"     {r['title'][:width]}")
        mc = r.get("matched_claim")
        if mc:
            s = f" ({mc['score']})" if mc["score"] is not None else ""
            print(f"     claim {mc['claim_no']}{s}: {mc['text'][:width - 20]}…")
        if r.get("terms_in_title_abstract"):
            print(f"     terms in title/abstract: {', '.join(r['terms_in_title_abstract'])}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Hybrid patent search")
    ap.add_argument("query", nargs="?", help="natural-language query")
    ap.add_argument("--index", default=_default_index(), help="cache folder from build_index.py")
    ap.add_argument("--cpc", action="append", default=[], help="classification prefix (repeatable, OR)")
    ap.add_argument("--kw", action="append", default=[], help="keyword that must appear (repeatable, AND)")
    ap.add_argument("--kw-field", default="any", choices=["any", "title", "abstract"])
    ap.add_argument("--title", help="exact title match")
    ap.add_argument("--from", dest="week_from", help="filing week >= YYYY-MM-DD")
    ap.add_argument("--to", dest="week_to", help="filing week <= YYYY-MM-DD")
    ap.add_argument("--like", help="use this patent (doc_number) as the query")
    ap.add_argument("--like-claim", help="use one claim as the query: DOC_NUMBER:CLAIM_NO")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    ap.add_argument("--strategy", default="pre", choices=["pre", "post"])
    ap.add_argument("--rerank", default=None, help="Part 3: rerank with minilm | bge | HF id")
    ap.add_argument("--k1", type=int, default=100, help="stage-1 candidates when --rerank is set")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    ap.add_argument("-i", "--interactive", action="store_true", help="prompt for queries in a loop")
    a = ap.parse_args()

    eng = SearchEngine(a.index, device=a.device)
    f = Filters(cpc=a.cpc, keywords=a.kw, kw_field=a.kw_field, title=a.title,
                week_from=a.week_from, week_to=a.week_to)

    rr = None
    if a.rerank:
        from rerank import Reranker
        rr = Reranker(a.rerank, device=a.device)

    def run(q):
        kw = dict(mode=a.mode, strategy=a.strategy, like=a.like, like_claim=a.like_claim)
        if rr is not None:
            from rerank import two_phase
            res = two_phase(eng, rr, q, f, k1=a.k1, k2=a.k, **kw)
        else:
            res = eng.search(q, f, k=a.k, **kw)
        print(json.dumps(res, indent=2, ensure_ascii=False)) if a.json else _print(res)

    if a.interactive:
        print(f"index {a.index} ({eng.manifest['model_id']}), {eng.n} patents, "
              f"filters: {f.describe()}. Empty line quits.")
        while (q := input("\nquery> ").strip()):
            run(q)
    else:
        run(a.query)


if __name__ == "__main__":
    main()
