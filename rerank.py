"""Part 3: two-phase search. Stage 1 (search.py) -> top k1 candidates ->
cross-encoder reranker -> top k2.

    python rerank.py "tire with a sensor that monitors air pressure"
    python rerank.py "spoke connecting hub to rim" --reranker bge --cpc B60B
    python search.py "..." --rerank minilm          # same thing from the search CLI

Why a cross-encoder: the stage-1 bi-encoder embeds query and patent SEPARATELY
(fast, precomputed). A cross-encoder reads the query and one candidate TOGETHER,
so it can weigh exact wording and interactions, but it costs one model pass per
candidate, so it is only affordable on a short list.

Candidate text given to the reranker: "TITLE. abstract  Claim N: <best claim>",
truncated to the reranker's max length (512 tokens). The best claim comes from
stage 1's explain step, so the reranker sees the most relevant claim, not claim 1.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

RERANKERS = {
    # 22M params, MS MARCO web-search pairs. Fast on CPU (Raya-friendly).
    "minilm": "cross-encoder/ms-marco-MiniLM-L6-v2",
    # 568M params, multilingual, stronger. Slow on CPU; best on a GPU.
    "bge": "BAAI/bge-reranker-v2-m3",
}


def candidate_text(r: dict) -> str:
    parts = [f"{r['title']}.", r.get("abstract", "")]
    mc = r.get("matched_claim")
    if mc:
        parts.append(f"Claim {mc['claim_no']}: {mc['text']}")
    return " ".join(p for p in parts if p)


class Reranker:
    def __init__(self, name: str = "minilm", device: str = "cpu", max_length: int = 512):
        from sentence_transformers import CrossEncoder
        self.name, self.model_id = name, RERANKERS.get(name, name)
        t = time.perf_counter()
        self.model = CrossEncoder(self.model_id, device=device, max_length=max_length)
        self.load_ms = round((time.perf_counter() - t) * 1000, 1)

    def cap_query(self, query: str, max_tokens: int | None) -> str:
        """Truncate a long query (e.g. a whole claim) so the candidate keeps most of
        the reranker's input budget. The pair is truncated longest-first, so an
        uncapped 300-token claim leaves the candidate only ~200 of 512 tokens."""
        if not max_tokens:
            return query
        tok = self.model.tokenizer
        ids = tok(query, add_special_tokens=False)["input_ids"]
        return query if len(ids) <= max_tokens else tok.decode(ids[:max_tokens])

    def score(self, query: str, results: list[dict], batch_size: int = 16,
              query_max_tokens: int | None = None) -> np.ndarray:
        q = self.cap_query(query, query_max_tokens)
        pairs = [(q, candidate_text(r)) for r in results]
        return np.asarray(self.model.predict(pairs, batch_size=batch_size,
                                             show_progress_bar=False), dtype=np.float32)

    @staticmethod
    def apply(results: list[dict], scores: np.ndarray, top_n: int = 30,
              fuse: str = "replace", rrf_k: int = 60) -> list[dict]:
        """Order `results` (stage-1 order) using reranker `scores`.

        fuse="replace": rank purely by reranker score (classic two-phase).
        fuse="rrf":     Reciprocal Rank Fusion of stage-1 rank and reranker rank, so
                        an out-of-domain reranker can reorder but not overturn stage 1.
        """
        if not results:
            return []
        if fuse == "rrf":
            rr_rank = np.empty(len(scores), dtype=np.int64)
            rr_rank[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
            s1_rank = np.arange(1, len(scores) + 1)  # results arrive in stage-1 order
            key = 1.0 / (rrf_k + s1_rank) + 1.0 / (rrf_k + rr_rank)
        else:
            key = scores
        order = np.argsort(-key, kind="stable")[:top_n]
        out = []
        for new_rank, j in enumerate(order, 1):
            r = dict(results[j])
            r["stage1_rank"], r["rank"] = r["rank"], new_rank
            r["rerank_score"] = round(float(scores[j]), 4)
            out.append(r)
        return out

    # Defaults = best variant in results/rerank_eval_qwen3_minilm_k100_len512_n100_fix_w1.md
    def rerank(self, query: str, results: list[dict], top_n: int = 30, fuse: str = "rrf",
               query_max_tokens: int | None = 64) -> list[dict]:
        if not results:
            return []
        s = self.score(query, results, query_max_tokens=query_max_tokens)
        return self.apply(results, s, top_n=top_n, fuse=fuse)


def two_phase(engine, reranker: Reranker, query: str, filters=None, k1: int = 100,
              k2: int = 30, fuse: str = "rrf", query_max_tokens: int | None = 64,
              **search_kw) -> dict:
    """Stage-1 search for k1 candidates, rerank, keep k2. Timings include both."""
    res = engine.search(query, filters, k=k1, **search_kw)
    t = time.perf_counter()
    qtext = query if query else res["results"][0]["title"] if res["results"] else ""
    res["results"] = reranker.rerank(qtext, res["results"], top_n=k2, fuse=fuse,
                                     query_max_tokens=query_max_tokens)
    res["timings_ms"]["rerank"] = round((time.perf_counter() - t) * 1000, 3)
    res["timings_ms"]["total"] = round(res["timings_ms"]["total"] + res["timings_ms"]["rerank"], 3)
    res["mode"] = f"{res['mode']}+rerank({reranker.name})"
    res["stage1_k"] = k1
    return res


def main():
    from search import Filters, SearchEngine, _default_index, _print
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Two-phase search: stage 1 + cross-encoder rerank")
    ap.add_argument("query")
    ap.add_argument("--index", default=_default_index())
    ap.add_argument("--reranker", default="minilm", help=f"{', '.join(RERANKERS)} or a HF id")
    ap.add_argument("--k1", type=int, default=100, help="stage-1 candidates")
    ap.add_argument("--k2", type=int, default=30, help="results kept after rerank")
    ap.add_argument("--show", type=int, default=10)
    ap.add_argument("--cpc", action="append", default=[])
    ap.add_argument("--kw", action="append", default=[])
    ap.add_argument("--fuse", default="rrf", choices=["replace", "rrf"])
    ap.add_argument("--query-max-tokens", type=int, default=64, help="0 = uncapped")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    eng = SearchEngine(a.index, device=a.device)
    rr = Reranker(a.reranker, device=a.device)
    res = two_phase(eng, rr, a.query, Filters(cpc=a.cpc, keywords=a.kw), k1=a.k1, k2=a.k2,
                    fuse=a.fuse, query_max_tokens=a.query_max_tokens or None)
    res["results"] = res["results"][:a.show]
    _print(res)
    print("\nrank movement (stage 1 -> reranked): " +
          ", ".join(f"{r['stage1_rank']}->{r['rank']}" for r in res["results"]))


if __name__ == "__main__":
    main()
