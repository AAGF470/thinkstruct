"""Search API + minimal search page over a (slim) index bundle.

    # local
    INDEX_DIR=deploy/index/qwen3 uvicorn app.main:app --port 8000
    # container: see deploy/

Endpoints (all GET):
    /                      search page (static)
    /api/health            index + model info, startup timings
    /api/search            q, cpc[], kw[], kw_field, title, week_from, week_to, k, mode,
                           rerank, like, like_claim  -> same JSON as SearchEngine.search()
    /api/patent/{doc}      full metadata + claims for one patent
    /api/explore           linked-views query: results + CPC + week facets (same set)
    /api/meta              corpus week axis, CPC totals, CPC titles
    /simple                the original single-file search page
    /api/docs              OpenAPI UI

Environment:
    INDEX_DIR      index bundle (default deploy/index/qwen3)
    DEVICE         cpu (default) | cuda
    ENABLE_RERANK  1 to allow ?rerank=true (loads the reranker on first use)
    RERANKER       minilm (default) | bge | HF id
    EMBED_DTYPE    float32 to override the model's default dtype (see build_index.py)
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from search import Filters, SearchEngine

INDEX_DIR = os.environ.get("INDEX_DIR", os.path.join("deploy", "index", "qwen3"))
DEVICE = os.environ.get("DEVICE", "cpu")
ENABLE_RERANK = os.environ.get("ENABLE_RERANK", "0") == "1"
RERANKER = os.environ.get("RERANKER", "minilm")
MAX_K, MAX_Q = 50, 2000
STATIC = os.path.join(os.path.dirname(__file__), "static")
# Testing aid for the UI's stale-response guard: lets /api/explore sleep `delay_ms`.
ALLOW_DEBUG_DELAY = os.environ.get("ALLOW_DEBUG_DELAY", "0") == "1"

# CPC titles for the levels present in this corpus (section B, class B60, its subclasses).
CPC_TITLES = {
    "B": "Performing operations; transporting",
    "B60": "Vehicles in general",
    "B60B": "Vehicle wheels; castors; axles",
    "B60C": "Vehicle tyres; tyre inflation and changing",
    "B60D": "Vehicle connections (towing, coupling)",
    "B60F": "Rail/road and amphibious vehicles",
    "B60G": "Vehicle suspension",
    "B60J": "Vehicle windows, roofs, doors",
    "B60R": "Vehicle fittings not otherwise provided for",
    "N/A": "No classification code",
}

state: dict = {}
lock = threading.Lock()  # one model instance; serialise inference (low-QPS demo)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    t = time.perf_counter()
    eng = SearchEngine(INDEX_DIR, device=DEVICE)
    t_index = time.perf_counter() - t
    t = time.perf_counter()
    eng.model()
    eng.embed_query("warm up")
    state.update(engine=eng, reranker=None, started_at=time.time(),
                 startup_s={"index": round(t_index, 2), "model": round(time.perf_counter() - t, 2)})
    yield


app = FastAPI(title="Patent search", lifespan=lifespan,
              docs_url="/api/docs", openapi_url="/api/openapi.json")


def _reranker():
    if state["reranker"] is None:
        from rerank import Reranker
        state["reranker"] = Reranker(RERANKER, device=DEVICE)
    return state["reranker"]


@app.get("/api/health")
def health():
    eng: SearchEngine = state["engine"]
    m = eng.manifest
    return {"status": "ok", "model": m["model_id"], "dim": int(eng.E.shape[1]),
            "n_patents": eng.n, "n_chunks": len(eng.chunks), "index_dir": INDEX_DIR,
            "index_created_at": m.get("created_at"), "rerank_enabled": ENABLE_RERANK,
            "reranker": RERANKER if ENABLE_RERANK else None, "startup_s": state["startup_s"],
            "uptime_s": round(time.time() - state["started_at"])}


@app.get("/api/search")
def search(
    q: Annotated[str | None, Query(max_length=MAX_Q)] = None,
    cpc: Annotated[list[str], Query()] = [],
    kw: Annotated[list[str], Query()] = [],
    kw_field: Annotated[str, Query(pattern="^(any|title|abstract)$")] = "any",
    title: str | None = None,
    week_from: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
    week_to: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
    k: Annotated[int, Query(ge=1, le=MAX_K)] = 10,
    mode: Annotated[str, Query(pattern="^(hybrid|dense|bm25)$")] = "hybrid",
    rerank: bool = False,
    like: str | None = None,
    like_claim: Annotated[str | None, Query(pattern=r"^\d+:\d+$")] = None,
):
    eng: SearchEngine = state["engine"]
    f = Filters(cpc=[c for c in cpc if c.strip()], keywords=[x for x in kw if x.strip()],
                kw_field=kw_field, title=title or None, week_from=week_from, week_to=week_to)
    q = (q or "").strip() or None
    if rerank and not ENABLE_RERANK:
        raise HTTPException(400, "reranking is disabled on this server (ENABLE_RERANK=0)")
    t = time.perf_counter()
    try:
        with lock:
            if rerank and (q or like or like_claim):
                from rerank import two_phase
                res = two_phase(eng, _reranker(), q, f, k1=100, k2=k, mode=mode,
                                like=like, like_claim=like_claim)
            else:
                res = eng.search(q, f, k=k, mode=mode, like=like, like_claim=like_claim)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))
    except ValueError as e:
        raise HTTPException(400, str(e))
    res["request_ms"] = round((time.perf_counter() - t) * 1000, 1)
    return res


@app.get("/api/meta")
def meta():
    m = state["engine"].meta()
    m["cpc_titles"] = CPC_TITLES
    m["rerank_enabled"] = ENABLE_RERANK
    m["debug_delay"] = ALLOW_DEBUG_DELAY
    return m


@app.get("/api/explore")
def explore(
    q: Annotated[str | None, Query(max_length=MAX_Q)] = None,
    cpc: Annotated[list[str], Query()] = [],
    kw: Annotated[list[str], Query()] = [],
    week_from: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
    week_to: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")] = None,
    k: Annotated[int, Query(ge=1, le=MAX_K)] = 20,
    offset: Annotated[int, Query(ge=0, le=1000)] = 0,
    match_n: Annotated[int, Query(ge=10, le=1000)] = 200,
    mode: Annotated[str, Query(pattern="^(hybrid|dense|bm25)$")] = "hybrid",
    like: str | None = None,
    rid: Annotated[str | None, Query(max_length=64)] = None,
    delay_ms: Annotated[int, Query(ge=0, le=5000)] = 0,
):
    """One response = results page + CPC counts + week counts for the SAME filtered
    set (see SearchEngine.explore), so the UI's views can never disagree. `rid` is
    echoed back so the client can discard out-of-order responses."""
    eng: SearchEngine = state["engine"]
    if delay_ms and ALLOW_DEBUG_DELAY:
        time.sleep(delay_ms / 1000)
    f = Filters(cpc=[c for c in cpc if c.strip()], keywords=[x for x in kw if x.strip()],
                week_from=week_from, week_to=week_to)
    t = time.perf_counter()
    try:
        with lock:
            res = eng.explore((q or "").strip() or None, f, k=k, offset=offset,
                              match_n=match_n, mode=mode, like=like)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))
    res["rid"] = rid
    res["server_ms"] = round((time.perf_counter() - t) * 1000, 2)
    return res


@app.get("/api/patent/{doc_number}")
def patent(doc_number: str):
    eng: SearchEngine = state["engine"]
    i = eng.doc_idx.get(doc_number)
    if i is None:
        raise HTTPException(404, f"unknown doc_number {doc_number}")
    return eng.docs[i]


@app.get("/", include_in_schema=False)
def index():
    """The linked-views UI (built from web/ into app/static/). Falls back to the
    simple single-file page if the UI has not been built."""
    built = os.path.join(STATIC, "index.html")
    return FileResponse(built if os.path.exists(built) else os.path.join(STATIC, "simple.html"))


@app.get("/simple", include_in_schema=False)
def simple():
    return FileResponse(os.path.join(STATIC, "simple.html"))


app.mount("/assets", StaticFiles(directory=os.path.join(STATIC, "assets"), check_dir=False),
          name="assets")
