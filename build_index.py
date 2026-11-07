"""Offline indexing: patents -> chunks -> embeddings -> a portable cache folder.

    python build_index.py                     # Qwen3-Embedding-0.6B (default)
    python build_index.py --model minilm      # fast baseline
    python build_index.py --model qwen3 --device cuda --with-description

The output folder (cache/<model alias>/) is self-contained so it can be copied
to another machine (e.g. the Raya server) and served with CPU only:

    embeddings.npy   float32 [n_chunks, dim], L2-normalised; row i == chunks[i]
    chunks.jsonl     one Chunk per line (text + which patent/claim it came from)
    patents.json     patent metadata (title, abstract, claims, cpc, week, ...)
    manifest.json    model id, query prompt, dims, counts, timings, load report

Search must embed queries with the SAME model and query prompt recorded in
manifest.json, otherwise the vectors are not comparable.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import time
from datetime import datetime, timezone

import numpy as np

from data import DEFAULT_DATA_DIR, chunk_patents, load_patents

# Retrieval-tuned query instruction for Qwen3 (documents are embedded without one).
_QWEN_QUERY = ("Instruct: Given a patent search query, retrieve patents whose claims "
               "or abstract describe the queried invention\nQuery: ")

MODELS = {
    "minilm": {"id": "sentence-transformers/all-MiniLM-L6-v2", "max_seq_length": 256,
               "query_prompt": ""},
    "qwen3": {"id": "Qwen/Qwen3-Embedding-0.6B", "max_seq_length": 2048,
              "query_prompt": _QWEN_QUERY},
}
DEFAULT_MODEL = "qwen3"


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(alias_or_id: str, device: str = "cpu", max_seq_length: int | None = None):
    """Shared by indexing and search so both sides configure the model identically."""
    from sentence_transformers import SentenceTransformer
    spec = MODELS.get(alias_or_id, {"id": alias_or_id, "max_seq_length": 512,
                                    "query_prompt": ""})
    # Qwen3 loads in bfloat16 by default. On CPUs with native bf16 (this laptop's
    # Ryzen) that is ~2x faster than float32 (162 vs 319 ms/query); on older CPUs
    # it can be much slower. Override with EMBED_DTYPE=float32 and benchmark on Raya.
    dtype = os.environ.get("EMBED_DTYPE")
    kwargs = {"model_kwargs": {"dtype": dtype}} if dtype else {}
    model = SentenceTransformer(spec["id"], device=device, **kwargs)
    model.max_seq_length = min(max_seq_length or spec["max_seq_length"],
                               model.max_seq_length or 10**9)
    return model, spec


def build(args) -> str:
    t0 = time.perf_counter()
    patents, report = load_patents(args.data, limit=args.limit)
    chunks = chunk_patents(patents, with_description=args.with_description)
    t_parse = time.perf_counter() - t0
    print(f"[parse] {report.patents_kept} patents -> {len(chunks)} chunks "
          f"in {t_parse:.2f}s  (failed files: {len(report.files_failed)}, "
          f"dupes: {report.duplicates_dropped}, empty: {report.patents_skipped_empty})")

    device = pick_device(args.device)
    t1 = time.perf_counter()
    model, spec = load_model(args.model, device, args.max_seq_length)
    t_load = time.perf_counter() - t1
    print(f"[model] {spec['id']} on {device}, max_seq_length={model.max_seq_length}, "
          f"loaded in {t_load:.1f}s")

    # Embed per kind so the progress bar and timings say where the time goes.
    texts = [c.text for c in chunks]
    t2 = time.perf_counter()
    emb = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    # Re-normalise in float32: bf16 models on GPU leave norms ~1 +/- 0.005.
    emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)
    t_embed = time.perf_counter() - t2
    print(f"[embed] {len(texts)} chunks -> {emb.shape} in {t_embed:.1f}s "
          f"({len(texts) / t_embed:.0f} chunks/s)")

    out = args.out or os.path.join("cache", args.model if args.model in MODELS
                                   else args.model.replace("/", "__"))
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "embeddings.npy"), emb)
    with open(os.path.join(out, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    with open(os.path.join(out, "patents.json"), "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in patents], f, ensure_ascii=False)

    kinds: dict[str, int] = {}
    for c in chunks:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_alias": args.model, "model_id": spec["id"],
        "query_prompt": spec["query_prompt"], "dim": int(emb.shape[1]),
        "max_seq_length": model.max_seq_length, "normalized": True, "dtype": "float32",
        "n_patents": len(patents), "n_chunks": len(chunks), "chunk_kinds": kinds,
        "with_description": args.with_description, "data_dir": args.data,
        "load_report": dataclasses.asdict(report),
        "timings_s": {"parse": round(t_parse, 2), "model_load": round(t_load, 2),
                      "embed": round(t_embed, 2),
                      "total": round(time.perf_counter() - t0, 2)},
        "machine": {"device": device, "platform": platform.platform(),
                    "python": platform.python_version()},
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[save] wrote {out}/ ({emb.nbytes / 1e6:.1f} MB of vectors), "
          f"total {manifest['timings_s']['total']}s")
    return out


def load_index(cache_dir: str) -> dict:
    """Load a cache folder written by build(); used by search.py and the web app."""
    with open(os.path.join(cache_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    with open(os.path.join(cache_dir, "chunks.jsonl"), encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    with open(os.path.join(cache_dir, "patents.json"), encoding="utf-8") as f:
        patents = {p["doc_number"]: p for p in json.load(f)}
    emb = np.load(os.path.join(cache_dir, "embeddings.npy"), mmap_mode="r")
    assert emb.shape[0] == len(chunks), "embeddings and chunks are out of sync"
    # Slim exports store float16; score in float32.
    return {"manifest": manifest, "chunks": chunks, "patents": patents,
            "embeddings": np.asarray(emb, dtype=np.float32)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"alias ({', '.join(MODELS)}) or any sentence-transformers id")
    ap.add_argument("--data", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", default=None, help="default: cache/<model>")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, default=None)
    ap.add_argument("--with-description", action="store_true",
                    help="also embed detailed-description paragraphs (~5x more chunks)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N patents")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
