"""Shrink an index cache into a small, read-only bundle for the server.

    python export_index.py                                # cache/qwen3 -> deploy/index/qwen3 (+ .tar.gz)
    python export_index.py --src cache/minilm --dims 0    # keep all dims

What gets smaller (measured in results/compression_qwen3.md):
  * vectors truncated to --dims (Qwen3 is Matryoshka-trained; 512-d loses
    ~0.002 nDCG@10) and stored as float16
  * "(canceled)" claim chunks removed, chunk_ids renumbered
  * chunk text dropped (the server rebuilds it from patents.json when needed)
  * description paragraphs dropped from patents.json (not used for search/display)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile

import numpy as np

from build_index import load_index
from data import is_canceled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="cache/qwen3")
    ap.add_argument("--out", default=None, help="default: deploy/index/<model alias>")
    ap.add_argument("--dims", type=int, default=512, help="Matryoshka truncation; 0 = keep all")
    ap.add_argument("--no-tar", action="store_true")
    a = ap.parse_args()

    ix = load_index(a.src)
    man, E, chunks, patents = ix["manifest"], ix["embeddings"], ix["chunks"], ix["patents"]
    out = a.out or os.path.join("deploy", "index", man["model_alias"])
    os.makedirs(out, exist_ok=True)

    keep, slim_chunks = [], []
    for c in chunks:
        if c["kind"] == "claim" and is_canceled(patents[c["doc_number"]]["claims"][c["claim_no"] - 1]):
            continue
        if c["kind"] == "desc":
            continue
        keep.append(c["chunk_id"])
        s = {k: v for k, v in c.items() if k != "text" and v is not None}
        s["chunk_id"] = len(slim_chunks)
        slim_chunks.append(s)

    V = np.asarray(E[keep], dtype=np.float32)
    dims = a.dims if a.dims and a.dims < V.shape[1] else V.shape[1]
    V = V[:, :dims]
    V /= np.linalg.norm(V, axis=1, keepdims=True).clip(min=1e-12)
    np.save(os.path.join(out, "embeddings.npy"), V.astype(np.float16))

    with open(os.path.join(out, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in slim_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "patents.json"), "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in p.items() if k != "description"} for p in patents.values()],
                  f, ensure_ascii=False)

    m = dict(man)
    m.update({"exported_from": a.src, "dim": int(dims), "full_dim": int(E.shape[1]),
              "dtype": "float16", "n_chunks": len(slim_chunks), "slim": True,
              "chunk_kinds": {k: sum(c["kind"] == k for c in slim_chunks)
                              for k in {c["kind"] for c in slim_chunks}}})
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)

    size = sum(os.path.getsize(os.path.join(out, n)) for n in os.listdir(out))
    print(f"wrote {out}/: {len(slim_chunks)} chunks x {dims}-d float16, {size / 1e6:.1f} MB "
          f"(source {sum(os.path.getsize(os.path.join(a.src, n)) for n in os.listdir(a.src)) / 1e6:.1f} MB)")

    if not a.no_tar:
        tar_path = out.rstrip("/\\") + ".tar.gz"
        with tarfile.open(tar_path, "w:gz") as t:
            t.add(out, arcname=os.path.basename(out.rstrip("/\\")))
        h = hashlib.sha256(open(tar_path, "rb").read()).hexdigest()
        print(f"wrote {tar_path}: {os.path.getsize(tar_path) / 1e6:.1f} MB, sha256 {h}")


if __name__ == "__main__":
    main()
