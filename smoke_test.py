"""End-to-end check of a running search API: can it find things we already know about?

    python smoke_test.py                                   # local: http://127.0.0.1:8000
    python smoke_test.py --url http://raya:8080 --user reviewer   # prompts for the password
    python smoke_test.py --quick                           # skip the 640-patent self-retrieval

Uses only the HTTP API (no imports from this repo), so it tests what a user gets.

  1. known items:   paraphrased descriptions (written by hand, avoiding title words)
                    of 12 specific patents -> where does the target rank?
  2. self-retrieval: each patent's own abstract / first claim as the query -> rank 1?
  3. filters:       every returned patent must satisfy the CPC / keyword / date filter
  4. more-like-this: never returns the source patent; neighbours share its CPC group?
  5. latency:       p50 / p95 per request, measured by the client
"""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import re
import statistics as st
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

KNOWN = [  # (target doc_number, query in everyday words)
    ("20250058592", "controller reads the pressure and opens a valve to push air from the truck's "
                    "air supply into a tire that is losing air"),
    ("20240399787", "carbon composite bicycle spoke in the shape of the letter V, bolted to the rim "
                    "at its point and to both hub flanges at its two arms"),
    ("20250115081", "tyre with small bumps in the circumferential grooves to stop the whistling "
                    "resonance of air trapped in the groove"),
    ("20240083199", "inflatable bladder inside the tire that blows up automatically when a puncture "
                    "is detected so the car can keep driving"),
    ("20240367465", "wheel sensor module that transmits on two different radio bands and protocols "
                    "so it works with several car makers' receivers"),
    ("20240359503", "airless wheel whose flexible ribs bend under load and lean on the neighbouring "
                    "rib for extra support"),
    ("20240140145", "pressure sensor where the coin battery sits inside the valve stem so it can be "
                    "swapped without taking the tire off"),
    ("20240198722", "composite fibre spoke with a pre-made slotted end that a threaded nipple slides "
                    "over for fastening"),
    ("20240051333", "spoke made of a central shaft with textured ends for grip, each end gripped by "
                    "a separate connector piece"),
    ("20240246357", "strengthening a metal spoke by hammering the neck under its head several times "
                    "to make it thicker and denser"),
    ("20240326519", "estimate how worn the tread is from acceleration peaks measured by the module "
                    "that also checks inflation"),
    ("20250001807", "tire that can be mounted facing either way and still perform well, using small "
                    "slits cut into the tread blocks"),
]

TOKEN = re.compile(r"[a-z0-9]+")


class Client:
    def __init__(self, url: str, user: str | None, password: str | None):
        self.url = url.rstrip("/")
        self.auth = ("Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
                     if user else None)
        self.lat: list[float] = []

    def get(self, path: str, **params):
        q = urllib.parse.urlencode([(k, x) for k, v in params.items() if v is not None
                                    for x in (v if isinstance(v, list) else [v])])
        req = urllib.request.Request(f"{self.url}{path}?{q}")
        if self.auth:
            req.add_header("Authorization", self.auth)
        t = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.load(r)
        self.lat.append((time.perf_counter() - t) * 1000)
        return body


def rank_of(results, doc):
    return next((r["rank"] for r in results if r["doc_number"] == doc), None)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--user", default=None)
    ap.add_argument("--quick", action="store_true", help="skip the full self-retrieval pass")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pw = None
    if a.user:
        pw = os.environ.get("SMOKE_PASSWORD") or getpass.getpass(f"password for {a.user}: ")
    c = Client(a.url, a.user, pw)
    L: list[str] = []
    fails: list[str] = []

    h = c.get("/api/health")
    L += [f"# Smoke test: {a.url}\n",
          f"- {h['n_patents']} patents, {h['model']} ({h['dim']}-d), index built {h['index_created_at']}\n"]

    # 1. known items -----------------------------------------------------------------
    L += ["## 1. Known items: paraphrased descriptions\n",
          "| target | title | hybrid | dense | bm25 |", "|---|---|---|---|---|"]
    ranks = {m: [] for m in ("hybrid", "dense", "bm25")}
    for doc, q in KNOWN:
        title = c.get(f"/api/patent/{doc}")["title"]
        row = []
        for m in ranks:
            rk = rank_of(c.get("/api/search", q=q, k=30, mode=m)["results"], doc)
            ranks[m].append(rk)
            row.append(f"#{rk}" if rk else "–")
        L.append(f"| {doc} | {title[:48]} | " + " | ".join(row) + " |")

    def summ(rs):
        n = len(rs)
        return (f"hit@1 {sum(r == 1 for r in rs)}/{n}, hit@3 {sum(bool(r) and r <= 3 for r in rs)}/{n}, "
                f"hit@10 {sum(bool(r) and r <= 10 for r in rs)}/{n}, "
                f"MRR {st.mean((1 / r) if r else 0 for r in rs):.2f}")
    L.append("")
    for m, rs in ranks.items():
        L.append(f"- **{m}**: {summ(rs)}")
    if sum(bool(r) and r <= 10 for r in ranks["hybrid"]) < len(KNOWN) * 0.75:
        fails.append("hybrid finds fewer than 75% of known items in the top 10")

    # 2. self-retrieval --------------------------------------------------------------
    L.append("\n## 2. Self-retrieval: a patent's own text as the query\n")
    # No "list all" endpoint (k <= 50), so enumerate via cheap metadata lookups, one
    # filing week at a time (~10 patents/week in the small set; no model calls).
    import datetime as dt
    all_docs, seen, day = [], set(), dt.date(2024, 1, 1)
    while day <= dt.date(2026, 12, 31) and len(all_docs) < h["n_patents"]:
        lo, hi = day.isoformat(), (day + dt.timedelta(days=6)).isoformat()
        for x in c.get("/api/search", week_from=lo, week_to=hi, k=50)["results"]:
            if x["doc_number"] not in seen:
                seen.add(x["doc_number"])
                all_docs.append(x["doc_number"])
        day += dt.timedelta(days=7)
    if a.quick:
        all_docs = all_docs[::16]  # every 16th patent (~40)
    L.append(f"Patents tested: {len(all_docs)}\n")
    L += ["| query = | rank 1 | top 3 | top 10 | misses (rank > 10) |", "|---|---|---|---|---|"]
    misses_detail = []
    for field in ("abstract", "first claim"):
        rs = []
        for doc in all_docs:
            p = c.get(f"/api/patent/{doc}")
            q = p["abstract"] if field == "abstract" else next(
                (x for x in p["claims"] if len(x.split()) >= 15 and "cancel" not in x.lower()), None)
            if not q:
                continue
            res = c.get("/api/search", q=q[:2000], k=10)["results"]
            rk = rank_of(res, doc)
            rs.append(rk)
            if not rk or rk > 3:
                misses_detail.append((field, doc, p["title"], rk, res[0]["doc_number"], res[0]["title"]))
        n = len(rs)
        L.append(f"| {field} | {sum(r == 1 for r in rs)}/{n} | {sum(bool(r) and r <= 3 for r in rs)}/{n} | "
                 f"{sum(bool(r) for r in rs)}/{n} | {sum(not r for r in rs)} |")
        if n and sum(bool(r) for r in rs) / n < 0.95:
            fails.append(f"self-retrieval by {field}: under 95% in top 10")
    if misses_detail:
        L.append("\nNot in the top 3 (what ranked #1 instead):\n")
        for f_, d, t, rk, d1, t1 in misses_detail[:15]:
            L.append(f"- {f_}: {d} *{t[:50]}* → rank {rk or '>10'}; #1 was {d1} *{t1[:50]}*")

    # 3. filters ---------------------------------------------------------------------
    L.append("\n## 3. Filter correctness\n")
    checks = [
        ("cpc=B60B", dict(q="wheel", cpc="B60B", k=30), lambda r: r["cpc"].startswith("B60B")),
        ("cpc=B60C 23/", dict(q="sensor", cpc="B60C 23/", k=30), lambda r: r["cpc"].startswith("B60C 23/")),
        ("kw=sensor", dict(q="tire", kw="sensor", k=30),
         lambda r: "sensor" in TOKEN.findall((r["title"] + " " + r["abstract"]).lower())),
        ("week>=2025-01-01", dict(q="tire", week_from="2025-01-01", k=30), lambda r: r["week"] >= "2025-01-01"),
        ("title exact", dict(title="LOW NOISE TIRE", k=30), lambda r: r["title"].lower() == "low noise tire"),
    ]
    for name, params, ok in checks:
        res = c.get("/api/search", **params)["results"]
        bad = [r["doc_number"] for r in res if not ok(r)]
        L.append(f"- {name}: {len(res)} results, {len(bad)} violate the filter" + (f" {bad[:3]}" if bad else ""))
        if bad:
            fails.append(f"filter {name} returned non-matching results")

    # 4. more-like-this --------------------------------------------------------------
    L.append("\n## 4. More like this\n")
    same, total, self_hits = 0, 0, 0
    for doc, _ in KNOWN:
        src = c.get(f"/api/patent/{doc}")
        g = src["cpc"]["display"].split("/")[0]
        res = c.get("/api/search", like=doc, k=5)["results"]
        self_hits += any(r["doc_number"] == doc for r in res)
        same += sum(r["cpc"].split("/")[0] == g for r in res)
        total += len(res)
    L.append(f"- source patent returned: {self_hits} times (expected 0)")
    L.append(f"- neighbours in the same CPC main group: {same}/{total} ({same / max(total, 1):.0%})")
    if self_hits:
        fails.append("more-like-this returned the source patent")

    # 5. latency ---------------------------------------------------------------------
    lat = sorted(c.lat)
    L.append(f"\n## 5. Latency (client-side, {len(lat)} requests)\n")
    L.append(f"- p50 {st.median(lat):.0f} ms, p95 {lat[int(0.95 * (len(lat) - 1))]:.0f} ms, "
             f"max {lat[-1]:.0f} ms")

    L.append("\n## Verdict\n")
    L.append("**PASS**" if not fails else "**FAIL**\n\n" + "\n".join(f"- {f}" for f in fails))
    text = "\n".join(L) + "\n"
    print(text)
    host = urllib.parse.urlparse(a.url).hostname.replace(".", "_")
    out = a.out or os.path.join("results", f"smoke_{host}.md")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {out}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
