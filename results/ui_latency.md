# Linked-views UI: update latency and consistency (small set, 640 patents)

**Setup:**
- Machine: the laptop, with the API (`uvicorn app.main:app`, `deploy/index/qwen3` 512-d bundle, CPU) and the browser on the same machine.
- **Query:** "valve for inflating a tire" (match set = top 200).
- **Filter changes:** 24, driven through the real UI. Each round adds a keyword chip, clicks CPC B60C, goes back to All, then removes the keyword.
- **Measurement:** `useExplore.ts`, from the state change to React committing the new response to the DOM (`commitMs`). The raw samples are in `window.__uiPerf` in the browser.

## Update latency

| update type | n | p50 | p95 | network round trip (p50) | server time (p50) |
|---|---|---|---|---|---|
| needs the server (new filter state) | 12 | **36 ms** | 56 ms | 31 ms | 5 ms |
| answered from the browser cache (state seen before) | 12 | **7 ms** | 14 ms | – | – |

- **Paint time.** Time to the next painted frame adds about one frame (8–16 ms) when the tab is visible. In this run the pane was in the background, where browsers throttle frames to about 1 Hz, so the paint numbers (about 1 s) are not representative and are left out.
- **Server time** (`/api/explore`, about 5 ms) covers ranking 640 patents, filtering, counting both facets, and the explain step for the first page. The query embedding is cached after the first request for a query (about 130 ms on CPU the first time).

## Consistency under fast filter changes

With the simulated slow server on (`ALLOW_DEBUG_DELAY=1`, so each request is randomly delayed 0–1.5 s and responses come back out of order), each round made 7 filter changes in about 150 ms. The views were sampled every 20 ms:

| round | samples where the views disagreed | final state = last action? | superseded requests discarded |
|---|---|---|---|
| 1 | 0 / 63 | yes (`kw=spoke`) | 3 |
| 2 | 0 / 69 | yes (`kw=noise`) | 3 |
| 3 | 0 / 34 | yes (`kw=sensor`) | 3 |

"Disagreed" means the results header, timeline total and CPC row total weren't all equal.

**A bug this test found and fixed.** The first version drew the CPC drill-down from the *requested* selection but the *previous* response's counts, so 21 of 33 mid-flight samples disagreed. Views now render from the state that produced their data (`applied` in `useExplore.ts`).

## What would change at 10M patents
- **Facets can't be counted in Python over the match set.** They'd come from the search backend's aggregations: Postgres `GROUP BY` on indexed `cpc_group` / `week` columns over the filtered match set, or OpenSearch-style aggregations.
- **The match set would be an approximate-index top-N** (HNSW, for example top 1,000) rather than an exact ranking.
- **Week buckets would become precomputed rollups,** e.g. a materialized view per CPC subclass × week, for the browse mode with no query.
- **The client design stays the same:** one response carries results and all facets, a request ID guards against stale responses, and there's a response cache. Server time would rise from about 5 ms to tens of ms. The extra latency would be absorbed by keeping the previous view visible but dimmed, and by caching recent filter states as now.
