// Fetches the linked-views payload for the current state.
//
// Stale-result guarantees:
//   * every request gets an increasing id (rid); the server echoes it back
//   * a new state aborts the in-flight request (AbortController)
//   * a response is rendered only if its rid is still the latest one, so a slow
//     older response can never overwrite a newer one, however they interleave
//   * all three views render from the ONE response object, so they always agree
//
// Update latency = time from the user's action (state change) to the frame after
// React commits the new response; samples are kept in window.__uiPerf.

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { fetchExplore, keyOf, type ExploreParams, type ExploreResponse } from './api'

export interface PerfSample {
  commitMs: number    // action -> React has written the new data to the DOM
  ms: number          // action -> next frame after the new data is committed (paint;
                      // inflated if the tab is in the background and frames are throttled)
  serverMs: number    // time spent in the search engine
  networkMs: number   // fetch round trip (includes serverMs)
  source: 'network' | 'cache'
  nResults: number
  at: number
}

declare global {
  interface Window { __uiPerf?: PerfSample[]; __uiStaleDropped?: number }
}

const CACHE_MAX = 80
const cache = new Map<string, ExploreResponse>()
function cacheSet(key: string, v: ExploreResponse) {
  cache.delete(key)
  cache.set(key, v)
  if (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value as string)
}

export function useExplore(state: ExploreParams, opts: { delayMs?: () => number } = {}) {
  const key = keyOf(state)
  const [data, setData] = useState<ExploreResponse | null>(null)
  // The state that PRODUCED `data`. Views must render selections from this, not from
  // the live (requested) state: while a request is in flight the live state is ahead
  // of the data, and mixing the two makes the views disagree (caught by the UI test).
  const [applied, setApplied] = useState<ExploreParams | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [samples, setSamples] = useState<PerfSample[]>([])
  const [staleDropped, setStaleDropped] = useState(0)

  const seq = useRef(0)
  const ctrl = useRef<AbortController | null>(null)
  const actionAt = useRef(performance.now())
  const pendingSample = useRef<Omit<PerfSample, 'ms' | 'commitMs' | 'at'> | null>(null)
  const stateRef = useRef(state)
  stateRef.current = state

  // The state changed: this is the user's action time for the latency measurement.
  useLayoutEffect(() => { actionAt.current = performance.now() }, [key])

  useEffect(() => {
    const my = ++seq.current
    const rid = String(my)
    ctrl.current?.abort()
    const hit = cache.get(key)
    const requested = stateRef.current
    if (hit) {
      pendingSample.current = { serverMs: 0, networkMs: 0, source: 'cache', nResults: hit.n_results }
      setData(hit)
      setApplied(requested)
      setPending(false)
      setError(null)
      return
    }
    const c = new AbortController()
    ctrl.current = c
    setPending(true)
    const t0 = performance.now()
    fetchExplore(requested, { rid, delayMs: opts.delayMs?.(), signal: c.signal })
      .then((res) => {
        if (my !== seq.current || res.rid !== rid) {   // an older answer arriving late
          window.__uiStaleDropped = (window.__uiStaleDropped ?? 0) + 1
          setStaleDropped((n) => n + 1)
          return
        }
        cacheSet(key, res)
        pendingSample.current = {
          serverMs: res.server_ms, networkMs: performance.now() - t0, source: 'network',
          nResults: res.n_results,
        }
        setData(res)
        setApplied(requested)
        setPending(false)
        setError(null)
      })
      .catch((e: unknown) => {
        if ((e as Error).name === 'AbortError') {
          window.__uiStaleDropped = (window.__uiStaleDropped ?? 0) + 1
          setStaleDropped((n) => n + 1)
          return
        }
        if (my !== seq.current) return
        setError((e as Error).message)
        setPending(false)
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  // After React commits new data, wait one frame (paint) and record the sample.
  useLayoutEffect(() => {
    const s = pendingSample.current
    if (!s || !data) return
    pendingSample.current = null
    const start = actionAt.current
    const commitMs = performance.now() - start
    requestAnimationFrame(() => {
      const sample: PerfSample = { ...s, commitMs, ms: performance.now() - start, at: Date.now() }
      window.__uiPerf = [...(window.__uiPerf ?? []), sample].slice(-500)
      setSamples((xs) => [...xs, sample].slice(-200))
    })
  }, [data])

  const [loadingMore, setLoadingMore] = useState(false)
  const loadMore = useCallback(async () => {
    if (!data || loadingMore || data.results.length >= data.n_results) return
    const forKey = key
    const rid = `${seq.current}-more`
    setLoadingMore(true)
    try {
      const page = await fetchExplore(stateRef.current, { rid, offset: data.results.length, k: 20 })
      if (keyOf(stateRef.current) !== forKey) return    // state moved on: drop the page
      setData((d) => {
        if (!d) return d
        const merged = { ...d, results: [...d.results, ...page.results] }
        cacheSet(forKey, merged)
        return merged
      })
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoadingMore(false)
    }
  }, [data, key, loadingMore])

  return { data, applied, pending, error, samples, staleDropped, loadMore, loadingMore }
}
