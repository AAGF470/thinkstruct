import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { fetchMeta, type Meta } from './api'
import { CpcTree } from './components/CpcTree'
import { Chips, SearchBar } from './components/Controls'
import { Results } from './components/Results'
import { Sidebar } from './components/Sidebar'
import { Timeline } from './components/Timeline'
import {
  EMPTY, fromHash, loadLast, loadSaved, reducer, storeLast, storeSaved, toHash, type SavedView,
  type ViewState,
} from './state'
import { useExplore } from './useExplore'

function initialState(): ViewState {
  return fromHash(window.location.hash) ?? loadLast() ?? EMPTY
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, undefined, initialState)
  const [meta, setMeta] = useState<Meta | null>(null)
  const [metaErr, setMetaErr] = useState<string | null>(null)
  const [saved, setSaved] = useState<SavedView[]>(loadSaved)
  const [slow, setSlow] = useState(false)
  const [drawer, setDrawer] = useState(false)
  const slowRef = useRef(slow)
  slowRef.current = slow

  useEffect(() => { fetchMeta().then(setMeta).catch((e) => setMetaErr(String(e.message ?? e))) }, [])

  const { data, applied, pending, error, samples, staleDropped, loadMore, loadingMore } = useExplore(state, {
    delayMs: () => (slowRef.current ? Math.round(Math.random() * 1500) : 0),
  })

  // URL hash + last state follow the filter state (shareable links, reload-safe).
  const fromPop = useRef(false)
  useEffect(() => {
    storeLast(state)
    const h = toHash(state)
    if (fromPop.current) { fromPop.current = false; return }
    if (h !== window.location.hash) history.pushState(null, '', h || window.location.pathname)
  }, [state])
  useEffect(() => {
    const onPop = () => { fromPop.current = true; dispatch({ type: 'restore', state: fromHash(location.hash) ?? EMPTY }) }
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  const onSave = useCallback((name: string) => {
    const v: SavedView = { id: crypto.randomUUID(), name, state, savedAt: Date.now(), n: data?.n_results }
    setSaved((xs) => { const next = [v, ...xs].slice(0, 30); storeSaved(next); return next })
  }, [state, data])
  const onDelete = useCallback((id: string) => {
    setSaved((xs) => { const next = xs.filter((x) => x.id !== id); storeSaved(next); return next })
  }, [])

  const titles = meta?.cpc_titles ?? {}
  const busy = pending
  const summary = useMemo(() => {
    if (!data) return ''
    const t = data.timings_ms
    return `${data.n_results} patents · server ${data.server_ms.toFixed(1)} ms` +
      (t.embed ? ` (query embedding ${t.embed.toFixed(0)} ms)` : '')
  }, [data])

  return (
    <div className="shell">
      <header className="mtop">
        <div className="avatar mono">PE</div>
        <div className="mtop-name">Patent Explorer</div>
        <button className="mtop-burger" onClick={() => setDrawer(true)} aria-label="Open menu">
          <svg viewBox="0 0 24 24"><path d="M4 7h16M4 12h16M4 17h16" /></svg>
        </button>
      </header>

      <Sidebar meta={meta} state={state} saved={saved} onSave={onSave}
        onOpen={(v) => { dispatch({ type: 'restore', state: v.state }); setDrawer(false) }}
        onDelete={onDelete} samples={samples} staleDropped={staleDropped}
        slow={slow} onSlow={setSlow} open={drawer} onClose={() => setDrawer(false)} />

      <main className="main">
        <div className="wrap">
          <div className="page-head">
            <h1>Patent Explorer</h1>
            <p>
              Search {meta?.n_patents ?? '…'} vehicle-wheel and tyre patent applications by meaning,
              then narrow by classification, filing week and keywords. Every view below describes the same set.
            </p>
          </div>

          <SearchBar state={state} onQuery={(q) => dispatch({ type: 'query', q })}
            onMode={(mode) => dispatch({ type: 'mode', mode })} />

          <div className="status-row">
            <Chips state={state} titles={titles}
              onCpc={(cpc) => dispatch({ type: 'cpc', cpc })}
              onRemoveKw={(kw) => dispatch({ type: 'removeKw', kw })}
              onAddKw={(kw) => dispatch({ type: 'addKw', kw })}
              onWeeks={(from, to) => dispatch({ type: 'weeks', from, to })}
              onLike={(doc) => dispatch({ type: 'like', doc })}
              onClear={() => dispatch({ type: 'clearFilters' })} />
            <div className={`summary mono ${busy ? 'busy' : ''}`} role="status">
              {busy ? 'Updating…' : summary}
            </div>
          </div>

          {(error || metaErr) && <div className="callout err"><span className="ico">!</span><p>{error ?? metaErr}</p></div>}

          {data && applied && meta ? (
            <>
              {/* Views render the selection that produced `data` (applied), never the
                  in-flight request, so all three always describe the same set. */}
              <Timeline axis={meta.weeks} counts={data.facets.weeks} from={applied.weekFrom} to={applied.weekTo}
                busy={busy} onSelect={(from, to) => dispatch({ type: 'weeks', from, to })} />
              <div className="grid">
                <Results data={data} busy={busy} loadingMore={loadingMore} onMore={loadMore}
                  onCpc={(cpc) => dispatch({ type: 'cpc', cpc })}
                  onWeek={(w) => dispatch({ type: 'weeks', from: w, to: w })}
                  onLike={(doc) => { dispatch({ type: 'like', doc }); window.scrollTo({ top: 0, behavior: 'smooth' }) }} />
                <CpcTree counts={data.facets.cpc} titles={titles} selected={applied.cpc} busy={busy}
                  onSelect={(cpc) => dispatch({ type: 'cpc', cpc })} />
              </div>
            </>
          ) : !error && !metaErr ? (
            <div className="loading mono">Loading index…</div>
          ) : null}

          <footer className="foot">
            <span>Thinkstruct coding challenge · hybrid semantic patent search</span>
            <span className="sp" />
            <a href="api/docs">API docs</a>
          </footer>
        </div>
      </main>
    </div>
  )
}
