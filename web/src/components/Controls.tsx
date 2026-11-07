import { useEffect, useRef, useState } from 'react'
import type { Mode } from '../api'
import { fmtRange } from '../lib'
import type { ViewState } from '../state'

export function SearchBar({ state, onQuery, onMode }: {
  state: ViewState
  onQuery: (q: string) => void
  onMode: (m: Mode) => void
}) {
  const [text, setText] = useState(state.q)
  const ref = useRef<HTMLInputElement>(null)
  useEffect(() => setText(state.q), [state.q])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === '/' && document.activeElement?.tagName !== 'INPUT') {
        e.preventDefault()
        ref.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
  const modes: [Mode, string][] = [['hybrid', 'Hybrid'], ['dense', 'Meaning'], ['bm25', 'Keywords']]
  return (
    <form className="search" role="search" onSubmit={(e) => { e.preventDefault(); onQuery(text) }}>
      <label className="search-box">
        <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></svg>
        <input
          ref={ref}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Describe an invention, or paste a patent claim"
          aria-label="Search query"
        />
        <kbd>/</kbd>
      </label>
      <div className="seg" role="radiogroup" aria-label="Ranking mode">
        {modes.map(([m, label]) => (
          <button key={m} type="button" role="radio" aria-checked={state.mode === m}
            className={state.mode === m ? 'on' : ''} onClick={() => onMode(m)}>{label}</button>
        ))}
      </div>
      <button className="btn btn-solid" type="submit">Search</button>
    </form>
  )
}

export function Chips({ state, titles, onCpc, onRemoveKw, onAddKw, onWeeks, onLike, onClear }: {
  state: ViewState
  titles: Record<string, string>
  onCpc: (c: string | null) => void
  onRemoveKw: (k: string) => void
  onAddKw: (k: string) => void
  onWeeks: (a: string | null, b: string | null) => void
  onLike: (d: string | null) => void
  onClear: () => void
}) {
  const [kw, setKw] = useState('')
  const any = state.cpc || state.kw.length || state.weekFrom || state.weekTo
  return (
    <div className="chips" aria-label="Active filters">
      {state.like && (
        <Chip k="similar to" v={state.like} onRemove={() => onLike(null)} />
      )}
      {state.cpc && <Chip k="cpc" v={state.cpc} title={titles[state.cpc.slice(0, 4)]} onRemove={() => onCpc(null)} />}
      {(state.weekFrom || state.weekTo) && (
        <Chip k="filed" v={fmtRange(state.weekFrom, state.weekTo)} onRemove={() => onWeeks(null, null)} />
      )}
      {state.kw.map((k) => <Chip key={k} k="keyword" v={k} onRemove={() => onRemoveKw(k)} />)}
      <form className="chip-add" onSubmit={(e) => { e.preventDefault(); onAddKw(kw); setKw('') }}>
        <input value={kw} onChange={(e) => setKw(e.target.value)} placeholder="+ keyword in title/abstract"
          aria-label="Add keyword filter" />
      </form>
      {any && <button className="link-btn" onClick={onClear}>Clear filters</button>}
    </div>
  )
}

function Chip({ k, v, title, onRemove }: { k: string; v: string; title?: string; onRemove: () => void }) {
  return (
    <span className="chip" title={title}>
      <span className="chip-k">{k}</span>
      <span className="chip-v">{v}</span>
      <button onClick={onRemove} aria-label={`Remove ${k} filter ${v}`}>×</button>
    </span>
  )
}
