import { useState } from 'react'
import type { ExploreResponse, Result } from '../api'
import { fmtWeek, highlight } from '../lib'

interface Props {
  data: ExploreResponse
  busy: boolean
  onCpc: (cpc: string) => void
  onWeek: (week: string) => void
  onLike: (doc: string) => void
  onMore: () => void
  loadingMore: boolean
}

export function Results({ data, busy, onCpc, onWeek, onLike, onMore, loadingMore }: Props) {
  const terms = data.query_terms
  const browse = data.mode === 'browse'
  return (
    <section className={`card results ${busy ? 'is-busy' : ''}`} aria-busy={busy} aria-live="polite">
      <header className="card-head">
        <div>
          <h2>Results</h2>
          <p>
            {data.n_results} {browse ? 'patents, newest first' : `of the top ${data.n_match} matches, best first`}
          </p>
        </div>
      </header>
      {data.results.length === 0 && (
        <div className="empty">No patents match these filters. Remove a filter above to widen the set.</div>
      )}
      <ol className="res-list">
        {data.results.map((r) => (
          <ResultCard key={r.doc_number} r={r} terms={terms} browse={browse}
            onCpc={onCpc} onWeek={onWeek} onLike={onLike} />
        ))}
      </ol>
      {data.results.length < data.n_results && (
        <button className="btn btn-ghost more" onClick={onMore} disabled={loadingMore}>
          {loadingMore ? 'Loading…' : `Show more (${data.n_results - data.results.length} left)`}
        </button>
      )}
    </section>
  )
}

function ResultCard({ r, terms, browse, onCpc, onWeek, onLike }: {
  r: Result; terms: string[]; browse: boolean
  onCpc: (c: string) => void; onWeek: (w: string) => void; onLike: (d: string) => void
}) {
  const [open, setOpen] = useState(false)
  const mc = r.matched_claim
  const kwHits = r.terms_in_title_abstract ?? []
  const group = r.cpc.includes('/') ? r.cpc.split('/')[0] + '/' : r.cpc
  return (
    <li className="res">
      <div className="res-top">
        <span className="res-rank mono">{String(r.rank).padStart(2, '0')}</span>
        <span className="mono dim">{r.doc_number}</span>
        <button className="tag" onClick={() => onCpc(group)} title="Filter to this CPC group">{r.cpc}</button>
        <button className="tag" onClick={() => onWeek(r.week)} title="Filter to this filing week">{fmtWeek(r.week)}</button>
        <span className="sp" />
        <button className="link-btn" onClick={() => onLike(r.doc_number)}>More like this ↗</button>
      </div>

      <h3 className="res-title">{highlight(r.title, terms)}</h3>

      {!browse && (
        <div className="why" aria-label="Why it matched">
          {r.dense !== undefined && (
            <span className="why-item" title="Cosine similarity between the query and this patent's title+abstract+claims embedding">
              <span className="why-k">meaning</span>
              <span className="meter"><span style={{ width: `${Math.max(0, Math.min(1, r.dense)) * 100}%` }} /></span>
              <span className="mono">{r.dense.toFixed(2)}</span>
            </span>
          )}
          {kwHits.length > 0 && (
            <span className="why-item">
              <span className="why-k">keywords</span>
              <span className="mono">{kwHits.slice(0, 6).join(', ')}</span>
            </span>
          )}
          {mc && (
            <span className="why-item">
              <span className="why-k">best claim</span>
              <span className="mono">#{mc.claim_no}{mc.score !== null ? ` · ${mc.score.toFixed(2)}` : ''}</span>
            </span>
          )}
        </div>
      )}

      {mc && !browse ? (
        <blockquote className={`claim ${open ? 'open' : ''}`}>
          <span className="claim-k mono">claim {mc.claim_no}</span>
          {highlight(mc.text, terms, open ? undefined : 260)}
        </blockquote>
      ) : (
        <p className="abstract">{highlight(r.abstract, terms, open ? undefined : 240)}</p>
      )}
      {open && !browse && <p className="abstract">{highlight(r.abstract, terms)}</p>}
      <button className="link-btn" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        {open ? 'Show less' : browse ? 'Show full abstract' : 'Show full claim and abstract'}
      </button>
    </li>
  )
}
