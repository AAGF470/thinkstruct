import { useState } from 'react'
import type { Meta } from '../api'
import { pct } from '../lib'
import { describe, type SavedView, type ViewState } from '../state'
import type { PerfSample } from '../useExplore'

interface Props {
  meta: Meta | null
  state: ViewState
  saved: SavedView[]
  onSave: (name: string) => void
  onOpen: (v: SavedView) => void
  onDelete: (id: string) => void
  samples: PerfSample[]
  staleDropped: number
  slow: boolean
  onSlow: (v: boolean) => void
  open: boolean
  onClose: () => void
}

export function Sidebar(p: Props) {
  const [naming, setNaming] = useState(false)
  const [name, setName] = useState('')
  const net = p.samples.filter((s) => s.source === 'network').map((s) => s.ms)
  const all = p.samples.map((s) => s.ms)
  const last = p.samples[p.samples.length - 1]

  return (
    <>
      {p.open && <div className="side-backdrop" onClick={p.onClose} />}
      <aside className={`side ${p.open ? 'open' : ''}`} aria-label="Sidebar">
        <div className="side-top">
          <div className="avatar mono">PE</div>
          <div className="side-id">
            <div className="n">Patent Explorer</div>
            <div className="status"><i />{p.meta ? `${p.meta.n_patents} patents indexed` : 'connecting…'}</div>
          </div>
        </div>

        <div className="side-scroll">
          <div className="grp">Saved views</div>
          {p.saved.length === 0 && <p className="side-note">Save the current query and filters to come back to them.</p>}
          {p.saved.map((v) => (
            <div key={v.id} className="saved">
              <button className="nav-item" onClick={() => p.onOpen(v)} title={describe(v.state)}>
                <span className="saved-name">{v.name}</span>
                {v.n !== undefined && <span className="saved-n mono">{v.n}</span>}
              </button>
              <button className="x" onClick={() => p.onDelete(v.id)} aria-label={`Delete saved view ${v.name}`}>×</button>
            </div>
          ))}
          {naming ? (
            <form className="save-form" onSubmit={(e) => {
              e.preventDefault()
              p.onSave(name.trim() || describe(p.state))
              setName(''); setNaming(false)
            }}>
              <input autoFocus value={name} onChange={(e) => setName(e.target.value)}
                placeholder={describe(p.state)} aria-label="Name for this view"
                onKeyDown={(e) => e.key === 'Escape' && setNaming(false)} />
            </form>
          ) : (
            <button className="side-resume" onClick={() => setNaming(true)}>Save this view <span>＋</span></button>
          )}
        </div>

        <div className="side-info">
          <div className="side-block">
            <div className="meta-line mono">UPDATE LATENCY</div>
            {last ? (
              <dl className="perf">
                <div><dt>last</dt><dd className="mono">{last.ms.toFixed(0)} ms{last.source === 'cache' ? ' (cache)' : ''}</dd></div>
                <div><dt>p50 / p95</dt><dd className="mono">{pct(all, 50).toFixed(0)} / {pct(all, 95).toFixed(0)} ms</dd></div>
                <div><dt>network p50</dt><dd className="mono">{net.length ? `${pct(net, 50).toFixed(0)} ms` : '–'}</dd></div>
                <div><dt>server</dt><dd className="mono">{last.serverMs.toFixed(1)} ms</dd></div>
                <div><dt>superseded</dt><dd className="mono">{p.staleDropped}</dd></div>
              </dl>
            ) : <p className="side-val">Measured from each filter change to the next painted frame.</p>}
          </div>
          {p.meta?.debug_delay && (
            <label className="side-toggle">
              <input type="checkbox" checked={p.slow} onChange={(e) => p.onSlow(e.target.checked)} />
              Simulate slow, out-of-order server (0–1.5 s)
            </label>
          )}
          <div className="side-block">
            <div className="meta-line mono">MODEL</div>
            <div className="side-val mono">{p.meta ? `${p.meta.model.split('/').pop()} · ${p.meta.dim}-d` : '…'}</div>
          </div>
          <a className="side-link" href="simple">Simple search page ↗</a>
        </div>
      </aside>
    </>
  )
}
