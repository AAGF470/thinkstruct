import { ancestors, cpcChildren } from '../lib'

interface Props {
  counts: Record<string, number>   // group -> count in the CURRENT filtered set
  titles: Record<string, string>
  selected: string | null
  onSelect: (cpc: string | null) => void
  busy: boolean
}

export function CpcTree({ counts, titles, selected, onSelect, busy }: Props) {
  const { level, rows } = cpcChildren(counts, selected)
  const total = Object.values(counts).reduce((a, b) => a + b, 0)
  const unclassified = counts['N/A'] ?? 0
  const max = Math.max(1, ...rows.map(([, n]) => n))
  const crumbs = ancestors(selected)
  const levelName = ['section', 'class', 'subclass', 'main group'][level]

  return (
    <section className={`card cpc ${busy ? 'is-busy' : ''}`} aria-busy={busy}>
      <header className="card-head">
        <div>
          <h2>Classification</h2>
          <p>{total} patents by CPC {levelName} · click to drill down</p>
        </div>
      </header>

      <nav className="crumbs" aria-label="CPC path">
        <button className={!selected ? 'on' : ''} onClick={() => onSelect(null)}>All</button>
        {crumbs.map((c) => (
          <span key={c}>
            <span className="sep">›</span>
            <button className={c === selected ? 'on' : ''} onClick={() => onSelect(c)} title={titles[c]}>
              {c}
            </button>
          </span>
        ))}
      </nav>

      <ul className="cpc-rows">
        {rows.map(([code, n]) => {
          const isSel = code === selected
          return (
            <li key={code}>
              <button
                className={`cpc-row ${isSel ? 'on' : ''}`}
                onClick={() => onSelect(isSel ? crumbs[crumbs.length - 2] ?? null : code)}
                aria-pressed={isSel}
              >
                <span className="cpc-code mono">{code}</span>
                <span className="cpc-title">{titles[code] ?? (level === 3 ? `Main group ${code.split(' ')[1]}` : '')}</span>
                <span className="cpc-n mono">{n}</span>
                <span className="cpc-bar" style={{ width: `${(n / max) * 100}%` }} />
              </button>
            </li>
          )
        })}
        {!rows.length && <li className="empty">No classified patents in this set.</li>}
      </ul>
      {unclassified > 0 && !selected && (
        <p className="foot-note">{unclassified} without a classification code</p>
      )}
    </section>
  )
}
