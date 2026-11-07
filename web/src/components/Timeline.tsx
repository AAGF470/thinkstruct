import { useMemo, useRef, useState } from 'react'
import { fmtWeek } from '../lib'

interface Props {
  axis: string[]                     // every filing week in the corpus (stable x-axis)
  counts: Record<string, number>     // counts in the CURRENT filtered result set
  from: string | null
  to: string | null
  onSelect: (from: string | null, to: string | null) => void
  busy: boolean
}

const H = 120, PAD_B = 22, PAD_T = 8

export function Timeline({ axis, counts, from, to, onSelect, busy }: Props) {
  const [drag, setDrag] = useState<{ a: number; b: number } | null>(null)
  const [hover, setHover] = useState<number | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const n = axis.length
  const max = Math.max(1, ...axis.map((w) => counts[w] ?? 0))
  const total = axis.reduce((s, w) => s + (counts[w] ?? 0), 0)

  const sel = useMemo(() => {
    if (drag) return [Math.min(drag.a, drag.b), Math.max(drag.a, drag.b)] as const
    if (!from && !to) return null
    const lo = from ? axis.findIndex((w) => w >= from) : 0
    let hi = to ? axis.length - 1 - [...axis].reverse().findIndex((w) => w <= to) : n - 1
    if (hi >= n) hi = n - 1
    return lo < 0 ? null : ([lo, hi] as const)
  }, [drag, from, to, axis, n])

  const idxAt = (clientX: number) => {
    const r = svgRef.current!.getBoundingClientRect()
    return Math.max(0, Math.min(n - 1, Math.floor(((clientX - r.left) / r.width) * n)))
  }

  const finish = () => {
    if (!drag) return
    const lo = Math.min(drag.a, drag.b), hi = Math.max(drag.a, drag.b)
    setDrag(null)
    // clicking the already-selected single week clears it
    if (lo === hi && from === axis[lo] && to === axis[hi]) onSelect(null, null)
    else onSelect(axis[lo], axis[hi])
  }

  // month ticks: first week of each month
  const ticks = axis
    .map((w, i) => ({ w, i }))
    .filter(({ w, i }) => i === 0 || w.slice(0, 7) !== axis[i - 1].slice(0, 7))
  const tip = hover ?? (drag ? drag.b : null)

  return (
    <section className={`card timeline ${busy ? 'is-busy' : ''}`} aria-busy={busy}>
      <header className="card-head">
        <div>
          <h2>Filings by week</h2>
          <p>{total} patents · drag to select a range, click a bar for one week</p>
        </div>
        <div className="card-head-aside mono">
          {tip !== null
            ? `${fmtWeek(axis[tip])} · ${counts[axis[tip]] ?? 0}`
            : sel ? `${fmtWeek(axis[sel[0]])} → ${fmtWeek(axis[sel[1]])}` : `${fmtWeek(axis[0])} → ${fmtWeek(axis[n - 1])}`}
        </div>
      </header>
      <svg
        ref={svgRef}
        className="tl-svg"
        viewBox={`0 0 ${n * 10} ${H}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={`Filings per week, ${total} patents across ${n} weeks`}
        onPointerDown={(e) => {
          (e.target as Element).setPointerCapture?.(e.pointerId)
          const i = idxAt(e.clientX)
          setDrag({ a: i, b: i })
        }}
        onPointerMove={(e) => {
          const i = idxAt(e.clientX)
          setHover(i)
          if (drag) setDrag({ ...drag, b: i })
        }}
        onPointerUp={finish}
        onPointerLeave={() => setHover(null)}
      >
        {sel && (
          <rect className="tl-sel" x={sel[0] * 10} y={0} width={(sel[1] - sel[0] + 1) * 10} height={H - PAD_B} />
        )}
        {axis.map((w, i) => {
          const c = counts[w] ?? 0
          const h = c ? Math.max(2, ((H - PAD_B - PAD_T) * c) / max) : 0
          const inSel = !sel || (i >= sel[0] && i <= sel[1])
          return (
            <rect
              key={w}
              className={`tl-bar ${inSel ? '' : 'dim'} ${hover === i ? 'hot' : ''}`}
              x={i * 10 + 1.5}
              y={H - PAD_B - h}
              width={7}
              height={h}
            />
          )
        })}
        <line className="tl-axis" x1={0} x2={n * 10} y1={H - PAD_B + 0.5} y2={H - PAD_B + 0.5} />
      </svg>
      <div className="tl-ticks" aria-hidden="true">
        {ticks.map(({ w, i }) => (
          <span key={w} style={{ left: `${(i / n) * 100}%` }}>
            {w.slice(5, 7) === '01' || i === 0 ? `${fmtWeek(w).split(' ')[0]} ’${w.slice(2, 4)}` : fmtWeek(w).split(' ')[0]}
          </span>
        ))}
      </div>
    </section>
  )
}
