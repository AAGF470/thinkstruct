import type { ReactNode } from 'react'

// ---- CPC hierarchy -----------------------------------------------------------------
// Facet keys from the API are main groups ("B60B 1/") or "N/A". Levels used for
// drill-down: section "B" -> class "B60" -> subclass "B60B" -> group "B60B 1/".
export type CpcLevel = 0 | 1 | 2 | 3
const LEVEL_LEN = [1, 3, 4] as const

export function cpcAt(group: string, level: CpcLevel): string {
  if (group === 'N/A') return 'N/A'
  return level === 3 ? group : group.slice(0, LEVEL_LEN[level])
}

export function levelOf(prefix: string | null): CpcLevel | -1 {
  if (!prefix) return -1
  if (prefix.includes('/')) return 3
  if (prefix.length >= 4) return 2
  if (prefix.length >= 3) return 1
  return 0
}

export function ancestors(prefix: string | null): string[] {
  const lv = levelOf(prefix)
  if (!prefix || lv < 0) return []
  const out: string[] = []
  for (let l = 0; l <= lv; l++) out.push(l === 3 ? prefix : prefix.slice(0, LEVEL_LEN[l as 0 | 1 | 2]))
  return out
}

export function cpcChildren(groups: Record<string, number>, prefix: string | null) {
  // The first level below `prefix` that splits into more than one bucket is the
  // useful one to show (all patents here are section B / class B60).
  const entries = Object.entries(groups).filter(([g]) => g !== 'N/A' && (!prefix || g.startsWith(prefix)))
  const start = (levelOf(prefix) + 1) as CpcLevel
  for (let level = start; level <= 3; level = (level + 1) as CpcLevel) {
    const counts = new Map<string, number>()
    for (const [g, n] of entries) counts.set(cpcAt(g, level), (counts.get(cpcAt(g, level)) ?? 0) + n)
    if (counts.size > 1 || level === 3) {
      return { level, rows: [...counts.entries()].sort((a, b) => b[1] - a[1]) }
    }
  }
  return { level: 3 as CpcLevel, rows: [] as [string, number][] }
}

// ---- dates -------------------------------------------------------------------------
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
export function fmtWeek(w: string): string {
  const [y, m, d] = w.split('-').map(Number)
  return `${MONTHS[m - 1]} ${d}, ${y}`
}
export function fmtRange(a: string | null, b: string | null): string {
  if (a && b && a === b) return `week of ${fmtWeek(a)}`
  return `${a ? fmtWeek(a) : 'start'} → ${b ? fmtWeek(b) : 'now'}`
}

// ---- query-term highlighting ("why it matched") -------------------------------------
function escapeRe(s: string) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') }

export function highlight(text: string, terms: string[], max?: number): ReactNode {
  const t = max && text.length > max ? text.slice(0, max).replace(/\s+\S*$/, '') + '…' : text
  const usable = terms.filter((x) => x.length > 2)
  if (!usable.length) return t
  const re = new RegExp(`\\b(${usable.map(escapeRe).join('|')})\\w*`, 'gi')
  const out: ReactNode[] = []
  let last = 0
  for (const m of t.matchAll(re)) {
    const i = m.index ?? 0
    if (i > last) out.push(t.slice(last, i))
    out.push(<mark key={i}>{m[0]}</mark>)
    last = i + m[0].length
  }
  if (last < t.length) out.push(t.slice(last))
  return out
}

export function pct(xs: number[], p: number): number {
  if (!xs.length) return 0
  const s = [...xs].sort((a, b) => a - b)
  return s[Math.min(s.length - 1, Math.floor((p / 100) * (s.length - 1)))]
}
