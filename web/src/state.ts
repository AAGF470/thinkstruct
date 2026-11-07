// Single source of truth for what the user asked for. Every view reads the
// *response* to this state, never its own copy, so views cannot drift apart.

import type { ExploreParams, Mode } from './api'

export type ViewState = ExploreParams

export const EMPTY: ViewState = {
  q: '', mode: 'hybrid', cpc: null, kw: [], weekFrom: null, weekTo: null, like: null,
}

export type Action =
  | { type: 'query'; q: string }
  | { type: 'mode'; mode: Mode }
  | { type: 'cpc'; cpc: string | null }
  | { type: 'addKw'; kw: string }
  | { type: 'removeKw'; kw: string }
  | { type: 'weeks'; from: string | null; to: string | null }
  | { type: 'like'; doc: string | null }
  | { type: 'clearFilters' }
  | { type: 'restore'; state: ViewState }

export function reducer(s: ViewState, a: Action): ViewState {
  switch (a.type) {
    // A new query keeps every filter (brief: "running a new query should keep the
    // current filters applied"); it only replaces the query / more-like-this source.
    case 'query': return { ...s, q: a.q, like: null }
    case 'mode': return { ...s, mode: a.mode }
    case 'cpc': return { ...s, cpc: a.cpc }
    case 'addKw': {
      const kw = a.kw.trim().toLowerCase()
      return !kw || s.kw.includes(kw) ? s : { ...s, kw: [...s.kw, kw] }
    }
    case 'removeKw': return { ...s, kw: s.kw.filter((k) => k !== a.kw) }
    case 'weeks': return { ...s, weekFrom: a.from, weekTo: a.to }
    case 'like': return { ...s, like: a.doc }
    case 'clearFilters': return { ...s, cpc: null, kw: [], weekFrom: null, weekTo: null }
    case 'restore': return { ...EMPTY, ...a.state }
  }
}

export function hasFilters(s: ViewState): boolean {
  return Boolean(s.cpc || s.kw.length || s.weekFrom || s.weekTo)
}

// ---- URL hash <-> state (shareable links, back/forward) --------------------------
export function toHash(s: ViewState): string {
  const p = new URLSearchParams()
  if (s.q) p.set('q', s.q)
  if (s.like) p.set('like', s.like)
  if (s.mode !== 'hybrid') p.set('mode', s.mode)
  if (s.cpc) p.set('cpc', s.cpc)
  for (const k of s.kw) p.append('kw', k)
  if (s.weekFrom) p.set('from', s.weekFrom)
  if (s.weekTo) p.set('to', s.weekTo)
  const str = p.toString()
  return str ? `#${str}` : ''
}

export function fromHash(hash: string): ViewState | null {
  const h = hash.replace(/^#/, '')
  if (!h) return null
  const p = new URLSearchParams(h)
  const mode = p.get('mode')
  return {
    q: p.get('q') ?? '',
    like: p.get('like'),
    mode: mode === 'dense' || mode === 'bm25' ? mode : 'hybrid',
    cpc: p.get('cpc'),
    kw: p.getAll('kw'),
    weekFrom: p.get('from'),
    weekTo: p.get('to'),
  }
}

// ---- saved views (localStorage; per browser, may be unavailable) -----------------
export interface SavedView { id: string; name: string; state: ViewState; savedAt: number; n?: number }
const SAVED_KEY = 'patent-explorer.saved-views.v1'
const LAST_KEY = 'patent-explorer.last-state.v1'

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key)
    return raw ? (JSON.parse(raw) as T) : fallback
  } catch {
    return fallback
  }
}
function write(key: string, value: unknown): void {
  try { localStorage.setItem(key, JSON.stringify(value)) } catch { /* private mode / quota */ }
}

export const loadSaved = () => read<SavedView[]>(SAVED_KEY, [])
export const storeSaved = (v: SavedView[]) => write(SAVED_KEY, v)
export const loadLast = () => read<ViewState | null>(LAST_KEY, null)
export const storeLast = (s: ViewState) => write(LAST_KEY, s)

export function describe(s: ViewState): string {
  const parts: string[] = []
  if (s.like) parts.push(`like ${s.like}`)
  else if (s.q) parts.push(`“${s.q.length > 40 ? s.q.slice(0, 40) + '…' : s.q}”`)
  if (s.cpc) parts.push(s.cpc)
  if (s.kw.length) parts.push(s.kw.map((k) => `+${k}`).join(' '))
  if (s.weekFrom || s.weekTo) parts.push(s.weekFrom === s.weekTo ? `${s.weekFrom}` : `${s.weekFrom ?? '…'}→${s.weekTo ?? '…'}`)
  return parts.join(' · ') || 'All patents'
}
