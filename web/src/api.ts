// Typed client for the FastAPI backend (app/main.py).

export type Mode = 'hybrid' | 'dense' | 'bm25'

export interface MatchedClaim {
  claim_no: number
  score: number | null
  text: string
}

export interface Result {
  rank: number
  doc_number: string
  title: string
  cpc: string
  week: string
  abstract: string
  score?: number
  dense?: number
  bm25?: number
  matched_claim?: MatchedClaim
  terms_in_title_abstract?: string[]
}

export interface ExploreResponse {
  results: Result[]
  n_results: number
  n_match: number
  match_n: number | null
  offset: number
  facets: { cpc: Record<string, number>; weeks: Record<string, number> }
  mode: string
  filters: string
  query_terms: string[]
  timings_ms: Record<string, number>
  model: string
  rid: string | null
  server_ms: number
}

export interface Meta {
  n_patents: number
  weeks: string[]
  cpc: Record<string, number>
  model: string
  dim: number
  cpc_titles: Record<string, string>
  rerank_enabled: boolean
  debug_delay: boolean
}

export interface ExploreParams {
  q: string
  mode: Mode
  cpc: string | null
  kw: string[]
  weekFrom: string | null
  weekTo: string | null
  like: string | null
}

export function toSearchParams(p: ExploreParams): URLSearchParams {
  const s = new URLSearchParams()
  if (p.like) s.set('like', p.like)
  else if (p.q.trim()) s.set('q', p.q.trim())
  s.set('mode', p.mode)
  if (p.cpc) s.set('cpc', p.cpc)
  for (const k of p.kw) s.append('kw', k)
  if (p.weekFrom) s.set('week_from', p.weekFrom)
  if (p.weekTo) s.set('week_to', p.weekTo)
  return s
}

/** Canonical key of the *filter state* (not paging / request ids): used for caching. */
export function keyOf(p: ExploreParams): string {
  const s = toSearchParams({ ...p, kw: [...p.kw].sort() })
  s.sort()
  return s.toString()
}

export async function fetchMeta(): Promise<Meta> {
  const r = await fetch('api/meta')
  if (!r.ok) throw new Error(`meta: HTTP ${r.status}`)
  return r.json()
}

export async function fetchExplore(
  p: ExploreParams,
  opts: { rid: string; offset?: number; k?: number; delayMs?: number; signal?: AbortSignal },
): Promise<ExploreResponse> {
  const s = toSearchParams(p)
  s.set('rid', opts.rid)
  s.set('k', String(opts.k ?? 20))
  if (opts.offset) s.set('offset', String(opts.offset))
  if (opts.delayMs) s.set('delay_ms', String(opts.delayMs))
  const r = await fetch(`api/explore?${s}`, { signal: opts.signal })
  const body = await r.json().catch(() => ({}))
  if (!r.ok) {
    const d = (body as { detail?: unknown }).detail
    throw new Error(typeof d === 'string' ? d : Array.isArray(d) ? String(d[0]?.msg ?? r.status) : `HTTP ${r.status}`)
  }
  return body as ExploreResponse
}
