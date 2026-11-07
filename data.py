"""Load, clean, and chunk the Thinkstruct patent JSON files.

Data in:  a folder of patents_ipa{YYMMDD}.json files, each a list of patent dicts
          (title, doc_number, filename, abstract, detailed_description[], claims[],
          bibtex, classification).
Data out: list[Patent] (one per application) and list[Chunk] (the units we embed).

Missing-field policy (README requirement):
  * A patent is KEPT if it has at least one of title / abstract / claims.
    Missing fields become "" or [] and are listed in Patent.missing, so the
    search layer can still filter on whatever metadata exists.
  * A patent is SKIPPED only if title, abstract and claims are all empty
    (nothing to embed or match on). Skips are counted in LoadReport.
  * Duplicate doc_numbers keep the first occurrence (files are read in date order).
  * Unreadable/corrupt files are skipped and recorded, not fatal.
  The small dataset has no missing fields; this mainly matters for the large set.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import asdict, dataclass, field

DEFAULT_DATA_DIR = os.path.join("UTF-8patent_data_small", "data", "patent_data_small")


# --------------------------------------------------------------------------- models
@dataclass
class Patent:
    doc_number: str
    title: str
    abstract: str
    claims: list[str]            # cleaned claim text, index 0 == claim 1
    description: list[str]       # detailed-description paragraphs
    classification: str          # raw code from the data, e.g. "B60B104FI"
    cpc: dict                    # parsed hierarchy, see parse_cpc()
    week: str                    # filing week from the file name, ISO "2024-02-15"
    source_file: str
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: int                # row number in the embedding matrix
    doc_number: str
    kind: str                    # "patent" (title+abstract+claims) | "claim" | "desc"
    text: str                    # exactly what gets embedded
    claim_no: int | None = None  # 1-based, for kind == "claim"
    parent_claim: int | None = None  # for dependent claims ("... of claim 1")
    para_no: int | None = None   # 0-based, for kind == "desc"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LoadReport:
    files_read: int = 0
    files_failed: list[str] = field(default_factory=list)
    patents_kept: int = 0
    patents_skipped_empty: int = 0
    duplicates_dropped: int = 0
    patents_with_missing_fields: int = 0


# --------------------------------------------------------------------------- cleaning
_CLAIM_NUM = re.compile(r"^\s*(\d+)\s*\.\s*")              # "2 . The spoke ..."
_PARENT = re.compile(r"\bclaim\s+(\d+)\b", re.IGNORECASE)  # "according to claim 1"
_SPACE_PUNCT = re.compile(r"\s+([,.;:)])")                 # "claim 1 ," -> "claim 1,"
_WS = re.compile(r"\s+")


def clean_text(s: str | None) -> str:
    if not s:
        return ""
    s = _WS.sub(" ", str(s)).strip()
    return _SPACE_PUNCT.sub(r"\1", s)


def clean_claim(raw: str) -> tuple[str, int | None]:
    """Strip the leading "N . " number and return (text, parent_claim_or_None)."""
    text = clean_text(_CLAIM_NUM.sub("", raw or ""))
    m = _PARENT.search(text)
    return text, (int(m.group(1)) if m else None)


_CANCELED = re.compile(r"\(\s*cancell?ed\s*\)", re.IGNORECASE)


def is_canceled(claim_text: str) -> bool:
    """Withdrawn-claim placeholders: "(canceled)", "1 - 15. (canceled)", ...
    (~150 in the small set, across 25 patents). Kept in Patent.claims so claim
    numbering stays intact, but never embedded or shown as a match."""
    return len(claim_text) < 60 and bool(_CANCELED.search(claim_text))


def parse_cpc(code: str | None) -> dict:
    """Split a code like "B60B104FI" into its CPC hierarchy.

    The data drops the space and slash of the usual "B60B 1/04" notation, so the
    split is inferred: subgroups have at least two digits, and main groups in
    this corpus (class B60) are at most two digits. Hence 3 digits = 1+2
    ("104" -> 1/04), 4 digits = 2+2 ("2700" -> 27/00), 5+ digits = 2+rest
    ("15024" -> 15/024, a tyre-bead code; "25138" -> 25/138, tyre changing).
    Other classes have 3-digit main groups (e.g. C07D 487/), so this rule would
    need a CPC lookup table for the full dataset. The trailing letters ("FI") are
    kept in `suffix`, and the raw code is always kept; bare prefix filters
    ("B60B") match the raw code.
    """
    code = (code or "").strip().upper()
    m = re.fullmatch(r"([A-H])(\d\d)([A-Z])(\d+)([A-Z]*)", code)
    if not m or len(m.group(4)) < 3:
        return {"raw": code, "section": code[:1], "class": code[:3],
                "subclass": code[:4], "group": None, "display": code or "N/A"}
    sec, cls, sub, digits, suffix = m.groups()
    cut = len(digits) - 2 if len(digits) <= 3 else 2
    group, subgroup = digits[:cut], digits[cut:]
    subclass = f"{sec}{cls}{sub}"
    return {"raw": code, "section": sec, "class": f"{sec}{cls}", "subclass": subclass,
            "group": f"{subclass} {group}/{subgroup}", "suffix": suffix,
            "display": f"{subclass} {group}/{subgroup}"}


def week_from_filename(path: str) -> str:
    """patents_ipa240215.json -> "2024-02-15"."""
    m = re.search(r"ipa(\d{2})(\d{2})(\d{2})", os.path.basename(path))
    return f"20{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


# --------------------------------------------------------------------------- loading
def load_patents(data_dir: str = DEFAULT_DATA_DIR, limit: int | None = None
                 ) -> tuple[list[Patent], LoadReport]:
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .json files under {data_dir!r}")

    report, seen, patents = LoadReport(), set(), []
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                records = json.load(f)
            if not isinstance(records, list):
                raise ValueError("top-level JSON is not a list")
        except Exception as e:  # corrupt / truncated file: record and move on
            report.files_failed.append(f"{os.path.basename(path)}: {e}")
            continue
        report.files_read += 1

        for r in records:
            if not isinstance(r, dict):
                continue
            doc = str(r.get("doc_number") or "").strip()
            if doc and doc in seen:
                report.duplicates_dropped += 1
                continue

            title = clean_text(r.get("title"))
            abstract = clean_text(r.get("abstract"))
            claims = [clean_claim(c)[0] for c in (r.get("claims") or []) if c]
            desc = [clean_text(p) for p in (r.get("detailed_description") or []) if p]
            if not (title or abstract or claims):
                report.patents_skipped_empty += 1
                continue

            missing = [name for name, val in [("title", title), ("abstract", abstract),
                       ("claims", claims), ("detailed_description", desc),
                       ("classification", r.get("classification"))] if not val]
            if missing:
                report.patents_with_missing_fields += 1

            seen.add(doc)
            patents.append(Patent(
                doc_number=doc, title=title, abstract=abstract, claims=claims,
                description=desc, classification=(r.get("classification") or "N/A"),
                cpc=parse_cpc(r.get("classification")), week=week_from_filename(path),
                source_file=os.path.basename(path), missing=missing))
            if limit and len(patents) >= limit:
                report.patents_kept = len(patents)
                return patents, report

    report.patents_kept = len(patents)
    return patents, report


# --------------------------------------------------------------------------- chunking
def patent_text(p: Patent) -> str:
    """Title + abstract + claims: the 'TAC' view, best for patent-level retrieval
    in the 2026 patent-embedding benchmark (arXiv 2605.24297)."""
    claims = " ".join(f"{i}. {c}" for i, c in enumerate(p.claims, 1) if not is_canceled(c))
    parts = [p.title, p.abstract, claims]
    return "\n".join(x for x in parts if x)


def chunk_patents(patents: list[Patent], with_description: bool = False) -> list[Chunk]:
    """One 'patent' chunk per application (ranking) + one chunk per claim
    (shows WHICH claim matched). Claim chunks are prefixed with the title because
    ~2/3 of claims are dependent ("The spoke of claim 1, wherein ...") and carry
    little topic on their own."""
    chunks: list[Chunk] = []

    def add(**kw):
        chunks.append(Chunk(chunk_id=len(chunks), **kw))

    for p in patents:
        add(doc_number=p.doc_number, kind="patent", text=patent_text(p))
        for i, raw in enumerate(p.claims, 1):
            if is_canceled(raw):
                continue
            _, parent = clean_claim(raw)
            add(doc_number=p.doc_number, kind="claim", claim_no=i,
                parent_claim=parent if parent != i else None,
                text=f"{p.title}. {raw}" if p.title else raw)
        if with_description:
            for j, para in enumerate(p.description):
                add(doc_number=p.doc_number, kind="desc", para_no=j, text=para)
    return chunks


# --------------------------------------------------------------------------- BM25 helpers
_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset("""a an and are as at be by for from has have in is it its of on or
that the to was were which with wherein said comprising least one each first second""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens for keyword search / BM25."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


if __name__ == "__main__":  # quick sanity check:  python data.py
    pats, rep = load_patents()
    chs = chunk_patents(pats)
    print(rep)
    print(f"{len(pats)} patents -> {len(chs)} chunks "
          f"({sum(c.kind == 'claim' for c in chs)} claims)")
    p = pats[0]
    print(p.doc_number, p.title, p.cpc["display"], p.week)
    print("claim 2:", p.claims[1][:120])
