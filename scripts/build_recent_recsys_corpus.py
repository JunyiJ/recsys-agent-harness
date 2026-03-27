from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path


USER_AGENT = "agentic-bench-corpus-builder/0.1"
TARGET_PER_BUCKET = 50

VENUE_BUCKETS = [
    ("RecSys", 2024, ["db/conf/recsys/recsys2024.bht"]),
    ("RecSys", 2025, ["db/conf/recsys/recsys2025.bht"]),
    ("KDD", 2024, ["db/conf/kdd/kdd2024.bht"]),
    ("KDD", 2025, ["db/conf/kdd/kdd2025-1.bht", "db/conf/kdd/kdd2025-2.bht"]),
    ("WWW", 2024, ["db/conf/www/www2024.bht"]),
    ("WWW", 2025, ["db/conf/www/www2025.bht"]),
]

EXCLUDED_TITLE_PATTERNS = [
    r"^Proceedings of\b",
    r"\bWorkshop\b",
    r"\bDoctoral Symposium\b",
    r"\bIndustry (Day|Track)\b",
    r"\bChallenge\b",
    r"\bCompetition\b",
    r"\bTutorial\b",
    r"\bDemo(nstration)?\b",
    r"\bPanel\b",
    r"\bKeynote\b",
    r"\bPreface\b",
    r"\bCall for\b",
    r"\bMessage from\b",
]
EXCLUDED_TITLE_RE = re.compile("|".join(EXCLUDED_TITLE_PATTERNS), re.IGNORECASE)
RELEVANCE_RE = re.compile(
    r"\b("
    r"recommend|recommender|recommendation|matching|personaliz|session|"
    r"slate|ctr|click-?through|e-?commerce|filter bubble|exposure|feed|"
    r"micro-video|short-video|collaborative filtering|matrix factorization"
    r")\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class CandidatePaper:
    doc_id: str
    title: str
    doi_url: str
    dblp_url: str
    venue: str
    year: int


def fetch_text(url: str) -> str:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            result = subprocess.run(
                ["curl", "-LksS", "--max-time", "30", "-A", USER_AGENT, url],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as error:
            last_error = error
            time.sleep(0.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def fetch_json(url: str) -> dict:
    return json.loads(fetch_text(url))


def normalize_title(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions: list[tuple[int, str]] = []
    for token, indexes in index.items():
        for pos in indexes:
            positions.append((pos, token))
    positions.sort()
    return " ".join(token for _, token in positions)


def keep_candidate(venue: str, title: str) -> bool:
    if EXCLUDED_TITLE_RE.search(title):
        return False
    if venue == "RecSys":
        return True
    return bool(RELEVANCE_RE.search(title))


def dblp_hits_for_toc(toc_path: str) -> list[dict]:
    query = urllib.parse.quote(f"toc:{toc_path}:", safe="")
    url = f"https://dblp.org/search/publ/api?q={query}&h=1000&format=json"
    payload = fetch_json(url)
    hits = payload["result"]["hits"].get("hit", [])
    if isinstance(hits, dict):
        return [hits]
    return hits


def load_candidates(venue: str, year: int, toc_paths: list[str]) -> list[CandidatePaper]:
    candidates: list[CandidatePaper] = []
    seen_doc_ids: set[str] = set()
    for toc_path in toc_paths:
        for hit in dblp_hits_for_toc(toc_path):
            info = hit.get("info", {})
            title = (info.get("title") or "").strip()
            key = info.get("key") or ""
            ee = info.get("ee") or ""
            dblp_url = info.get("url") or ""
            if not key or not title or not ee or not dblp_url:
                continue
            if not keep_candidate(venue=venue, title=title):
                continue
            doc_id = key.replace("/", "_")
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            candidates.append(
                CandidatePaper(
                    doc_id=doc_id,
                    title=title,
                    doi_url=ee,
                    dblp_url=dblp_url,
                    venue=venue,
                    year=year,
                )
            )
    return candidates


def enrich_candidate(candidate: CandidatePaper) -> dict | None:
    doi = candidate.doi_url.removeprefix("https://doi.org/")
    openalex_url = "https://api.openalex.org/works/doi:" + urllib.parse.quote(doi, safe="")
    try:
        work = fetch_json(openalex_url)
    except Exception:
        return None

    openalex_title = (work.get("title") or "").strip()
    if normalize_title(openalex_title) != normalize_title(candidate.title):
        return None

    text = abstract_from_inverted_index(work.get("abstract_inverted_index"))
    if not text:
        return None

    return {
        "doc_id": candidate.doc_id,
        "title": openalex_title,
        "source": candidate.venue,
        "year": candidate.year,
        "url": candidate.dblp_url,
        "text": text,
    }


def collect_bucket(venue: str, year: int, toc_paths: list[str], per_bucket: int) -> list[dict]:
    candidates = load_candidates(venue=venue, year=year, toc_paths=toc_paths)
    rows: list[dict] = []
    for candidate in candidates:
        row = enrich_candidate(candidate)
        if row is None:
            continue
        rows.append(row)
        if len(rows) >= per_bucket:
            break
        time.sleep(0.05)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a verified recent recsys paper corpus.")
    parser.add_argument(
        "--output",
        default="data/corpus/recsys_recent_2024_2025_300.jsonl",
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--per-bucket",
        type=int,
        default=TARGET_PER_BUCKET,
        help="Number of accepted papers per venue/year bucket.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[1] / output_path

    all_rows: list[dict] = []
    for venue, year, toc_paths in VENUE_BUCKETS:
        rows = collect_bucket(venue=venue, year=year, toc_paths=toc_paths, per_bucket=args.per_bucket)
        print(f"{venue} {year}: kept {len(rows)} papers", file=sys.stderr)
        if len(rows) < args.per_bucket:
            raise SystemExit(f"Failed to collect {args.per_bucket} verified papers for {venue} {year}.")
        all_rows.extend(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_rows)} papers to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
