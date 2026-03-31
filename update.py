#!/usr/bin/env python3
"""
AI Trust & Safety Digest generator.

Fetches recent trust-and-safety relevant items from official labs,
research groups, evaluators, and policy organizations, curates the
highest-signal developments into a compact daily digest, and writes the
GitHub Pages assets that power the site.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import feedparser
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
ARCHIVE_DIR = DOCS_DIR / "archive"

DIGEST_TITLE = "AI Trust & Safety Digest"
DIGEST_TAGLINE = "A curated current snapshot of AI trust, safety, and governance, including as many materially important fresh items as the day warrants."
MODEL = "gpt-5.4"
HEADERS = {"User-Agent": "AIResearchDigest/0.1"}
REQUEST_TIMEOUT = 20
MAX_STORIES_PER_FEED = 25
MAX_CANDIDATES = 80
CURATED_ITEM_MIN = 5
CURATED_ITEM_HARD_MAX = 24
PREFERRED_STORIES_PER_SOURCE = 2
RECENT_WINDOW_DAYS = 5
FEEDS = [
    {"name": "OpenAI News", "url": "https://openai.com/news/rss.xml"},
    {"name": "Google Research Blog", "url": "https://research.google/blog/rss/"},
    {"name": "Google DeepMind News", "url": "https://deepmind.google/blog/rss.xml"},
    {"name": "NIST Artificial Intelligence", "url": "https://www.nist.gov/news-events/artificial%20intelligence/rss.xml"},
    {"name": "Partnership on AI", "url": "https://partnershiponai.org/feed/"},
    {"name": "METR", "url": "https://metr.org/feed.xml"},
]
DIGEST_NOTE = (
    "Selected from recent trust-and-safety relevant posts by labs, evaluators, "
    "standards bodies, and policy organizations, then summarized with GPT-5.4. "
    "Includes as many materially important fresh items as the day warrants."
)

LOW_SIGNAL_PATTERNS = [
    r"\bopinion\b",
    r"\beditorial\b",
    r"\bpodcast\b",
    r"\bwebinar\b",
    r"\bhow to\b",
    r"\bguide\b",
    r"\bexplainer\b",
    r"\bwhat is\b",
    r"\bbest ai\b",
    r"\bprompts?\b",
    r"\bcase study\b",
    r"\bcustomer stories?\b",
    r"\bwebcast\b",
    r"\bcareer(s)?\b",
    r"\bintern(ship|ships)?\b",
    r"\bjob(s)?\b",
    r"\bmeetup\b",
    r"\bcommunity update(s)?\b",
    r"\bevent(s)?\b",
    r"\bworkshop\b",
]

TRUST_AND_SAFETY_PATTERNS = [
    r"\balignment\b",
    r"\binterpretability\b",
    r"\bsafety\b",
    r"\btrustworthy\b",
    r"\btrust\b",
    r"\bresponsib(?:le|ility)\b",
    r"\bgovernance\b",
    r"\bpolicy\b",
    r"\bevaluation(s)?\b",
    r"\bevals?\b",
    r"\bbenchmark(s)?\b",
    r"\bred[- ]?team(?:ing)?\b",
    r"\bsecurity\b",
    r"\bprivacy\b",
    r"\btransparency\b",
    r"\bsafeguard(s)?\b",
    r"\bpreparedness\b",
    r"\brisk(s)?\b",
    r"\brobust(?:ness)?\b",
    r"\baudit(s|ing)?\b",
    r"\bassurance\b",
    r"\bcompliance\b",
    r"\bincident(s)?\b",
    r"\bsystem card(s)?\b",
    r"\bmodel spec\b",
    r"\bbiosecurity\b",
    r"\bcybersecurity\b",
    r"\bfairness\b",
    r"\bbias\b",
    r"\bmisuse\b",
]

def collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def clean_title(title: str, source: str) -> str:
    title = collapse_whitespace(title)
    if source:
        suffix = f" - {source}"
        if title.endswith(suffix):
            title = title[: -len(suffix)].strip()
    return title


def canonicalize_link(link: str) -> str:
    collapsed = collapse_whitespace(link)
    if not collapsed:
        return ""

    parsed = urlsplit(collapsed)
    if not parsed.scheme or not parsed.netloc:
        return collapsed

    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")


def parse_published(raw_value: str) -> datetime:
    if not raw_value:
        return datetime.now(timezone.utc)
    try:
        parsed = parsedate_to_datetime(raw_value)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def recent_cutoff() -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=RECENT_WINDOW_DAYS)


def is_low_signal(title: str) -> bool:
    title_lower = title.lower()
    return any(re.search(pattern, title_lower) for pattern in LOW_SIGNAL_PATTERNS)


def extract_entry_text(entry: Any) -> str:
    parts = [
        getattr(entry, "title", ""),
        getattr(entry, "summary", ""),
        getattr(entry, "description", ""),
    ]

    if hasattr(entry, "tags") and isinstance(entry.tags, list):
        for tag in entry.tags:
            if isinstance(tag, dict):
                parts.append(tag.get("term", ""))

    return collapse_whitespace(" ".join(part for part in parts if part))


def story_topic_score(text: str) -> int:
    text_lower = text.lower()
    return sum(1 for pattern in TRUST_AND_SAFETY_PATTERNS if re.search(pattern, text_lower))


def story_key(story: dict[str, Any]) -> str:
    canonical_link = canonicalize_link(story["link"])
    if canonical_link:
        return canonical_link.lower()

    normalized = re.sub(r"[^a-z0-9]+", " ", story["title"].lower()).strip()
    source = re.sub(r"[^a-z0-9]+", " ", story["source"].lower()).strip()
    return f"{normalized}|{source}"


def get_entry_source(entry: Any, feed_name: str) -> str:
    if hasattr(entry, "source") and isinstance(entry.source, dict):
        source_title = collapse_whitespace(entry.source.get("title") or "")
        if source_title:
            return source_title
    return feed_name


def fetch_feed_payload(url: str) -> bytes:
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.content
    except requests.RequestException as exc:
        print(f"requests failed for {url}: {exc}. Retrying with curl.")

    try:
        result = subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--silent",
                "--show-error",
                "--max-time",
                str(REQUEST_TIMEOUT),
                "-A",
                HEADERS["User-Agent"],
                url,
            ],
            check=True,
            capture_output=True,
            timeout=REQUEST_TIMEOUT + 5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Unable to fetch {url}") from exc

    return result.stdout


def fetch_candidates() -> list[dict[str, Any]]:
    stories: list[dict[str, Any]] = []
    cutoff = recent_cutoff()

    for feed_config in FEEDS:
        feed_name = feed_config["name"]
        url = feed_config["url"]

        try:
            payload = fetch_feed_payload(url)
        except RuntimeError as exc:
            print(f"Skipping feed {feed_name}: {exc}")
            continue

        feed = feedparser.parse(payload)
        added_for_feed = 0

        for entry in feed.entries:
            if added_for_feed >= MAX_STORIES_PER_FEED:
                break

            source = get_entry_source(entry, feed_name)
            title = clean_title(getattr(entry, "title", ""), source)
            link = collapse_whitespace(getattr(entry, "link", ""))
            summary = collapse_whitespace(
                getattr(entry, "summary", "") or getattr(entry, "description", "")
            )[:320]
            topic_score = story_topic_score(extract_entry_text(entry))
            published_raw = (
                collapse_whitespace(getattr(entry, "published", ""))
                or collapse_whitespace(getattr(entry, "updated", ""))
                or collapse_whitespace(getattr(entry, "created", ""))
            )

            published_at = parse_published(published_raw)
            if not title or not link or is_low_signal(title) or topic_score <= 0:
                continue
            if published_at < cutoff:
                continue

            stories.append(
                {
                    "title": title,
                    "source": source or "Unknown source",
                    "link": link,
                    "summary": summary,
                    "published_raw": published_raw or "Unknown time",
                    "published_at": published_at,
                    "published_iso": published_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "feed": feed_name,
                    "topic_score": topic_score,
                }
            )
            added_for_feed += 1

        print(f"Fetched {added_for_feed} candidates from: {feed_name}")

    ranked = sorted(
        stories,
        key=lambda item: (item["published_at"], item["topic_score"]),
        reverse=True,
    )

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()

    for story in ranked:
        key = story_key(story)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(story)

    trimmed = prioritize_source_diversity(deduped)
    print(f"Kept {len(trimmed)} total candidate stories after dedupe")
    return trimmed


def build_digest_prompt(stories: list[dict[str, Any]]) -> str:
    lines = []
    for index, story in enumerate(stories, start=1):
        lines.append(f"{index}. {story['title']}")
        lines.append(f"   Source: {story['source']}")
        lines.append(f"   Published: {story['published_iso']}")
        lines.append(f"   Feed: {story['feed']}")
        if story.get("summary"):
            lines.append(f"   Summary: {story['summary']}")
        lines.append(f"   Topic score: {story['topic_score']}")
        lines.append(f"   Link: {story['link']}")
    return "\n".join(lines)


def prioritize_source_diversity(
    stories: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    preferred: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}

    for story in stories:
        source = story["source"]
        count = source_counts.get(source, 0)

        if count < PREFERRED_STORIES_PER_SOURCE:
            preferred.append(story)
            source_counts[source] = count + 1
            continue

        overflow.append(story)

    return (preferred + overflow)[:MAX_CANDIDATES]


def curate_digest(stories: list[dict[str, Any]]) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    client = OpenAI(api_key=api_key)
    prompt = build_digest_prompt(stories)

    system_message = f"""
You are the editor of {DIGEST_TITLE}, a minimalist daily briefing for someone
who wants to stay current on AI trust, safety, and governance in under
three minutes.

Editorial priorities:
- Prioritize developments in alignment, evaluations, interpretability,
  red teaming, model behavior, safeguards, security, privacy, governance,
  policy, standards, audits, preparedness, and risk management.
- Prefer primary-source updates from labs, evaluators, standards bodies,
  and policy organizations over commentary about them.
- Prefer items that change how frontier models are evaluated, governed,
  secured, deployed, or understood.
- Skip general model launches, product marketing, funding news, and
  corporate announcements unless the core story is explicitly about trust
  and safety.

Return valid JSON with this exact shape:
{{
  "lead": "2-3 sentences, 45-80 words total.",
  "items": [
    {{
      "title": "clean headline",
      "source": "publication name",
      "why_it_matters": "one sentence, max 18 words",
      "link": "https://..."
    }}
  ]
}}

Rules:
- Include every materially important fresh story from the provided list.
- There is no need to keep the digest artificially short.
- Aim for at least {CURATED_ITEM_MIN} items when enough genuinely distinct
  trust-and-safety stories exist.
- If fewer than {CURATED_ITEM_MIN} strong items exist, return fewer.
- Do not omit materially important fresh stories just to stay near five.
- Never exceed {CURATED_ITEM_HARD_MAX} items.
- Use only the provided links and sources.
- Keep the tone neutral, crisp, and non-hyped.
- Prefer the most recent items when multiple candidates are similarly strong.
- Prefer source diversity. Avoid stacking too many items from one
  organization if other strong options exist.
- The lead should synthesize the day's trust-and-safety signal, not list
  the headlines.
- why_it_matters should explain the trust, safety, governance, or risk
  implication for the reader.
- Output JSON only.
""".strip()

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
    )

    raw = response.choices[0].message.content.strip()
    payload = json.loads(raw)

    valid_links = {story["link"] for story in stories}
    stories_by_link = {story["link"]: story for story in stories}
    curated_items: list[dict[str, str]] = []

    for item in payload.get("items", []):
        link = collapse_whitespace(item.get("link", ""))
        if link not in valid_links:
            continue

        story = stories_by_link[link]
        curated_items.append(
            {
                "title": collapse_whitespace(item.get("title", "")),
                "source": collapse_whitespace(item.get("source", "")) or "Unknown source",
                "why_it_matters": collapse_whitespace(item.get("why_it_matters", "")),
                "link": link,
                "published_iso": story["published_iso"],
            }
        )

    if not curated_items:
        raise ValueError("Digest generation returned no valid stories")

    generated_at = datetime.now(timezone.utc)
    return {
        "title": DIGEST_TITLE,
        "tagline": DIGEST_TAGLINE,
        "edition_date": generated_at.strftime("%Y-%m-%d"),
        "generated_at_utc": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lead": collapse_whitespace(payload.get("lead", "")),
        "items": curated_items[:CURATED_ITEM_HARD_MAX],
        "note": DIGEST_NOTE,
        "meta": {
            "story_count_considered": len(stories),
            "scope": "trust_and_safety",
            "recency_window_days": RECENT_WINDOW_DAYS,
            "curated_item_min": CURATED_ITEM_MIN,
            "curated_item_hard_max": CURATED_ITEM_HARD_MAX,
            "preferred_stories_per_source": PREFERRED_STORIES_PER_SOURCE,
            "max_candidates": MAX_CANDIDATES,
            "feeds": [feed["name"] for feed in FEEDS],
        },
    }


def digest_to_markdown(digest: dict[str, Any]) -> str:
    lines = [
        f"# {digest['title']}",
        "",
        digest["tagline"],
        "",
        digest["lead"],
        "",
        "## Worth Opening",
        "",
    ]

    for item in digest["items"]:
        published_iso = collapse_whitespace(item.get("published_iso", ""))
        published_text = f" Published: {published_iso}." if published_iso else ""
        lines.append(
            f"- [{item['title']}]({item['link']}) - {item['why_it_matters']} Source: {item['source']}.{published_text}"
        )

    lines.extend(
        [
            "",
            digest["note"],
            "",
            f"Generated at: {digest['generated_at_utc']}",
        ]
    )
    return "\n".join(lines)


def build_archive_entry(digest: dict[str, Any]) -> dict[str, Any]:
    edition_date = collapse_whitespace(digest.get("edition_date", ""))
    return {
        "date": edition_date,
        "generated_at_utc": collapse_whitespace(digest.get("generated_at_utc", "")),
        "path": f"./archive/{edition_date}.json",
        "item_count": len(digest.get("items", [])),
    }


def load_archive_index(index_path: Path) -> list[dict[str, Any]]:
    if not index_path.exists():
        return []

    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(payload, list):
        return []

    archive_entries: list[dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue

        date = collapse_whitespace(entry.get("date", ""))
        path = collapse_whitespace(entry.get("path", ""))
        generated_at_utc = collapse_whitespace(entry.get("generated_at_utc", ""))
        item_count = entry.get("item_count", 0)

        if not date or not path:
            continue
        if not isinstance(item_count, int) or item_count <= 0:
            continue

        archive_entries.append(
            {
                "date": date,
                "generated_at_utc": generated_at_utc,
                "path": path,
                "item_count": item_count,
            }
        )

    return archive_entries


def write_archive_outputs(digest: dict[str, Any]) -> None:
    if digest.get("sample") or not digest.get("items"):
        return

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    edition_date = digest["edition_date"]
    archive_digest_path = ARCHIVE_DIR / f"{edition_date}.json"
    archive_index_path = ARCHIVE_DIR / "index.json"

    archive_digest_path.write_text(
        json.dumps(digest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    next_entry = build_archive_entry(digest)
    existing_entries = load_archive_index(archive_index_path)
    filtered_entries = [entry for entry in existing_entries if entry.get("date") != edition_date]
    archive_entries = sorted(
        [next_entry, *filtered_entries],
        key=lambda entry: entry["date"],
        reverse=True,
    )
    archive_index_path.write_text(
        json.dumps(archive_entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved archive JSON -> {archive_digest_path}")
    print(f"Saved archive index -> {archive_index_path}")


def write_outputs(digest: dict[str, Any]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    digest_json_path = DOCS_DIR / "digest.json"
    digest_md_path = ROOT / "ai_digest.md"

    digest_json_path.write_text(
        json.dumps(digest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    digest_md_path.write_text(digest_to_markdown(digest) + "\n", encoding="utf-8")
    write_archive_outputs(digest)

    print(f"Saved JSON -> {digest_json_path}")
    print(f"Saved Markdown -> {digest_md_path}")


def main() -> None:
    stories = fetch_candidates()
    if not stories:
        raise RuntimeError("No trust-and-safety stories were fetched from the configured feeds")

    digest = curate_digest(stories)
    write_outputs(digest)
    print("\nAI Trust & Safety Digest update complete.")


if __name__ == "__main__":
    main()
