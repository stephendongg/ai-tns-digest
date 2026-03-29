#!/usr/bin/env python3
"""
AI Research Digest generator.

Fetches recent AI coverage from Google News search feeds, curates the
highest-signal developments into a compact daily digest, and writes the
GitHub Pages assets that power the site.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import feedparser
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"

DIGEST_TITLE = "AI Research Digest"
DIGEST_TAGLINE = "One box, one brief, five links worth your time."
MODEL = "gpt-5.4"
HEADERS = {"User-Agent": "AIResearchDigest/0.1"}
REQUEST_TIMEOUT = 20
MAX_STORIES_PER_QUERY = 15
MAX_CANDIDATES = 24
CURATED_ITEM_COUNT = 5
LANGUAGE_PARAMS = "hl=en-US&gl=US&ceid=US:en"

SEARCH_QUERIES = [
    '"artificial intelligence" when:1d',
    '"large language model" OR LLM OR chatbot when:1d',
    'OpenAI OR Anthropic OR DeepMind OR "Meta AI" OR Mistral OR Nvidia when:1d',
]

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
]


def build_search_feed_url(query: str) -> str:
    return f"https://news.google.com/rss/search?q={quote(query)}&{LANGUAGE_PARAMS}"


def collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def clean_title(title: str, source: str) -> str:
    title = collapse_whitespace(title)
    if source:
        suffix = f" - {source}"
        if title.endswith(suffix):
            title = title[: -len(suffix)].strip()
    return title


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


def is_low_signal(title: str) -> bool:
    title_lower = title.lower()
    return any(re.search(pattern, title_lower) for pattern in LOW_SIGNAL_PATTERNS)


def story_key(story: dict[str, Any]) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", story["title"].lower()).strip()
    source = re.sub(r"[^a-z0-9]+", " ", story["source"].lower()).strip()
    return f"{normalized}|{source}"


def fetch_candidates() -> list[dict[str, Any]]:
    stories: list[dict[str, Any]] = []

    for query in SEARCH_QUERIES:
        url = build_search_feed_url(query)
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        feed = feedparser.parse(response.text)

        added_for_query = 0
        for entry in feed.entries:
            if added_for_query >= MAX_STORIES_PER_QUERY:
                break

            source = ""
            if hasattr(entry, "source") and isinstance(entry.source, dict):
                source = collapse_whitespace(entry.source.get("title") or "")

            title = clean_title(getattr(entry, "title", ""), source)
            link = collapse_whitespace(getattr(entry, "link", ""))
            published_raw = (
                collapse_whitespace(getattr(entry, "published", ""))
                or collapse_whitespace(getattr(entry, "updated", ""))
            )

            if not title or not link or is_low_signal(title):
                continue

            published_at = parse_published(published_raw)
            stories.append(
                {
                    "title": title,
                    "source": source or "Unknown source",
                    "link": link,
                    "published_raw": published_raw or "Unknown time",
                    "published_at": published_at,
                    "published_iso": published_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "query": query,
                }
            )
            added_for_query += 1

        print(f"Fetched {added_for_query} candidates for query: {query}")

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()

    for story in sorted(stories, key=lambda item: item["published_at"], reverse=True):
        key = story_key(story)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(story)

    trimmed = deduped[:MAX_CANDIDATES]
    print(f"Kept {len(trimmed)} total candidate stories after dedupe")
    return trimmed


def build_digest_prompt(stories: list[dict[str, Any]]) -> str:
    lines = []
    for index, story in enumerate(stories, start=1):
        lines.append(f"{index}. {story['title']}")
        lines.append(f"   Source: {story['source']}")
        lines.append(f"   Published: {story['published_iso']}")
        lines.append(f"   Query: {story['query']}")
        lines.append(f"   Link: {story['link']}")
    return "\n".join(lines)


def curate_digest(stories: list[dict[str, Any]]) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    client = OpenAI(api_key=api_key)
    prompt = build_digest_prompt(stories)

    system_message = f"""
You are the editor of {DIGEST_TITLE}, a minimalist daily briefing for someone
who wants to stay current on AI in under three minutes.

Editorial priorities:
- Prioritize developments that materially change the AI landscape: research,
  model launches, major product updates, chips and infrastructure, policy,
  safety, funding, partnerships, and platform strategy.
- Prefer distinct developments over repeated rewrites of the same story.
- Skip low-signal content such as generic explainers, listicles, prompts,
  shallow opinion pieces, and vague trend reports.

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
- Choose exactly {CURATED_ITEM_COUNT} items unless fewer than
  {CURATED_ITEM_COUNT} genuinely distinct stories exist.
- Use only the provided links and sources.
- Keep the tone neutral, crisp, and non-hyped.
- The lead should synthesize the day, not list the headlines.
- why_it_matters should explain why the reader should care today.
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
    curated_items: list[dict[str, str]] = []

    for item in payload.get("items", []):
        link = collapse_whitespace(item.get("link", ""))
        if link not in valid_links:
            continue

        curated_items.append(
            {
                "title": collapse_whitespace(item.get("title", "")),
                "source": collapse_whitespace(item.get("source", "")) or "Unknown source",
                "why_it_matters": collapse_whitespace(item.get("why_it_matters", "")),
                "link": link,
            }
        )

    if not curated_items:
        raise ValueError("Digest generation returned no valid stories")

    generated_at = datetime.now(timezone.utc)
    return {
        "title": DIGEST_TITLE,
        "tagline": DIGEST_TAGLINE,
        "generated_at_utc": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lead": collapse_whitespace(payload.get("lead", "")),
        "items": curated_items[:CURATED_ITEM_COUNT],
        "note": "Curated from Google News AI search results and summarized with GPT-5.4.",
        "meta": {
            "story_count_considered": len(stories),
            "queries": SEARCH_QUERIES,
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
        lines.append(
            f"- [{item['title']}]({item['link']}) - {item['why_it_matters']} Source: {item['source']}."
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


def write_outputs(digest: dict[str, Any]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    digest_json_path = DOCS_DIR / "digest.json"
    digest_md_path = ROOT / "ai_digest.md"

    digest_json_path.write_text(
        json.dumps(digest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    digest_md_path.write_text(digest_to_markdown(digest) + "\n", encoding="utf-8")

    print(f"Saved JSON -> {digest_json_path}")
    print(f"Saved Markdown -> {digest_md_path}")


def main() -> None:
    stories = fetch_candidates()
    if not stories:
        raise RuntimeError("No AI stories were fetched from Google News")

    digest = curate_digest(stories)
    write_outputs(digest)
    print("\nAI Research Digest update complete.")


if __name__ == "__main__":
    main()
