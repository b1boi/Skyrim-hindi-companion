"""
UESP Wiki Dump Stream Parser & SQLite Full-Text Search (FTS5) Indexer.

Parses official UESP XML dumps (.xml.bz2) streamingly with zero memory bloat,
extracting clean prose for Skyrim & Lore namespaces into a sub-millisecond SQLite FTS5 database.
"""

from __future__ import annotations

import argparse
import bz2
import logging
import os
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

logger = logging.getLogger(__name__)

# Canonical UESP namespaces relevant for Skyrim & Elder Scrolls Lore
TARGET_NAMESPACES = {
    "134": "Skyrim",
    "130": "Lore",
    "146": "Dragonborn",
    "140": "Books",
}


class UESPWikiDumpIndexer:
    """Streams UESP bz2 XML dump and builds a high-speed SQLite FTS5 knowledge index."""

    def __init__(
        self,
        dump_path: Optional[Union[str, Path]] = None,
        db_path: Optional[Union[str, Path]] = None,
    ):
        self.dump_path = (
            Path(dump_path)
            if dump_path
            else repo_root / "uespwiki-2026-10-25-current.xml.bz2"
        )
        self.db_path = (
            Path(db_path)
            if db_path
            else repo_root / "data" / "uesp_knowledge_base.db"
        )

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def init_database(self) -> sqlite3.Connection:
        """Creates SQLite tables and FTS5 virtual search index."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA journal_mode = WAL;")

        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT UNIQUE,
                    clean_title TEXT,
                    namespace TEXT,
                    summary TEXT,
                    url TEXT
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_clean_title ON pages(clean_title COLLATE NOCASE);
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
                    clean_title,
                    summary,
                    content='pages',
                    content_rowid='id'
                );
            """)
            # Triggers to keep FTS index synced
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS pages_ai AFTER INSERT ON pages BEGIN
                    INSERT INTO pages_fts(rowid, clean_title, summary) VALUES (new.id, new.clean_title, new.summary);
                END;
            """)
        return conn

    @staticmethod
    def clean_wikitext(raw_text: str) -> str:
        """Extracts clean, readable prose from MediaWiki wikitext."""
        if not raw_text:
            return ""

        # Remove HTML comments
        text = re.sub(r"<!--.*?-->", "", raw_text, flags=re.DOTALL)

        # Remove ref tags
        text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
        text = re.sub(r"<ref[^>]*/>", "", text)
        text = re.sub(r"<[^>]+>", "", text)

        # Remove MediaWiki tables {| ... |}
        text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)

        # Remove nested templates {{ ... }}
        for _ in range(2):
            text = re.sub(r"\{\{[^{}]*\}\}", "", text)

        # Remove files, images, categories
        text = re.sub(r"\[\[(?:File|Image|Category):[^\]]+\]\]", "", text, flags=re.IGNORECASE)

        # Normalize internal wiki links [[Target|Anchor]] -> Anchor, [[Target]] -> Target
        text = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", text)
        text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

        # Remove bold / italic markup
        text = re.sub(r"'{2,5}", "", text)

        # Extract only prose paragraphs (filter out tables, infobox lines, bullet lists)
        paragraphs = []
        for line in text.split("\n"):
            line = line.strip()
            if len(line) >= 30 and not line.startswith(("|", "!", "{", "}", "*", "#", ";")):
                paragraphs.append(line)

        prose = " ".join(paragraphs[:3])
        prose = re.sub(r"\s+", " ", prose).strip()
        return prose

    def stream_pages(
        self, max_pages: Optional[int] = None
    ) -> Generator[Dict[str, str], None, None]:
        """
        Streams relevant pages from the compressed XML dump using iterparse.
        Yields clean page dictionaries without loading the full file into memory.
        """
        if not self.dump_path.exists():
            raise FileNotFoundError(f"UESP dump file not found at: {self.dump_path}")

        ns_prefix = "{http://www.mediawiki.org/xml/export-0.10/}"
        title_tag = f"{ns_prefix}title"
        ns_tag = f"{ns_prefix}ns"
        redirect_tag = f"{ns_prefix}redirect"
        rev_tag = f"{ns_prefix}revision"
        text_tag = f"{ns_prefix}text"

        count = 0

        with bz2.open(str(self.dump_path), "rb") as f:
            for event, elem in ET.iterparse(f, events=("end",)):
                if elem.tag.endswith("page"):
                    # Skip redirects
                    if elem.find(redirect_tag) is not None:
                        elem.clear()
                        continue

                    title_el = elem.find(title_tag)
                    ns_el = elem.find(ns_tag)

                    if title_el is not None and title_el.text:
                        raw_title = title_el.text.strip()
                        ns_id = ns_el.text.strip() if ns_el is not None and ns_el.text else ""

                        # Filter for Skyrim / Lore namespaces
                        if ns_id in TARGET_NAMESPACES or raw_title.startswith(("Skyrim:", "Lore:", "Dragonborn:")):
                            rev_el = elem.find(rev_tag)
                            text_el = rev_el.find(text_tag) if rev_el is not None else None
                            raw_text = text_el.text if text_el is not None and text_el.text else ""

                            clean_summary = self.clean_wikitext(raw_text)

                            if clean_summary and len(clean_summary) > 40:
                                # Clean title: strip namespace prefix
                                clean_title = re.sub(r"^(?:Skyrim|Lore|Dragonborn|Books):", "", raw_title).strip()
                                clean_title = clean_title.replace("_", " ")

                                ns_name = TARGET_NAMESPACES.get(ns_id, raw_title.split(":")[0])

                                count += 1
                                yield {
                                    "title": raw_title,
                                    "clean_title": clean_title,
                                    "namespace": ns_name,
                                    "summary": clean_summary,
                                    "url": f"https://en.uesp.net/wiki/{raw_title.replace(' ', '_')}",
                                }

                                if max_pages and count >= max_pages:
                                    elem.clear()
                                    break

                    elem.clear()

    def build_index(self, max_pages: Optional[int] = None, batch_size: int = 500) -> Dict[str, Any]:
        """Builds or updates the SQLite FTS5 index from the XML dump."""
        t_start = time.perf_counter()
        conn = self.init_database()

        logger.info(f"Starting indexing from {self.dump_path.name} -> {self.db_path.name}...")
        print(f"Indexing UESP dump: {self.dump_path.name}")

        batch = []
        indexed_count = 0

        insert_sql = """
            INSERT OR REPLACE INTO pages (title, clean_title, namespace, summary, url)
            VALUES (?, ?, ?, ?, ?);
        """

        try:
            for page in self.stream_pages(max_pages=max_pages):
                batch.append((
                    page["title"],
                    page["clean_title"],
                    page["namespace"],
                    page["summary"],
                    page["url"],
                ))

                if len(batch) >= batch_size:
                    with conn:
                        conn.executemany(insert_sql, batch)
                    indexed_count += len(batch)
                    batch.clear()
                    print(f"  Indexed {indexed_count} pages...", end="\r", flush=True)

            if batch:
                with conn:
                    conn.executemany(insert_sql, batch)
                indexed_count += len(batch)

        finally:
            conn.close()

        t_elapsed = time.perf_counter() - t_start
        print(f"\nIndexed {indexed_count} canonical UESP pages in {t_elapsed:.1f}s.")
        logger.info(f"Database build complete: {indexed_count} pages in {t_elapsed:.1f}s.")

        return {
            "indexed_pages": indexed_count,
            "elapsed_sec": round(t_elapsed, 2),
            "db_path": str(self.db_path),
        }

    def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Sub-millisecond lore retrieval using exact title index and FTS5 ranking.
        """
        if not self.db_path.exists():
            return []

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        results = []

        query_clean = query.strip()

        try:
            # 1. Exact match on clean_title
            exact_rows = conn.execute(
                "SELECT title, clean_title, namespace, summary, url FROM pages WHERE clean_title = ? COLLATE NOCASE LIMIT 1;",
                (query_clean,),
            ).fetchall()

            for r in exact_rows:
                results.append(dict(r))

            # 2. If no exact match or limit > 1, execute FTS5 search
            if len(results) < limit:
                # Sanitize query for FTS5 (remove special characters)
                sanitized_q = re.sub(r'["*^~:]', ' ', query_clean).strip()
                words = [w for w in sanitized_q.split() if len(w) > 2]

                if words:
                    fts_query = " OR ".join(f'"{w}"*' for w in words[:4])
                    fts_sql = """
                        SELECT p.title, p.clean_title, p.namespace, p.summary, p.url
                        FROM pages_fts f
                        JOIN pages p ON f.rowid = p.id
                        WHERE pages_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?;
                    """
                    try:
                        fts_rows = conn.execute(fts_sql, (fts_query, limit - len(results))).fetchall()
                        for r in fts_rows:
                            d = dict(r)
                            if d["title"] not in [x["title"] for x in results]:
                                results.append(d)
                    except Exception as fts_err:
                        logger.debug(f"FTS5 query failed: {fts_err}")

        finally:
            conn.close()

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UESP XML Dump Indexer for Skyrim AI Companion")
    parser.add_argument("--dump", type=str, default=None, help="Path to uespwiki XML bz2 dump")
    parser.add_argument("--db", type=str, default=None, help="Path to output SQLite database")
    parser.add_argument("--index", action="store_true", help="Build or update the SQLite FTS5 index")
    parser.add_argument("--max_pages", type=int, default=None, help="Max pages to index")
    parser.add_argument("--query", type=str, default=None, help="Search indexed lore")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    indexer = UESPWikiDumpIndexer(dump_path=args.dump, db_path=args.db)

    if args.index:
        res = indexer.build_index(max_pages=args.max_pages)
        print(f"Index built successfully at {res['db_path']}")
    elif args.query:
        matches = indexer.search(args.query)
        print(f"\n--- Search results for '{args.query}' ({len(matches)} found) ---")
        for idx, m in enumerate(matches, 1):
            print(f"[{idx}] {m['clean_title']} ({m['namespace']}):")
            print(f"    Summary: {m['summary'][:200]}...")
            print(f"    URL:     {m['url']}")
    else:
        print("Usage:")
        print("  python3 src/uesp_indexer.py --index [--max_pages 2000]")
        print("  python3 src/uesp_indexer.py --query 'Alduin'")

