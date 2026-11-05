"""
UESP (Unofficial Elder Scrolls Pages) Knowledge Base & Lore Retriever.

Source: https://en.uesp.net/wiki/
Integrates canonical Elder Scrolls lore into the companion's context window.
Supports offline local cached lore + dynamic MediaWiki API fetching with auto-caching.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

logger = logging.getLogger(__name__)

UESP_BASE_URL = "https://en.uesp.net/wiki/"
UESP_API_ENDPOINT = "https://en.uesp.net/w/api.php"
UESP_USER_AGENT = "SkyrimHindiCompanion/1.0 (Educational/Mod Research; Skyrim Lore Companion)"


class UESPKnowledgeBase:
    """Manages UESP lore retrieval, caching, and context injection for the LLM."""

    def __init__(
        self,
        cache_path: Optional[Union[str, Path]] = None,
        enable_online_fetch: bool = True,
        timeout_sec: float = 3.5,
    ):
        self.cache_path = (
            Path(cache_path)
            if cache_path
            else repo_root / "data" / "skyrim_lore_kb.json"
        )
        self.enable_online_fetch = enable_online_fetch
        self.timeout_sec = timeout_sec

        self.lore_data: Dict[str, Any] = {}
        self.entities: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Loads cached UESP lore from disk."""
        if not self.cache_path.exists():
            logger.warning(f"UESP cache not found at {self.cache_path}. Creating new index.")
            self.lore_data = {
                "source": UESP_BASE_URL,
                "description": "Skyrim Lore Knowledge Base from UESP",
                "entities": {},
            }
            self.entities = self.lore_data["entities"]
            return

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self.lore_data = json.load(f)
            self.entities = self.lore_data.get("entities", {})
            logger.info(f"Loaded {len(self.entities)} UESP lore entries from {self.cache_path.name}")
        except Exception as e:
            logger.error(f"Failed to load UESP cache: {e}")
            self.entities = {}

    def _save_cache(self) -> None:
        """Saves updated lore cache to disk."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.lore_data["entities"] = self.entities
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.lore_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved {len(self.entities)} lore entries to {self.cache_path}")
        except Exception as e:
            logger.warning(f"Could not save UESP cache: {e}")

    def lookup(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Looks up lore for a given entity query in local cache.
        Supports both English names ('Whiterun') and Hindi Devanagari ('व्हाइटरन').
        """
        query_clean = query.strip()
        query_lower = query_clean.lower()

        # 1. Exact match on entity key
        for key, entry in self.entities.items():
            if key.lower() == query_lower:
                return entry

        # 2. Match on Hindi name or Title
        for key, entry in self.entities.items():
            hindi_name = entry.get("hindi_name", "").lower()
            title = entry.get("title", "").lower()
            if query_lower in hindi_name or query_lower in title or hindi_name in query_lower:
                return entry

        # 3. Substring keyword search across key lore terms
        for key, entry in self.entities.items():
            key_words = key.lower().split()
            if any(w in query_lower for w in key_words if len(w) > 3):
                return entry

        # 4. If not found in cache and online fetching is enabled, try UESP API
        if self.enable_online_fetch:
            return self.fetch_from_uesp_api(query_clean)

        return None

    def fetch_from_uesp_api(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        Fetches canonical intro extract from UESP MediaWiki API.
        URL: https://en.uesp.net/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&titles=...
        """
        # Try both Skyrim: namespace and Lore: namespace
        candidates = [f"Skyrim:{entity_name}", f"Lore:{entity_name}", entity_name]
        headers = {"User-Agent": UESP_USER_AGENT}

        for title in candidates:
            params = {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "titles": title,
                "redirects": 1,
            }
            try:
                resp = requests.get(
                    UESP_API_ENDPOINT,
                    headers=headers,
                    params=params,
                    timeout=self.timeout_sec,
                )
                if resp.status_code != 200:
                    continue

                data = resp.json()
                pages = data.get("query", {}).get("pages", {})
                for page_id, page_info in pages.items():
                    if page_id != "-1" and "extract" in page_info:
                        extract = page_info["extract"].strip()
                        if extract:
                            resolved_title = page_info.get("title", title)
                            entry = {
                                "title": resolved_title,
                                "uesp_url": f"{UESP_BASE_URL}{resolved_title.replace(' ', '_')}",
                                "hindi_name": entity_name,
                                "category": "online_fetched",
                                "summary": extract[:400] + ("..." if len(extract) > 400 else ""),
                                "lore_context_hi": f"स्काईरिम इतिहास: {extract[:200]}",
                            }
                            # Cache the new entity
                            self.entities[entity_name] = entry
                            self._save_cache()
                            logger.info(f"Fetched and cached '{entity_name}' from UESP API.")
                            return entry

            except Exception as e:
                logger.debug(f"UESP API fetch skipped for {title}: {e}")

        return None

    def get_lore_context(self, user_prompt: str, max_chars: int = 300) -> str:
        """
        Extracts relevant UESP lore context for a given player prompt.
        Returns a concise context string suitable for injecting into the LLM prompt.
        """
        prompt_lower = user_prompt.lower()
        matched_entries: List[Dict[str, Any]] = []

        # Find all relevant entities mentioned in user prompt
        for key, entry in self.entities.items():
            key_lower = key.lower()
            hi_name = entry.get("hindi_name", "").lower()

            matches = (
                key_lower in prompt_lower
                or any(w in prompt_lower for w in key_lower.split() if len(w) > 3)
                or (hi_name and (hi_name in prompt_lower or any(p in prompt_lower for p in hi_name.split() if len(p) >= 2)))
            )

            if matches:
                matched_entries.append(entry)

        if not matched_entries:
            return ""

        # Take top 2 matches
        snippets = []
        for entry in matched_entries[:2]:
            hi_lore = entry.get("lore_context_hi")
            summary = entry.get("summary")
            # Prefer Hindi context if available, else summary
            chosen = hi_lore if hi_lore else summary
            if chosen:
                snippets.append(chosen)

        combined = " | ".join(snippets)
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "..."

        return f"[UESP स्काईरिम इतिहास ज्ञान: {combined}]"


if __name__ == "__main__":
    import sys

    kb = UESPKnowledgeBase()
    query = sys.argv[1] if len(sys.argv) > 1 else "Whiterun"
    print(f"--- UESP Lore Lookup for '{query}' ---")
    result = kb.lookup(query)
    if result:
        print(f"Title:    {result.get('title')}")
        print(f"URL:      {result.get('uesp_url')}")
        print(f"Hindi:    {result.get('hindi_name')}")
        print(f"Summary:  {result.get('summary')}")
    else:
        print("Exact title match not found. Checking sentence lore extraction...")

    context = kb.get_lore_context(query)
    print(f"Context:  {context if context else 'None'}")
