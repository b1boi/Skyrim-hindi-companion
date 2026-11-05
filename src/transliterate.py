"""
Hinglish and Skyrim Lore Normalizer for Hindi TTS.

Transforms English/Hinglish Skyrim proper nouns, numbers, and Latin script artifacts
into phonetically accurate Devanagari script for optimal synthesis with XTTS-v2.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class SkyrimHindiNormalizer:
    """Normalizes mixed English/Hinglish game text into clean Devanagari for TTS synthesis."""

    def __init__(self, lore_dict_path: Optional[str | Path] = None):
        self.lore_dict_path = (
            Path(lore_dict_path)
            if lore_dict_path
            else Path(__file__).resolve().parent.parent / "data" / "lore_transliteration.json"
        )
        self.term_mapping: Dict[str, str] = {}
        self.number_mapping: Dict[str, str] = {}
        self._compiled_patterns: list[Tuple[re.Pattern, str]] = []

        self._load_dictionary()

    def _load_dictionary(self) -> None:
        """Loads and compiles regex patterns from the transliteration dictionary."""
        if not self.lore_dict_path.exists():
            logger.warning(
                f"Lore dictionary not found at {self.lore_dict_path}. Using fallback rule set."
            )
            self._load_fallbacks()
            return

        try:
            with open(self.lore_dict_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.term_mapping = data.get("terms", {})
            self.number_mapping = data.get("number_words", {})
            self._compile_regexes()
            logger.info(
                f"Loaded {len(self.term_mapping)} lore terms and {len(self.number_mapping)} numbers."
            )
        except Exception as e:
            logger.error(f"Failed to load lore dictionary: {e}. Falling back to default rules.")
            self._load_fallbacks()

    def _load_fallbacks(self) -> None:
        """Basic fallback map if dictionary file is missing."""
        self.term_mapping = {
            "Skyrim": "स्काईरिम",
            "Dovahkiin": "दोवाकीन",
            "Dragonborn": "ड्रैगनबॉर्न",
            "Whiterun": "व्हाइटरन",
            "Alduin": "एल्डुइन",
            "Jarl": "जारल",
            "Balgruuf": "बालग्रूफ़",
        }
        self.number_mapping = {
            "1": "एक",
            "2": "दो",
            "3": "तीन",
            "4": "चार",
            "5": "पाँच",
            "10": "दस",
            "100": "सौ",
        }
        self._compile_regexes()

    def _compile_regexes(self) -> None:
        """Sorts patterns by length descending to replace multi-word phrases first."""
        self._compiled_patterns.clear()

        # Sort terms by length in descending order to prioritize longer matches
        sorted_terms = sorted(self.term_mapping.items(), key=lambda item: len(item[0]), reverse=True)

        for eng, dev in sorted_terms:
            # Word boundary pattern for English words
            pattern = re.compile(rf"\b{re.escape(eng)}\b", flags=re.IGNORECASE)
            self._compiled_patterns.append((pattern, dev))

    def normalize_numbers(self, text: str) -> str:
        """Converts digit sequences to spoken Hindi words where defined."""
        # Multi-digit exact matches first
        sorted_numbers = sorted(
            self.number_mapping.items(), key=lambda item: len(item[0]), reverse=True
        )
        for num_str, word in sorted_numbers:
            pattern = re.compile(rf"\b{re.escape(num_str)}\b")
            text = pattern.sub(word, text)
        return text

    def clean_text_artifacts(self, text: str) -> str:
        """Removes markdown artifacts, parentheticals, and unwanted symbols."""
        # Remove actions or stage directions in brackets, e.g. [whispering] or (laughs)
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\(.*?\)", "", text)

        # Remove markdown formatting (*italics*, **bold**, `code`, #headers)
        text = re.sub(r"[*_~`#]", "", text)

        # Normalize multiple spaces and newlines
        text = re.sub(r"\s+", " ", text).strip()

        # Normalize western sentence periods to Devanagari purna viram (।) when preceded by Devanagari
        text = re.sub(r"([\u0900-\u097F])\s*\.", r"\1 ।", text)

        return text

    def normalize(self, text: str) -> str:
        """
        Executes full normalization pipeline on input string:
        1. Strips meta artifacts and stage directions
        2. Replaces Skyrim lore terms with Devanagari equivalents
        3. Converts numeric digits to Hindi spoken words
        4. Cleans trailing punctuation and spacing for speech synthesis
        """
        if not text:
            return ""

        # Step 1: Clean artifacts
        cleaned = self.clean_text_artifacts(text)

        # Step 2: Replace lore terms
        for pattern, replacement in self._compiled_patterns:
            cleaned = pattern.sub(replacement, cleaned)

        # Step 3: Replace numbers
        cleaned = self.normalize_numbers(cleaned)

        # Step 4: Ensure clean speech boundary
        cleaned = cleaned.strip()
        if cleaned and not cleaned[-1] in ("।", "?", "!", "."):
            cleaned += " ।"

        return cleaned

    def contains_latin(self, text: str) -> bool:
        """Returns True if the text still contains Latin alphabet characters."""
        return bool(re.search(r"[a-zA-Z]", text))


if __name__ == "__main__":
    import sys

    normalizer = SkyrimHindiNormalizer()
    sample = (
        "Hello Dovahkiin! We must travel to Whiterun to warn Jarl Balgruuf about Alduin. "
        "I have 500 gold for the carriage."
    )
    if len(sys.argv) > 1:
        sample = " ".join(sys.argv[1:])

    normalized = normalizer.normalize(sample)
    print("--- Transliteration Demo ---")
    print(f"Input:      {sample}")
    print(f"Normalized: {normalized}")
    print(f"Contains Latin remnants: {normalizer.contains_latin(normalized)}")

