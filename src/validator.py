"""
Hindi Output Validator and Normalization Guard for Sarkar Bhaiya Companion.

Ensures LLM output strictly adheres to persona constraints:
1. Pure Devanagari Hindi (no Latin characters, no English words).
2. No bracketed stage directions or action descriptions ([हंसते हुए], (looking around)).
3. No emojis or special symbols.
4. No internal reasoning or special tags (<think>, <s>, etc.).
5. Valid Devanagari punctuation (danda '।', '?', '!', ellipses '...').
6. Non-empty, speakable dialogue.
7. Controlled retry instruction generation for at most 1 regeneration attempt.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# Emojis regex pattern matching standard emoji ranges
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)

# Bracketed directions pattern (square brackets, parentheses, curly braces)
BRACKET_PATTERN = re.compile(r"(\[.*?\]|\(.*?\)|<.*?>|\{.*?\})", flags=re.DOTALL)

# Latin alphabet pattern
LATIN_PATTERN = re.compile(r"[a-zA-Z]")

# Special tokens pattern
SPECIAL_TOKENS_PATTERN = re.compile(
    r"(<think>.*?</think>|<think>.*|</think>|<s>|</s>|<pad>|<unk>|<\|.*?\|>)",
    flags=re.DOTALL | re.IGNORECASE,
)


@dataclass
class ValidationResult:
    """Detailed result of companion output validation."""
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    cleaned_text: str = ""
    retry_needed: bool = False
    retry_prompt: Optional[str] = None


class CompanionOutputValidator:
    """Validates and cleans generated companion utterances."""

    def __init__(self, allow_mild_latin: bool = False):
        self.allow_mild_latin = allow_mild_latin

    def validate(self, text: str) -> ValidationResult:
        """
        Validates the raw LLM output against persona speech rules.
        Returns ValidationResult with issues identified and cleaned text.
        """
        if not text or not text.strip():
            return ValidationResult(
                is_valid=False,
                issues=["Empty response"],
                cleaned_text="",
                retry_needed=True,
                retry_prompt="तुम्हारा उत्तर खाली था। कृपया सीधे और तुरंत देवनागरी हिंदी में संवाद बोलो।",
            )

        issues: List[str] = []

        # 1. Check special tokens / reasoning tags
        if re.search(SPECIAL_TOKENS_PATTERN, text):
            issues.append("Contains special/internal tokens or <think> tags")

        # 2. Check bracketed stage directions
        bracket_matches = BRACKET_PATTERN.findall(text)
        if bracket_matches:
            issues.append(f"Contains bracketed stage directions: {bracket_matches[:3]}")

        # 3. Check emojis
        emoji_matches = EMOJI_PATTERN.findall(text)
        if emoji_matches:
            issues.append("Contains emojis")

        # 4. Check Latin characters
        latin_matches = LATIN_PATTERN.findall(text)
        if latin_matches and not self.allow_mild_latin:
            issues.append(f"Contains Latin/English characters: {len(latin_matches)} characters found")

        # 5. Check Devanagari presence
        has_devanagari = any("\u0900" <= c <= "\u097f" for c in text)
        if not has_devanagari:
            issues.append("No Devanagari script detected")

        cleaned = self.clean(text)

        # Re-check cleaned text length
        if len(cleaned.strip()) < 3:
            issues.append("Cleaned text is too short or empty after removing artifacts")

        is_valid = len(issues) == 0

        retry_prompt = None
        if not is_valid:
            retry_prompt = self._build_retry_prompt(issues)

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            cleaned_text=cleaned,
            retry_needed=not is_valid,
            retry_prompt=retry_prompt,
        )

    def clean(self, text: str) -> str:
        """
        Cleans and normalizes text for speech synthesis:
        - Strips <think>...</think> and internal tags
        - Strips bracketed stage directions
        - Strips emojis
        - Converts Latin artifacts or drops them
        - Normalizes punctuation (replaces '.' with '।')
        """
        if not text:
            return ""

        s = text

        # Strip reasoning tags
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
        if "<think>" in s:
            s = re.sub(r"<think>.*", "", s, flags=re.DOTALL)
        s = s.replace("</think>", "")

        # Strip special token markers
        s = re.sub(r"<\|.*?\|>", "", s)
        s = re.sub(r"</?[a-zA-Z0-9_\-]+>", "", s)

        # Strip bracketed content [हंसते हुए], (मुस्कुराते हुए)
        s = re.sub(r"\[.*?\]", "", s)
        s = re.sub(r"\(.*?\)", "", s)
        s = re.sub(r"\{.*?\}", "", s)

        # Strip emojis
        s = EMOJI_PATTERN.sub("", s)

        # Remove Latin words / characters if any leftover
        # Replace common latin quotes with unicode
        s = s.replace('"', '').replace("'", "")
        # Remove remaining latin words
        s = re.sub(r"[a-zA-Z]+", "", s)

        # Normalize punctuation:
        # Full stops to Devanagari purna viram (।)
        s = re.sub(r"\.(?!\.)", "।", s)
        # Normalize ellipses (... or …)
        s = re.sub(r"\.{2,}", "...", s)
        s = re.sub(r"…+", "...", s)
        # Normalize multiple dandas
        s = re.sub(r"।{2,}", "।", s)
        # Normalize multiple exclamation / question marks
        s = re.sub(r"!{2,}", "!", s)
        s = re.sub(r"\?{2,}", "?", s)

        # Remove conflicting punctuation like '! ।' or '? ।'
        s = re.sub(r"([!?])\s*[।.]+", r"\1", s)
        s = re.sub(r"[।.]+\s*([!?])", r"\1", s)

        # Clean excess whitespace
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n+", " ", s)
        s = s.strip()

        # Ensure terminal punctuation if missing
        if s and s[-1] not in "।?!.":
            s += "।"

        return s

    def _build_retry_prompt(self, issues: List[str]) -> str:
        """Generates an explicit correction directive for regeneration retry."""
        reasons = []
        for issue in issues:
            if "Latin" in issue or "Devanagari" in issue:
                reasons.append("केवल और केवल शुद्ध देवनागरी हिंदी में बोलें, कोई अंग्रेजी शब्द या अक्षर नहीं")
            elif "bracket" in issue:
                reasons.append("कोई ब्रैकेट या स्टेज डायरेक्शन [जैसे ...] बिल्कुल न लिखें")
            elif "emoji" in issue:
                reasons.append("कोई इमोजी न लिखें")
            elif "special" in issue or "<think>" in issue:
                reasons.append("सीधे बोलने योग्य संवाद दें, कोई <think> या आंतरिक सोच न लिखें")

        instructions = "; ".join(reasons) if reasons else "सीधे और तुरंत देवनागरी हिंदी में एक छोटा बोलने योग्य संवाद दें।"
        return f"[सुधार निर्देश: पिछले उत्तर में त्रुटि थी ({instructions})। पुनः सरकार भैया के रूप में केवल शुद्ध देवनागरी हिंदी में 1-2 वाक्य बोलें।]"


# Global validator instance
companion_validator = CompanionOutputValidator()
