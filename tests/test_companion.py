"""
Unit Tests for Skyrim Hindi AI Companion — 'सरकार भैया' (Sarkar Bhaiya).

Tests:
1. Persona Engine & Directive Formatting (src/persona.py)
2. Deterministic Profanity Budgeting & Seed Reproducibility
3. Gameplay Event Severity Mappings & Credit Theft
4. Output Validation & Cleaning Guard (src/validator.py)
5. Streaming Sentence Chunker & FIFO Playback Queue (src/tts_streamer.py)
6. Companion LLM & In-Character Fallback (src/llm.py)
7. End-to-End Pipeline & Telemetry Structure (src/pipeline.py)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.llm import SkyrimCompanionLLM
from src.persona import (
    GAMEPLAY_EVENTS,
    PROFANITY_VOCAB,
    ProfanitySeverity,
    SARKAR_BHAIYA_SYSTEM_PROMPT,
    SarkarBhaiyaEngine,
)
from src.pipeline import SkyrimHindiPipeline
from src.tts_streamer import StreamingSentenceChunker, ThreadSafeAudioPlaybackQueue
from src.validator import CompanionOutputValidator, companion_validator


class TestSarkarBhaiyaPersona(unittest.TestCase):
    """Tests Sarkar Bhaiya persona engine, events, and deterministic profanity."""

    def setUp(self):
        self.engine = SarkarBhaiyaEngine(profanity_chance=0.33, meltdown_chance=0.02, seed=42)

    def test_system_prompt_present(self):
        prompt = self.engine.get_system_prompt()
        self.assertIn("सरकार भैया", prompt)
        self.assertIn("भाइयो और बहनो", prompt)
        self.assertIn("ऐतिहासिक क्षण", prompt)

    def test_deterministic_profanity_rate(self):
        """Verify that over 1000 trials, profanity frequency approximates ~33% with fixed seed."""
        self.engine.set_seed(12345)
        trials = 1000
        count = 0
        for _ in range(trials):
            fired, word, sev = self.engine.select_profanity(event_name="missed_attack")
            if fired:
                count += 1
                self.assertIsNotNone(word)
                self.assertIn(sev, [ProfanitySeverity.LIGHT, ProfanitySeverity.MEDIUM, ProfanitySeverity.HEAVY])

        rate = count / trials
        # Should be closely bounded around 0.33
        self.assertTrue(0.28 <= rate <= 0.38, f"Profanity rate {rate} outside expected ~0.33 window")

    def test_event_severity_rules(self):
        """Verify that event-specific severity limits are respected."""
        # 'greeting' only allows LIGHT insults when profanity is forced
        for _ in range(30):
            fired, word, sev = self.engine.select_profanity(
                event_name="greeting", force_profanity=True
            )
            if fired:
                self.assertIn(word, PROFANITY_VOCAB[ProfanitySeverity.LIGHT])
                self.assertEqual(sev, ProfanitySeverity.LIGHT)

        # 'player_damaged' allows LIGHT or MEDIUM insults
        for _ in range(30):
            fired, word, sev = self.engine.select_profanity(
                event_name="player_damaged", force_profanity=True
            )
            if fired:
                allowed = PROFANITY_VOCAB[ProfanitySeverity.LIGHT] + PROFANITY_VOCAB[ProfanitySeverity.MEDIUM]
                self.assertIn(word, allowed)
                self.assertIn(sev, [ProfanitySeverity.LIGHT, ProfanitySeverity.MEDIUM])

        # Test explicit force_severity
        fired, word, sev = self.engine.select_profanity(
            event_name="player_damaged", force_profanity=True, force_severity=ProfanitySeverity.MEDIUM
        )
        self.assertTrue(fired)
        self.assertIn(word, PROFANITY_VOCAB[ProfanitySeverity.MEDIUM])
        self.assertEqual(sev, ProfanitySeverity.MEDIUM)

    def test_credit_steal_and_directive_formatting(self):
        """Verify that successful_kill triggers credit stealing directive."""
        prompt, meta = self.engine.format_user_prompt_with_directive(
            "शत्रु मर गया!", event_name="successful_kill"
        )
        self.assertIn("इवेंट: successful_kill", meta["directive_str"])
        self.assertTrue(meta["credit_steal"])
        self.assertIn("क्रेडिट: मुख्य श्रेय खुद लो", meta["directive_str"])
        self.assertIn("खिलाड़ी: शत्रु मर गया!", prompt)


class TestCompanionOutputValidator(unittest.TestCase):
    """Tests validation of Devanagari Hindi text and cleaning of artifacts."""

    def setUp(self):
        self.validator = CompanionOutputValidator()

    def test_valid_hindi_passes(self):
        valid_text = "मित्रों, देखिए हमारे महान योद्धा को। क्या बात है!"
        res = self.validator.validate(valid_text)
        self.assertTrue(res.is_valid)
        self.assertEqual(len(res.issues), 0)
        self.assertFalse(res.retry_needed)

    def test_brackets_and_emojis_flagged_and_cleaned(self):
        dirty_text = "मित्रों, देखिए [हंसते हुए] इस गांडू को! 😀"
        res = self.validator.validate(dirty_text)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("bracket" in i for i in res.issues))
        self.assertTrue(any("emoji" in i for i in res.issues))
        self.assertEqual(res.cleaned_text, "मित्रों, देखिए इस गांडू को!")
        self.assertIn("कोई ब्रैकेट", res.retry_prompt)

    def test_latin_characters_flagged_and_cleaned(self):
        latin_text = "मित्रों, यह तो pure failure है!"
        res = self.validator.validate(latin_text)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("Latin" in i for i in res.issues))
        self.assertIn("केवल और केवल शुद्ध देवनागरी", res.retry_prompt)

    def test_punctuation_normalization(self):
        text_with_dots = "मित्रों, यह ऐतिहासिक क्षण है. हम जीत गए."
        cleaned = self.validator.clean(text_with_dots)
        self.assertNotIn(".", cleaned)
        self.assertIn("।", cleaned)


class TestStreamingChunkerAndQueue(unittest.TestCase):
    """Tests sentence-level streaming token buffer and audio FIFO queue."""

    def test_sentence_chunking_on_punctuation(self):
        chunker = StreamingSentenceChunker(min_chars=15)
        tokens = ["मित्रों", ", ", "देखिए", " ", "हमारे", " ", "योद्धा", " ", "को। ", "हम ", "आगे ", "बढ़ेंगे!"]
        chunks = []
        for t in tokens:
            chunks.extend(chunker.feed(t))
        chunks.extend(chunker.flush())

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "मित्रों, देखिए हमारे योद्धा को।")
        self.assertEqual(chunks[1], "हम आगे बढ़ेंगे!")

    def test_fifo_playback_queue(self):
        queue = ThreadSafeAudioPlaybackQueue(sample_rate=24000, mock_playback=True)
        queue.start()

        chunk1 = np.ones(1200, dtype=np.float32) * 0.5
        chunk2 = np.ones(1200, dtype=np.float32) * -0.5

        queue.enqueue(chunk1)
        queue.enqueue(chunk2)
        queue.finish_stream()
        queue.wait_until_complete()

        full_audio = queue.get_full_audio()
        self.assertEqual(len(full_audio), 2400)
        self.assertAlmostEqual(full_audio[0], 0.5)
        self.assertAlmostEqual(full_audio[1200], -0.5)


class TestCompanionLLMAndPipeline(unittest.TestCase):
    """Tests LLM fallback generation and end-to-end pipeline execution."""

    def setUp(self):
        self.llm = SkyrimCompanionLLM(rng_seed=42)
        self.pipeline = SkyrimHindiPipeline(mock_playback=True)

    def test_llm_fallback_theatrical_tone(self):
        res = self.llm.generate("सामने ट्रोल है!", event="danger")
        self.assertTrue(res["is_fallback"])
        self.assertIn("राष्ट्र के नागरिकों", res["text"])
        self.assertIn("ट्रोल", res["text"])
        self.assertEqual(res["event"], "danger")

    def test_pipeline_turn_metrics(self):
        res = self.pipeline.process_turn(
            user_input_text="व्हाइटरन चलें?",
            event="greeting",
            play_audio=True,
            save_audio=False,
            streaming=True,
        )

        expected_keys = [
            "player_input",
            "stt_latency_ms",
            "stt_bypassed",
            "event",
            "normalized_tts_text",
            "norm_latency_ms",
            "llm_ttft_ms",
            "llm_total_ms",
            "tts_ttfa_ms",
            "first_audible_ms",
            "tts_total_ms",
            "e2e_latency_ms",
        ]
        for key in expected_keys:
            self.assertIn(key, res, f"Metric key '{key}' missing from pipeline turn summary")

        self.assertTrue(res["stt_bypassed"])
        self.assertGreater(len(res["normalized_tts_text"]), 0)


if __name__ == "__main__":
    unittest.main()
