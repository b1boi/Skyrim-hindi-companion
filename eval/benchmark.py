"""
Automated Evaluation Harness for Skyrim Hindi Companion — 'सरकार भैया' (Sarkar Bhaiya).

Benchmarks:
  1. Per-stage & End-to-End Latency (ASR, Normalization, LLM TTFT, TTS TTFA, First Audible, Total E2E)
  2. Persona Consistency & Sarkar Bhaiya Oratory Style (Theatrical orator, credit theft, player blame)
  3. Out-Of-Character (OOC) Trap Defense Rate
  4. Script & Speech Purity (Pure Devanagari Hindi, no Latin leakage, no bracketed stage directions)
  5. Distinction between live LM Studio GPU inference vs. offline in-character fallback

Exports:
  - eval/results/latency_breakdown.csv
  - eval/results/persona_evaluation.csv
  - eval/results/benchmark_summary.json
  - Formatted resume bullets ready for documentation
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.pipeline import SkyrimHindiPipeline
from src.validator import companion_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_benchmark")


class BenchmarkHarness:
    """Orchestrates quantitative evaluation across all prompt and gameplay event sets."""

    def __init__(
        self,
        prompts_path: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        mock_playback: bool = True,
    ):
        self.prompts_path = prompts_path or repo_root / "eval" / "test_prompts.json"
        self.results_dir = results_dir or repo_root / "eval" / "results"
        self.audio_samples_dir = self.results_dir / "audio_samples"

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.audio_samples_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline = SkyrimHindiPipeline(mock_playback=mock_playback)
        self.prompts: List[Dict[str, Any]] = self._load_prompts()

    def _load_prompts(self) -> List[Dict[str, Any]]:
        """Loads prompt evaluation definitions."""
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"Prompts file not found at {self.prompts_path}")
        with open(self.prompts_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_persona(
        self,
        prompt_item: Dict[str, Any],
        companion_reply: str,
    ) -> Dict[str, Any]:
        """
        Evaluates companion response against Sarkar Bhaiya persona rubric:
        - In-Character Score (1-5)
        - Theatrical orator markers
        - Credit-stealing / player-blame compliance
        - OOC Resistance (Pass/Fail)
        - Script & Formatting cleanliness (no Latin, no brackets, valid Devanagari)
        """
        category = prompt_item.get("category", "general")
        event = prompt_item.get("event", "generic_conversation")
        rubric = prompt_item.get("evaluation_rubric", {})
        must_include = rubric.get("must_include", [])
        forbidden_terms = rubric.get("forbidden_terms", [])
        expect_credit_steal = rubric.get("credit_steal", False)
        expect_blame_player = rubric.get("blame_player", False)

        reply_lower = companion_reply.lower()

        # 1. Check forbidden terms (OOC leaks)
        leaked_terms = [t for t in forbidden_terms if t.lower() in reply_lower]
        ooc_passed = (len(leaked_terms) == 0)

        # 2. Check in-world lore keywords
        matched_lore = [term for term in must_include if term in companion_reply]
        lore_ratio = (len(matched_lore) / len(must_include)) if must_include else 1.0

        # 3. Output validation (no brackets, no emojis, no Latin)
        val = companion_validator.validate(companion_reply)
        script_clean = val.is_valid

        # 4. Check orator markers
        orator_keywords = [
            "मित्रों", "भाइयो", "बहनो", "ऐतिहासिक", "गौरव", "नागरिकों", "श्रोताओं",
            "दर्शकों", "सरकार भैया", "इतिहास", "प्रहार", "रिकॉर्ड", "उपलब्धि"
        ]
        has_orator_markers = any(k in companion_reply for k in orator_keywords)

        # 5. Check credit stealing / player blame
        credit_stolen = any(k in companion_reply for k in ["हमने", "मेरा", "मेरी", "रणनीति", "प्रहार", "मुख्य रूप से"])
        player_blamed = any(k in companion_reply for k in ["तुम्हारी", "नाकामी", "अनाड़ी", "गांडू", "भोसड़ीके", "चूतिये", "हरामी", "कमीने"])

        # Persona scoring logic (1.0 to 5.0)
        score = 5.0

        if not ooc_passed:
            score -= 3.0  # Major penalty for breaking character
        if not script_clean:
            score -= 0.5  # Deduction for formatting/script issues
        if not has_orator_markers and category != "ooc_trap":
            score -= 0.5  # Deduction for sounding generic rather than theatrical
        if expect_credit_steal and not credit_stolen:
            score -= 0.5
        if expect_blame_player and not player_blamed:
            score -= 0.5

        final_score = max(1.0, min(5.0, round(score, 1)))

        return {
            "persona_score": final_score,
            "ooc_passed": ooc_passed,
            "leaked_forbidden_terms": leaked_terms,
            "matched_lore_terms": matched_lore,
            "lore_coverage": round(lore_ratio * 100.0, 1),
            "script_clean": script_clean,
            "has_orator_markers": has_orator_markers,
            "credit_stolen": credit_stolen,
            "player_blamed": player_blamed,
        }

    def run(self, max_prompts: Optional[int] = None) -> Dict[str, Any]:
        """Executes automated benchmark across test cases."""
        test_items = self.prompts[:max_prompts] if max_prompts else self.prompts
        logger.info(f"Starting evaluation on {len(test_items)} Sarkar Bhaiya test prompts...")

        latency_rows: List[Dict[str, Any]] = []
        persona_rows: List[Dict[str, Any]] = []

        total_e2e_latencies = []
        total_llm_ttft = []
        total_tts_ttfa = []
        total_first_audible = []
        total_persona_scores = []
        ooc_total_tests = 0
        ooc_passed_tests = 0
        fallback_count = 0

        for idx, item in enumerate(test_items, 1):
            prompt_id = item["id"]
            category = item["category"]
            event = item.get("event", "generic_conversation")
            user_text = item["prompt_text"]

            logger.info(f"[{idx}/{len(test_items)}] Testing [{category.upper()} | {event}]: {user_text}")

            out_audio = self.audio_samples_dir / f"{prompt_id}.wav"
            metrics = self.pipeline.process_turn(
                user_input_text=user_text,
                event=event,
                save_audio=True,
                play_audio=False,
                streaming=True,
            )

            if metrics.get("audio_path") and Path(metrics["audio_path"]).exists():
                try:
                    Path(metrics["audio_path"]).replace(out_audio)
                    metrics["audio_path"] = str(out_audio)
                except Exception:
                    pass

            if metrics.get("is_llm_fallback"):
                fallback_count += 1

            persona_eval = self.evaluate_persona(item, metrics["normalized_tts_text"])

            if category == "ooc_trap":
                ooc_total_tests += 1
                if persona_eval["ooc_passed"]:
                    ooc_passed_tests += 1

            total_e2e_latencies.append(metrics["e2e_latency_ms"])
            total_llm_ttft.append(metrics["llm_ttft_ms"])
            total_tts_ttfa.append(metrics["tts_ttfa_ms"])
            total_first_audible.append(metrics.get("first_audible_ms", metrics["tts_ttfa_ms"]))
            total_persona_scores.append(persona_eval["persona_score"])

            latency_rows.append({
                "prompt_id": prompt_id,
                "category": category,
                "event": event,
                "is_llm_fallback": metrics.get("is_llm_fallback", False),
                "stt_latency_ms": metrics["stt_latency_ms"],
                "norm_latency_ms": metrics["norm_latency_ms"],
                "llm_ttft_ms": metrics["llm_ttft_ms"],
                "llm_total_ms": metrics["llm_total_ms"],
                "tts_ttfa_ms": metrics["tts_ttfa_ms"],
                "first_audible_ms": metrics.get("first_audible_ms", metrics["tts_ttfa_ms"]),
                "tts_total_ms": metrics["tts_total_ms"],
                "e2e_latency_ms": metrics["e2e_latency_ms"],
            })

            persona_rows.append({
                "prompt_id": prompt_id,
                "category": category,
                "event": event,
                "player_prompt": user_text,
                "companion_reply": metrics["normalized_tts_text"],
                "profanity_word": metrics.get("profanity_word", ""),
                "profanity_severity": metrics.get("profanity_severity", "none"),
                "persona_score": persona_eval["persona_score"],
                "ooc_passed": persona_eval["ooc_passed"],
                "script_clean": persona_eval["script_clean"],
                "orator_markers": persona_eval["has_orator_markers"],
                "audio_file": str(out_audio.name),
            })

        # Calculate Summary Metrics
        mean_e2e = float(np.mean(total_e2e_latencies))
        p50_e2e = float(np.percentile(total_e2e_latencies, 50))
        p90_e2e = float(np.percentile(total_e2e_latencies, 90))
        p95_e2e = float(np.percentile(total_e2e_latencies, 95))

        mean_llm_ttft = float(np.mean(total_llm_ttft))
        mean_tts_ttfa = float(np.mean(total_tts_ttfa))
        mean_first_audible = float(np.mean(total_first_audible))

        mean_persona_score = float(np.mean(total_persona_scores))
        persona_consistency_pct = (mean_persona_score / 5.0) * 100.0
        ooc_defense_rate_pct = (
            (ooc_passed_tests / ooc_total_tests * 100.0) if ooc_total_tests > 0 else 100.0
        )

        is_offline_test = (fallback_count == len(test_items))

        summary = {
            "persona": "सरकार भैया (Sarkar Bhaiya)",
            "total_prompts_evaluated": len(test_items),
            "offline_fallback_active": is_offline_test,
            "fallback_responses_count": fallback_count,
            "persona_consistency_pct": round(persona_consistency_pct, 1),
            "average_persona_score_out_of_5": round(mean_persona_score, 2),
            "ooc_defense_rate_pct": round(ooc_defense_rate_pct, 1),
            "ooc_traps_tested": ooc_total_tests,
            "ooc_traps_defended": ooc_passed_tests,
            "latency": {
                "e2e_mean_ms": round(mean_e2e, 1),
                "e2e_p50_ms": round(p50_e2e, 1),
                "e2e_p90_ms": round(p90_e2e, 1),
                "e2e_p95_ms": round(p95_e2e, 1),
                "llm_ttft_mean_ms": round(mean_llm_ttft, 1),
                "tts_ttfa_mean_ms": round(mean_tts_ttfa, 1),
                "first_audible_mean_ms": round(mean_first_audible, 1),
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        self._save_csv(self.results_dir / "latency_breakdown.csv", latency_rows)
        self._save_csv(self.results_dir / "persona_evaluation.csv", persona_rows)

        with open(self.results_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.print_resume_summary(summary)
        return summary

    def _save_csv(self, file_path: Path, rows: List[Dict[str, Any]]) -> None:
        """Writes rows to CSV."""
        if not rows:
            return
        keys = list(rows[0].keys())
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Saved {file_path.name}")

    def print_resume_summary(self, summary: Dict[str, Any]) -> None:
        """Prints copy-pasteable bullet points with concrete metrics."""
        lat = summary["latency"]
        print("\n" + "=" * 70)
        print("     👑 SARKAR BHAIYA (सरकार भैया) EVALUATION BENCHMARK SUMMARY     ")
        print("=" * 70)
        print(f"• Persona Consistency: {summary['persona_consistency_pct']}% across {summary['total_prompts_evaluated']} Skyrim event & lore prompts")
        print(f"• OOC Trap Defense Rate: {summary['ooc_defense_rate_pct']}% ({summary['ooc_traps_defended']}/{summary['ooc_traps_tested']} prompt injections deflected)")
        print(f"• Average End-to-End Latency: {lat['e2e_mean_ms']} ms (P90: {lat['e2e_p90_ms']} ms, P95: {lat['e2e_p95_ms']} ms)")
        print(f"• Streaming TTFT: {lat['llm_ttft_mean_ms']} ms | First Audible Audio: {lat['first_audible_mean_ms']} ms")
        print(f"• Mode: {'Headless Offline Fallback' if summary['offline_fallback_active'] else 'Live LM Studio GPU Inference'}")
        print("=" * 70)
        print(f"Results exported to: {self.results_dir.resolve()}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sarkar Bhaiya Evaluation Harness")
    parser.add_argument("--max_prompts", type=int, default=None, help="Limit number of prompts")
    parser.add_argument("--prompts", type=str, default=None, help="Custom test prompts JSON path")
    args = parser.parse_args()

    prompts_path = Path(args.prompts) if args.prompts else None
    harness = BenchmarkHarness(prompts_path=prompts_path)
    harness.run(max_prompts=args.max_prompts)
