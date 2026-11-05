"""
Automated Evaluation Harness for Skyrim Hindi Companion.

Benchmarks:
  1. Per-stage & End-to-End Latency (ASR, Normalization, LLM TTFT, TTS TTFA, Total E2E)
  2. Persona Consistency & Skyrim Lore Grounding (Rubric-based evaluation)
  3. Out-Of-Character (OOC) Trap Defense Rate
  4. Synthesis Real-Time Factor (RTF) and Audio Generation for Blind MOS Testing

Exports:
  - eval/results/latency_breakdown.csv
  - eval/results/persona_evaluation.csv
  - eval/results/benchmark_summary.json
  - Formatted resume bullets ready for copy-pasting
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_benchmark")


class BenchmarkHarness:
    """Orchestrates comprehensive quantitative evaluation across all prompt sets."""

    def __init__(
        self,
        prompts_path: Optional[Path] = None,
        results_dir: Optional[Path] = None,
    ):
        self.prompts_path = prompts_path or repo_root / "eval" / "test_prompts.json"
        self.results_dir = results_dir or repo_root / "eval" / "results"
        self.audio_samples_dir = self.results_dir / "audio_samples"

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.audio_samples_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline = SkyrimHindiPipeline()
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
        Evaluates companion response against persona rubric:
        - In-Character Score (1-5)
        - OOC Resistance (Pass/Fail)
        - Lore Grounding (1-5)
        - Hindi Fluency & Script Check (Devanagari vs Latin remnants)
        """
        category = prompt_item.get("category", "general")
        rubric = prompt_item.get("evaluation_rubric", {})
        must_include = rubric.get("must_include", [])
        forbidden_terms = rubric.get("forbidden_terms", [])

        reply_lower = companion_reply.lower()

        # Check forbidden terms (OOC leaks)
        leaked_terms = [t for t in forbidden_terms if t.lower() in reply_lower]
        ooc_passed = (len(leaked_terms) == 0)

        # Check in-world lore keywords
        matched_lore = [term for term in must_include if term in companion_reply]
        lore_ratio = (len(matched_lore) / len(must_include)) if must_include else 1.0

        # Script check: penalize if excessive Latin characters leak into Hindi speech
        has_latin = self.pipeline.normalizer.contains_latin(companion_reply)

        # Persona scoring logic (1 to 5 scale)
        score = 5.0

        if not ooc_passed:
            score -= 3.0  # Major penalty for breaking character / AI admission
        if has_latin:
            score -= 0.5  # Minor deduction for un-transliterated English words
        if lore_ratio < 0.5 and category in ("lore", "combat"):
            score -= 1.0  # Deduction for ignoring key tactical/lore context

        # Ensure bounds [1.0, 5.0]
        final_score = max(1.0, min(5.0, round(score, 1)))

        return {
            "persona_score": final_score,
            "ooc_passed": ooc_passed,
            "leaked_forbidden_terms": leaked_terms,
            "matched_lore_terms": matched_lore,
            "lore_coverage": round(lore_ratio * 100.0, 1),
            "script_clean": not has_latin,
        }

    def run(self, max_prompts: Optional[int] = None) -> Dict[str, Any]:
        """Executes automated benchmark across all test cases."""
        test_items = self.prompts[:max_prompts] if max_prompts else self.prompts
        logger.info(f"Starting evaluation on {len(test_items)} test prompts...")

        latency_rows: List[Dict[str, Any]] = []
        persona_rows: List[Dict[str, Any]] = []

        total_e2e_latencies = []
        total_llm_ttft = []
        total_tts_ttfa = []
        total_tts_rtf = []
        total_persona_scores = []
        ooc_total_tests = 0
        ooc_passed_tests = 0

        for idx, item in enumerate(test_items, 1):
            prompt_id = item["id"]
            category = item["category"]
            user_text = item["prompt_text"]

            logger.info(f"[{idx}/{len(test_items)}] Testing [{category.upper()}]: {user_text}")

            # Run through end-to-end companion pipeline
            out_audio = self.audio_samples_dir / f"{prompt_id}.wav"
            t_start = time.perf_counter()
            metrics = self.pipeline.process_turn(
                user_input_text=user_text,
                save_audio=True,
                play_audio=False,
            )
            # Copy or save audio
            if metrics.get("audio_path") and Path(metrics["audio_path"]).exists():
                Path(metrics["audio_path"]).replace(out_audio)
                metrics["audio_path"] = str(out_audio)

            # Evaluate persona adherence
            persona_eval = self.evaluate_persona(item, metrics["normalized_tts_text"])

            if category == "ooc_trap":
                ooc_total_tests += 1
                if persona_eval["ooc_passed"]:
                    ooc_passed_tests += 1

            total_e2e_latencies.append(metrics["e2e_latency_ms"])
            total_llm_ttft.append(metrics["llm_ttft_ms"])
            total_tts_ttfa.append(metrics["tts_ttfa_ms"])
            total_tts_rtf.append(metrics["tts_rtf"])
            total_persona_scores.append(persona_eval["persona_score"])

            # Store per-turn records
            latency_rows.append({
                "prompt_id": prompt_id,
                "category": category,
                "stt_latency_ms": metrics["stt_latency_ms"],
                "norm_latency_ms": metrics["norm_latency_ms"],
                "llm_ttft_ms": metrics["llm_ttft_ms"],
                "llm_total_ms": metrics["llm_total_ms"],
                "tts_ttfa_ms": metrics["tts_ttfa_ms"],
                "tts_total_ms": metrics["tts_total_ms"],
                "tts_duration_sec": metrics["tts_audio_duration_sec"],
                "tts_rtf": metrics["tts_rtf"],
                "e2e_latency_ms": metrics["e2e_latency_ms"],
            })

            persona_rows.append({
                "prompt_id": prompt_id,
                "category": category,
                "player_prompt": user_text,
                "companion_reply": metrics["normalized_tts_text"],
                "persona_score": persona_eval["persona_score"],
                "ooc_passed": persona_eval["ooc_passed"],
                "lore_coverage_pct": persona_eval["lore_coverage"],
                "leaked_terms": ";".join(persona_eval["leaked_forbidden_terms"]),
                "audio_file": str(out_audio.name),
            })

        # Calculate Summary Metrics
        mean_e2e = float(np.mean(total_e2e_latencies))
        p50_e2e = float(np.percentile(total_e2e_latencies, 50))
        p90_e2e = float(np.percentile(total_e2e_latencies, 90))
        p95_e2e = float(np.percentile(total_e2e_latencies, 95))

        mean_llm_ttft = float(np.mean(total_llm_ttft))
        mean_tts_ttfa = float(np.mean(total_tts_ttfa))
        mean_rtf = float(np.mean(total_tts_rtf))

        mean_persona_score = float(np.mean(total_persona_scores))
        persona_consistency_pct = (mean_persona_score / 5.0) * 100.0
        ooc_defense_rate_pct = (
            (ooc_passed_tests / ooc_total_tests * 100.0) if ooc_total_tests > 0 else 100.0
        )

        summary = {
            "total_prompts_evaluated": len(test_items),
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
                "tts_rtf_mean": round(mean_rtf, 3),
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save CSV outputs
        self._save_csv(self.results_dir / "latency_breakdown.csv", latency_rows)
        self._save_csv(self.results_dir / "persona_evaluation.csv", persona_rows)

        # Save summary JSON
        with open(self.results_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Print formatted resume block
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
        print("          🎯 RESUME METRICS SUMMARY (TIER 1 BASELINE)           ")
        print("=" * 70)
        print(f"• Character Consistency: {summary['persona_consistency_pct']}% on {summary['total_prompts_evaluated']}-prompt Skyrim eval harness")
        print(f"• OOC Trap Defense Rate: {summary['ooc_defense_rate_pct']}% ({summary['ooc_traps_defended']}/{summary['ooc_traps_tested']} injections deflected in-character)")
        print(f"• Average End-to-End Latency: {lat['e2e_mean_ms']} ms (P90: {lat['e2e_p90_ms']} ms, P95: {lat['e2e_p95_ms']} ms)")
        print(f"• Sub-system Latencies: LLM TTFT: {lat['llm_ttft_mean_ms']} ms | TTS TTFA: {lat['tts_ttfa_mean_ms']} ms")
        print(f"• Voice Synthesis Throughput: {lat['tts_rtf_mean']} Real-Time Factor (RTF)")
        print("=" * 70)
        print(f"Results exported to: {self.results_dir.resolve()}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Skyrim Hindi Companion Evaluation Harness")
    parser.add_argument("--max_prompts", type=int, default=None, help="Limit number of prompts")
    parser.add_argument("--prompts", type=str, default=None, help="Custom test prompts JSON path")
    args = parser.parse_args()

    prompts_path = Path(args.prompts) if args.prompts else None
    harness = BenchmarkHarness(prompts_path=prompts_path)
    harness.run(max_prompts=args.max_prompts)

