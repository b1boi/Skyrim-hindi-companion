"""
End-to-End Voice Interactive Hindi Companion Pipeline for Skyrim.

Orchestrates:
  Microphone / File / Text Input
    ↓
  Faster-Whisper (ASR)
    ↓
  Skyrim Devanagari Normalizer
    ↓
  Local LLM (Persona Prompt + Streaming TTFT)
    ↓
  Coqui XTTS-v2 (Zero-shot Hindi Synthesis + Streaming Chunker)
    ↓
  Audio Playback & Latency Profiler

Ready to connect directly with the Mantella Skyrim mod pipeline in Tier 3.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple, Union

import yaml

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.asr import FasterWhisperASR
from src.llm import SkyrimCompanionLLM
from src.transliterate import SkyrimHindiNormalizer
from src.tts import XTTSHindiSynthesizer

logger = logging.getLogger(__name__)

# Check for rich library
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    RICH_AVAILABLE = True
    console = Console()
except (ImportError, Exception):
    RICH_AVAILABLE = False
    console = None


class SkyrimHindiPipeline:
    """End-to-end ASR -> LLM -> TTS pipeline with real-time latency instrumentation."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        provider: Optional[str] = None,
        sarvam_speaker: Optional[str] = None,
    ):
        self.config_path = (
            Path(config_path) if config_path else repo_root / "config.yaml"
        )
        self.config = self._load_config()

        # Resolve provider: "local" (Faster-Whisper / XTTS) or "sarvam" (Sarvam Cloud Indic AI)
        self.provider = (
            provider
            or self.config.get("general", {}).get("provider", "local")
        ).lower()

        sarvam_cfg = self.config.get("sarvam", {})
        raw_key = sarvam_cfg.get("api_key", "")
        if raw_key and raw_key.startswith("${") and raw_key.endswith("}"):
            self.sarvam_key = os.environ.get(raw_key[2:-1])
        else:
            self.sarvam_key = raw_key or os.environ.get("SARVAM_API_KEY")

        self.sarvam_speaker = sarvam_speaker or sarvam_cfg.get("tts", {}).get("speaker", "arvind")

        logger.info(
            f"Initializing Skyrim Hindi AI Companion Pipeline (Provider: {self.provider.upper()})..."
        )

        # 1. Initialize Normalizer
        lore_dict = self.config.get("transliteration", {}).get(
            "lore_dict_path", "data/lore_transliteration.json"
        )
        self.normalizer = SkyrimHindiNormalizer(
            lore_dict_path=repo_root / lore_dict if not Path(lore_dict).is_absolute() else Path(lore_dict)
        )

        # 2. Initialize ASR
        asr_cfg = self.config.get("asr", {})
        self.asr = FasterWhisperASR(
            model_id=asr_cfg.get("model_id", "medium"),
            device=asr_cfg.get("device", "auto"),
            compute_type=asr_cfg.get("compute_type", "default"),
            language=asr_cfg.get("language", "hi"),
            vad_filter=asr_cfg.get("vad_filter", True),
            beam_size=asr_cfg.get("beam_size", 5),
            provider="sarvam" if self.provider == "sarvam" else "faster_whisper",
            sarvam_api_key=self.sarvam_key,
            sarvam_model=sarvam_cfg.get("stt", {}).get("model", "saaras:v4"),
        )

        # 3. Initialize LLM & UESP Knowledge Base
        llm_cfg = self.config.get("llm", {})
        kb_cfg = self.config.get("knowledge_base", {})
        kb_path = kb_cfg.get("cache_path", "data/skyrim_lore_kb.json")

        self.llm = SkyrimCompanionLLM(
            api_base=llm_cfg.get("api_base", "http://127.0.0.1:1234/v1"),
            api_key=llm_cfg.get("api_key", "none"),
            model_name=llm_cfg.get("model_name", "qwen2.5:7b"),
            temperature=llm_cfg.get("temperature", 0.7),
            top_p=llm_cfg.get("top_p", 0.9),
            max_tokens=llm_cfg.get("max_tokens", 512),
            system_prompt=llm_cfg.get("system_prompt", None),
            enable_kb=kb_cfg.get("enabled", True),
            kb_cache_path=repo_root / kb_path if not Path(kb_path).is_absolute() else Path(kb_path),
        )

        # 4. Initialize TTS
        tts_cfg = self.config.get("tts", {})
        ref_path = tts_cfg.get("reference_audio_path", "data/reference_speaker.wav")
        self.tts = XTTSHindiSynthesizer(
            model_id=tts_cfg.get("model_id", "tts_models/multilingual/multi-dataset/xtts_v2"),
            reference_audio_path=repo_root / ref_path if not Path(ref_path).is_absolute() else Path(ref_path),
            device=tts_cfg.get("device", "auto"),
            language=tts_cfg.get("language", "hi"),
            sample_rate=tts_cfg.get("sample_rate", 24000),
            temperature=tts_cfg.get("temperature", 0.65),
            speed=tts_cfg.get("speed", 1.02),
            provider="sarvam" if self.provider == "sarvam" else "xtts",
            sarvam_api_key=self.sarvam_key,
            sarvam_speaker=self.sarvam_speaker,
            sarvam_model=sarvam_cfg.get("tts", {}).get("model", "bulbul:v4-flash"),
        )

        self.audio_output_dir = repo_root / self.config.get("tts", {}).get(
            "audio_output_dir", "eval/results/audio_samples"
        )
        self.audio_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Pipeline ready for execution.")

    def _load_config(self) -> Dict[str, Any]:
        """Loads YAML configuration."""
        if not self.config_path.exists():
            logger.warning(f"Config {self.config_path} not found. Using defaults.")
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def process_turn(
        self,
        user_input_text: Optional[str] = None,
        audio_file_path: Optional[Union[str, Path]] = None,
        raw_audio_array: Optional[Any] = None,
        play_audio: bool = False,
        save_audio: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes a complete single turn:
        1. ASR (if audio input provided)
        2. Normalization of player query
        3. LLM generation with persona adherence
        4. Transliteration / normalization of companion text
        5. XTTS-v2 synthesis
        6. Playback and latency telemetry calculation
        """
        metrics: Dict[str, Any] = {}
        t_turn_start = time.perf_counter()

        # Step 1: Input handling & Speech Recognition
        player_text = ""
        stt_latency_ms = 0.0

        if raw_audio_array is not None:
            asr_res = self.asr.transcribe_audio_array(raw_audio_array)
            player_text = asr_res["text"]
            stt_latency_ms = asr_res["latency_ms"]
        elif audio_file_path is not None:
            asr_res = self.asr.transcribe_file(audio_file_path)
            player_text = asr_res["text"]
            stt_latency_ms = asr_res["latency_ms"]
        elif user_input_text is not None:
            player_text = user_input_text.strip()
            stt_latency_ms = 0.0
        else:
            raise ValueError("No input provided. Supply user_input_text, audio_file_path, or raw_audio_array.")

        metrics["player_input"] = player_text
        metrics["stt_latency_ms"] = round(stt_latency_ms, 2)

        # Step 2: Query Normalization
        t_norm_start = time.perf_counter()
        normalized_query = self.normalizer.normalize(player_text)
        t_norm_end = time.perf_counter()
        norm_query_ms = (t_norm_end - t_norm_start) * 1000.0

        # Step 3: LLM Companion Persona Response
        t_llm_start = time.perf_counter()
        llm_res = self.llm.generate(normalized_query)
        raw_companion_reply = llm_res["text"]
        llm_ttft_ms = llm_res["ttft_ms"]
        llm_total_ms = llm_res["total_ms"]
        llm_tokens = llm_res["tokens"]
        llm_tps = llm_res["tokens_per_sec"]

        metrics["raw_companion_reply"] = raw_companion_reply
        metrics["llm_ttft_ms"] = round(llm_ttft_ms, 2)
        metrics["llm_total_ms"] = round(llm_total_ms, 2)
        metrics["llm_tokens"] = llm_tokens
        metrics["llm_tps"] = round(llm_tps, 2)

        # Step 4: Companion Response Normalization for TTS
        t_norm_reply_start = time.perf_counter()
        clean_tts_text = self.normalizer.normalize(raw_companion_reply)
        norm_reply_ms = (time.perf_counter() - t_norm_reply_start) * 1000.0
        metrics["normalized_tts_text"] = clean_tts_text
        metrics["norm_latency_ms"] = round(norm_query_ms + norm_reply_ms, 2)

        # Step 5: TTS Speech Synthesis
        timestamp_str = int(time.time() * 1000)
        out_wav_path = self.audio_output_dir / f"turn_{timestamp_str}.wav" if save_audio else None

        tts_res = self.tts.synthesize(clean_tts_text, output_path=out_wav_path)
        metrics["tts_ttfa_ms"] = tts_res["ttfa_ms"]
        metrics["tts_total_ms"] = tts_res["latency_ms"]
        metrics["tts_audio_duration_sec"] = tts_res["duration_sec"]
        metrics["tts_rtf"] = tts_res["rtf"]
        metrics["audio_path"] = str(out_wav_path) if out_wav_path else None

        # Step 6: Audio Playback (optional)
        if play_audio:
            self.tts.play_audio(tts_res["audio"], sample_rate=tts_res["sample_rate"])

        # End-to-End Latency Calculation
        t_turn_end = time.perf_counter()
        e2e_latency_ms = (t_turn_end - t_turn_start) * 1000.0
        metrics["e2e_latency_ms"] = round(e2e_latency_ms, 2)

        # Mantella event hook dispatch
        self.dispatch_mantella_hook(
            player_text=player_text,
            companion_text=clean_tts_text,
            audio_path=str(out_wav_path) if out_wav_path else "",
        )

        return metrics

    def dispatch_mantella_hook(self, player_text: str, companion_text: str, audio_path: str) -> None:
        """
        Extensible hook for Tier 3 Skyrim Mantella integration.
        Mantella reads companion audio files from disk and triggers Skyrim Papyrus NPC LipSync events.
        """
        logger.debug(
            f"[Mantella Hook] Player: '{player_text}' -> Companion: '{companion_text}' (Audio: {audio_path})"
        )

    def print_turn_summary(self, metrics: Dict[str, Any]) -> None:
        """Renders rich terminal tables with telemetry."""
        if RICH_AVAILABLE and console:
            console.print("\n")
            console.print(Panel(f"[bold cyan]{metrics['player_input']}[/bold cyan]", title="👤 Player Input"))
            console.print(Panel(f"[bold green]{metrics['normalized_tts_text']}[/bold green]", title="⚔️ Veer (Companion)"))

            table = Table(title="⚡ Latency Telemetry Breakdown", show_header=True, header_style="bold magenta")
            table.add_column("Pipeline Stage", style="dim")
            table.add_column("Metric", justify="right")
            table.add_column("Target (SLA)", justify="right")
            table.add_column("Status", justify="center")

            def sla_badge(val, threshold):
                return "[bold green]PASS[/bold green]" if val <= threshold else "[bold yellow]WARN[/bold yellow]"

            table.add_row("ASR (Faster-Whisper)", f"{metrics['stt_latency_ms']} ms", "≤ 800 ms", sla_badge(metrics['stt_latency_ms'], 800))
            table.add_row("Normalization", f"{metrics['norm_latency_ms']} ms", "≤ 50 ms", sla_badge(metrics['norm_latency_ms'], 50))
            table.add_row("LLM TTFT", f"{metrics['llm_ttft_ms']} ms", "≤ 600 ms", sla_badge(metrics['llm_ttft_ms'], 600))
            table.add_row("LLM Total", f"{metrics['llm_total_ms']} ms ({metrics['llm_tps']} tps)", "≤ 1500 ms", sla_badge(metrics['llm_total_ms'], 1500))
            table.add_row("TTS TTFA", f"{metrics['tts_ttfa_ms']} ms", "≤ 900 ms", sla_badge(metrics['tts_ttfa_ms'], 900))
            table.add_row("TTS Total", f"{metrics['tts_total_ms']} ms (RTF: {metrics['tts_rtf']})", "≤ 1500 ms", sla_badge(metrics['tts_total_ms'], 1500))
            table.add_row("[bold]Total End-to-End[/bold]", f"[bold]{metrics['e2e_latency_ms']} ms[/bold]", "≤ 2800 ms", sla_badge(metrics['e2e_latency_ms'], 2800))

            console.print(table)
        else:
            print("\n" + "=" * 60)
            print(f"👤 Player:    {metrics['player_input']}")
            print(f"⚔️ Companion: {metrics['normalized_tts_text']}")
            print("-" * 60)
            print(f"STT Latency:   {metrics['stt_latency_ms']} ms")
            print(f"Norm Latency:  {metrics['norm_latency_ms']} ms")
            print(f"LLM TTFT:      {metrics['llm_ttft_ms']} ms ({metrics['llm_tps']} tps)")
            print(f"TTS TTFA:      {metrics['tts_ttfa_ms']} ms (RTF: {metrics['tts_rtf']})")
            print(f"End-to-End:    {metrics['e2e_latency_ms']} ms")
            print("=" * 60)


def interactive_cli() -> None:
    """Launches the interactive Push-to-Talk and Text console."""
    parser = argparse.ArgumentParser(description="Skyrim Hindi AI Companion Voice Pipeline")
    parser.add_argument("--mode", choices=["text", "mic", "file"], default="text", help="Interaction mode")
    parser.add_argument("--audio", type=str, default=None, help="Input audio file for file mode")
    parser.add_argument("--play", action="store_true", help="Play synthesized audio through speakers")
    parser.add_argument("--sarvam", action="store_true", help="Use Sarvam AI Indic models (Bulbul TTS & Saaras STT)")
    parser.add_argument("--speaker", type=str, default="arvind", help="Sarvam TTS speaker (arvind, meera, ishita_enhi_companion)")
    args = parser.parse_args()

    pipeline = SkyrimHindiPipeline(
        provider="sarvam" if args.sarvam else None,
        sarvam_speaker=args.speaker,
    )

    if args.mode == "file":
        if not args.audio:
            print("Error: --audio <path> is required in file mode.")
            sys.exit(1)
        res = pipeline.process_turn(audio_file_path=args.audio, play_audio=args.play)
        pipeline.print_turn_summary(res)
        return

    print("\n⚔️ ======================================================= ⚔️")
    print("   Skyrim Hindi AI Companion — 'Veer' (Tier 1 Baseline)    ")
    print("⚔️ ======================================================= ⚔️")
    print(f"Mode: {args.mode.upper()} | Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            if args.mode == "mic":
                audio_arr = FasterWhisperASR.record_microphone_push_to_talk()
                if len(audio_arr) == 0:
                    continue
                res = pipeline.process_turn(raw_audio_array=audio_arr, play_audio=args.play)
            else:
                user_msg = input("\n[Player] Type prompt (Hindi or Hinglish): ")
                if user_msg.strip().lower() in ("exit", "quit", "q"):
                    print("Leaving Skyrim. Safe travels, Dovahkiin!")
                    break
                if not user_msg.strip():
                    continue
                res = pipeline.process_turn(user_input_text=user_msg, play_audio=args.play)

            pipeline.print_turn_summary(res)

        except KeyboardInterrupt:
            print("\nExiting companion pipeline. May the Gods watch over your battles!")
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    interactive_cli()

