"""
End-to-End Voice Interactive Hindi Companion Pipeline for Skyrim — 'सरकार भैया'.

Orchestrates:
  Microphone / File / Text Input
    ↓
  Faster-Whisper (ASR)
    ↓
  Skyrim Devanagari Normalizer
    ↓
  Local LLM (Sarkar Bhaiya Persona + Gemma 4 E2B Instruct + Streaming TTFT)
    ↓
  Coqui XTTS-v2 / Sarvam Bulbul (Sentence-level Streaming Chunker)
    ↓
  Thread-Safe Sequential Audio Playback Queue (Synthesize N while N-1 plays)
    ↓
  Mantella Event Dispatch & Latency Profiler

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
from src.persona import GAMEPLAY_EVENTS
from src.transliterate import SkyrimHindiNormalizer
from src.tts import XTTSHindiSynthesizer
from src.tts_streamer import StreamingTTSPipeline

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
        mock_playback: bool = False,
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
            f"Initializing Skyrim Hindi AI Companion Pipeline — सरकार भैया (Provider: {self.provider.upper()})..."
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
        dump_db = kb_cfg.get("dump_db_path", "data/uesp_knowledge_base.db")
        dump_file = kb_cfg.get("dump_path", "uespwiki-2026-10-25-current.xml.bz2")

        persona_cfg = self.config.get("persona", {})

        self.llm = SkyrimCompanionLLM(
            api_base=llm_cfg.get("api_base", "http://127.0.0.1:1234/v1"),
            api_key=llm_cfg.get("api_key", "none"),
            model_name=llm_cfg.get("model_name", "google/gemma-4-e2b-instruct"),
            temperature=llm_cfg.get("temperature", 0.7),
            top_p=llm_cfg.get("top_p", 0.9),
            max_tokens=llm_cfg.get("max_tokens", 96),
            context_window=llm_cfg.get("context_window", 8192),
            system_prompt=llm_cfg.get("system_prompt", None),
            enable_kb=kb_cfg.get("enabled", True),
            kb_cache_path=repo_root / kb_path if not Path(kb_path).is_absolute() else Path(kb_path),
            dump_db_path=repo_root / dump_db if not Path(dump_db).is_absolute() else Path(dump_db),
            dump_file_path=repo_root / dump_file if not Path(dump_file).is_absolute() else Path(dump_file),
            profanity_prob=persona_cfg.get("profanity_chance", 0.33),
            meltdown_prob=persona_cfg.get("meltdown_chance", 0.02),
            rng_seed=self.config.get("general", {}).get("seed", 42),
        )

        # 4. Initialize TTS & Streaming Audio Coordinator
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

        self.stream_pipeline = StreamingTTSPipeline(
            tts_synthesizer=self.tts,
            mock_playback=mock_playback,
        )

        self.audio_output_dir = repo_root / self.config.get("tts", {}).get(
            "audio_output_dir", "eval/results/audio_samples"
        )
        self.audio_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Pipeline ready for execution.")

    def _load_config(self) -> Dict[str, Any]:
        """Loads YAML configuration, merging config.local.yaml overrides if present."""
        base_cfg = {}
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                base_cfg = yaml.safe_load(f) or {}

        local_cfg_path = repo_root / "config.local.yaml"
        if local_cfg_path.exists():
            try:
                with open(local_cfg_path, "r", encoding="utf-8") as f:
                    local_cfg = yaml.safe_load(f) or {}
                for k, v in local_cfg.items():
                    if isinstance(v, dict) and isinstance(base_cfg.get(k), dict):
                        base_cfg[k].update(v)
                    else:
                        base_cfg[k] = v
            except Exception as e:
                logger.warning(f"Error reading config.local.yaml: {e}")

        return base_cfg

    def process_turn(
        self,
        user_input_text: Optional[str] = None,
        audio_file_path: Optional[Union[str, Path]] = None,
        raw_audio_array: Optional[Any] = None,
        event: str = "generic_conversation",
        play_audio: bool = False,
        save_audio: bool = True,
        streaming: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes a complete single interactive turn:
        1. ASR (transcription if audio input provided)
        2. Normalization of player query
        3. LLM generation with Sarkar Bhaiya persona & event directives
        4. Sentence-level streaming TTS synthesis & sequential playback queue
        5. Mantella Papyrus hook dispatch
        6. Latency telemetry calculation
        """
        metrics: Dict[str, Any] = {}
        t_turn_start = time.perf_counter()

        # Step 1: Input handling & Speech Recognition
        player_text = ""
        stt_latency_ms = 0.0
        stt_bypassed = False

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
            stt_bypassed = True
        else:
            raise ValueError("No input provided. Supply user_input_text, audio_file_path, or raw_audio_array.")

        metrics["player_input"] = player_text
        metrics["stt_latency_ms"] = round(stt_latency_ms, 2)
        metrics["stt_bypassed"] = stt_bypassed
        metrics["event"] = event

        # Step 2: Query Normalization
        t_norm_start = time.perf_counter()
        normalized_query = self.normalizer.normalize(player_text)
        norm_query_ms = (time.perf_counter() - t_norm_start) * 1000.0

        timestamp_str = int(time.time() * 1000)
        out_wav_path = self.audio_output_dir / f"turn_{timestamp_str}.wav" if save_audio else None

        if streaming:
            # Step 3 & 4: Streaming LLM into Thread-Safe Audio Playback Queue
            token_stream = self.llm.stream_generate(normalized_query, event=event)
            stream_res = self.stream_pipeline.run_streaming_turn(
                token_stream=token_stream,
                play_audio=play_audio,
                output_wav_path=out_wav_path,
                turn_start_time=t_turn_start,
            )

            raw_reply = stream_res["full_text"]
            llm_m = stream_res["llm_metrics"]

            # Companion normalization
            t_norm_reply_start = time.perf_counter()
            clean_tts_text = self.normalizer.normalize(raw_reply)
            norm_reply_ms = (time.perf_counter() - t_norm_reply_start) * 1000.0

            metrics["raw_companion_reply"] = raw_reply
            metrics["normalized_tts_text"] = clean_tts_text
            metrics["norm_latency_ms"] = round(norm_query_ms + norm_reply_ms, 2)
            metrics["llm_ttft_ms"] = llm_m.get("ttft_ms", 0.0)
            metrics["llm_total_ms"] = llm_m.get("total_ms", 0.0)
            metrics["llm_tokens"] = llm_m.get("tokens", len(clean_tts_text.split()))
            metrics["llm_tps"] = llm_m.get("tokens_per_sec", 0.0)
            metrics["is_llm_fallback"] = llm_m.get("is_fallback", False)
            metrics["profanity_word"] = llm_m.get("profanity_word", None)
            metrics["profanity_severity"] = llm_m.get("profanity_severity", "none")

            metrics["tts_ttfa_ms"] = stream_res["ttfa_ms"]
            metrics["first_audible_ms"] = stream_res["first_audible_ms"]
            metrics["tts_total_ms"] = stream_res["tts_total_synth_ms"]
            metrics["tts_audio_duration_sec"] = stream_res["audio_duration_sec"]
            metrics["tts_rtf"] = (
                round(stream_res["tts_total_synth_ms"] / (stream_res["audio_duration_sec"] * 1000.0), 3)
                if stream_res["audio_duration_sec"] > 0
                else 0.0
            )
            metrics["is_tts_fallback"] = self.tts.tts is None and self.provider != "sarvam"
            metrics["audio_path"] = str(out_wav_path) if out_wav_path else None
            metrics["e2e_latency_ms"] = stream_res["e2e_latency_ms"]

        else:
            # Non-streaming execution path
            llm_res = self.llm.generate(normalized_query, event=event)
            raw_reply = llm_res["text"]

            t_norm_reply_start = time.perf_counter()
            clean_tts_text = self.normalizer.normalize(raw_reply)
            norm_reply_ms = (time.perf_counter() - t_norm_reply_start) * 1000.0

            tts_res = self.tts.synthesize(clean_tts_text, output_path=out_wav_path)
            if play_audio:
                self.tts.play_audio(tts_res["audio"], sample_rate=tts_res["sample_rate"])

            t_end = time.perf_counter()
            e2e_latency_ms = (t_end - t_turn_start) * 1000.0

            metrics["raw_companion_reply"] = raw_reply
            metrics["normalized_tts_text"] = clean_tts_text
            metrics["norm_latency_ms"] = round(norm_query_ms + norm_reply_ms, 2)
            metrics["llm_ttft_ms"] = llm_res["ttft_ms"]
            metrics["llm_total_ms"] = llm_res["total_ms"]
            metrics["llm_tokens"] = llm_res["tokens"]
            metrics["llm_tps"] = llm_res["tokens_per_sec"]
            metrics["is_llm_fallback"] = llm_res["is_fallback"]
            metrics["profanity_word"] = llm_res.get("profanity_word", None)
            metrics["profanity_severity"] = llm_res.get("profanity_severity", "none")

            metrics["tts_ttfa_ms"] = tts_res["ttfa_ms"]
            metrics["first_audible_ms"] = tts_res["ttfa_ms"]
            metrics["tts_total_ms"] = tts_res["latency_ms"]
            metrics["tts_audio_duration_sec"] = tts_res["duration_sec"]
            metrics["tts_rtf"] = tts_res["rtf"]
            metrics["is_tts_fallback"] = tts_res["is_fallback"]
            metrics["audio_path"] = str(out_wav_path) if out_wav_path else None
            metrics["e2e_latency_ms"] = round(e2e_latency_ms, 2)

        # Dispatch Mantella integration hook
        self.dispatch_mantella_hook(
            player_text=player_text,
            companion_text=clean_tts_text,
            audio_path=str(out_wav_path) if out_wav_path else "",
            event=event,
        )

        return metrics

    def dispatch_mantella_hook(
        self, player_text: str, companion_text: str, audio_path: str, event: str = "generic_conversation"
    ) -> None:
        """
        Extensible hook for Tier 3 Skyrim Mantella integration.
        Mantella reads companion audio files from disk and triggers Skyrim Papyrus NPC LipSync events.
        """
        logger.debug(
            f"[Mantella Hook] Event: '{event}' | Player: '{player_text}' -> Sarkar: '{companion_text}' (Audio: {audio_path})"
        )

    def print_turn_summary(self, metrics: Dict[str, Any]) -> None:
        """Renders rich terminal tables with telemetry."""
        evt = metrics.get("event", "generic_conversation")
        prof = metrics.get("profanity_word")
        prof_info = f" | Insult: '{prof}' ({metrics.get('profanity_severity')})" if prof else ""

        if RICH_AVAILABLE and console:
            console.print("\n")
            console.print(Panel(f"[bold cyan]{metrics['player_input']}[/bold cyan]", title=f"👤 Player Input (Event: {evt})"))
            console.print(
                Panel(
                    f"[bold yellow]{metrics['normalized_tts_text']}[/bold yellow]",
                    title=f"👑 सरकार भैया — Sarkar Bhaiya{prof_info}",
                )
            )

            table = Table(title="⚡ Latency Telemetry Breakdown (Sarkar Bhaiya Voice Pipeline)", show_header=True, header_style="bold magenta")
            table.add_column("Pipeline Stage", style="dim")
            table.add_column("Metric", justify="right")
            table.add_column("Target (SLA)", justify="right")
            table.add_column("Status", justify="center")

            def sla_badge(val, threshold):
                return "[bold green]PASS[/bold green]" if val <= threshold else "[bold yellow]WARN[/bold yellow]"

            stt_str = f"{metrics['stt_latency_ms']} ms" if not metrics.get("stt_bypassed") else "Bypassed (Text)"
            table.add_row("ASR (Faster-Whisper)", stt_str, "≤ 800 ms", sla_badge(metrics['stt_latency_ms'], 800))
            table.add_row("Normalization", f"{metrics['norm_latency_ms']} ms", "≤ 50 ms", sla_badge(metrics['norm_latency_ms'], 50))
            table.add_row("LLM TTFT (Gemma 4)", f"{metrics['llm_ttft_ms']} ms", "≤ 600 ms", sla_badge(metrics['llm_ttft_ms'], 600))
            table.add_row("LLM Total", f"{metrics['llm_total_ms']} ms ({metrics['llm_tps']} tps)", "≤ 1500 ms", sla_badge(metrics['llm_total_ms'], 1500))
            table.add_row("TTS TTFA (First Chunk)", f"{metrics['tts_ttfa_ms']} ms", "≤ 900 ms", sla_badge(metrics['tts_ttfa_ms'], 900))
            table.add_row("First Audible Playback", f"{metrics.get('first_audible_ms', metrics['tts_ttfa_ms'])} ms", "≤ 1200 ms", sla_badge(metrics.get('first_audible_ms', 0), 1200))
            table.add_row("TTS Total", f"{metrics['tts_total_ms']} ms (RTF: {metrics['tts_rtf']})", "≤ 1500 ms", sla_badge(metrics['tts_total_ms'], 1500))
            table.add_row("[bold]Total End-to-End[/bold]", f"[bold]{metrics['e2e_latency_ms']} ms[/bold]", "≤ 2800 ms", sla_badge(metrics['e2e_latency_ms'], 2800))

            console.print(table)
        else:
            print("\n" + "=" * 60)
            print(f"👤 Player ({evt}): {metrics['player_input']}")
            print(f"👑 सरकार भैया:    {metrics['normalized_tts_text']}{prof_info}")
            print("-" * 60)
            print(f"STT Latency:    {metrics['stt_latency_ms']} ms {'(Bypassed)' if metrics.get('stt_bypassed') else ''}")
            print(f"Norm Latency:   {metrics['norm_latency_ms']} ms")
            print(f"LLM TTFT:       {metrics['llm_ttft_ms']} ms ({metrics['llm_tps']} tps)")
            print(f"TTS TTFA:       {metrics['tts_ttfa_ms']} ms")
            print(f"First Audible:  {metrics.get('first_audible_ms', 0.0)} ms")
            print(f"Total E2E:      {metrics['e2e_latency_ms']} ms")
            print("=" * 60)


def interactive_cli() -> None:
    """Launches the interactive Push-to-Talk and Text console with event simulation."""
    parser = argparse.ArgumentParser(description="Skyrim Hindi AI Companion Voice Pipeline — Sarkar Bhaiya")
    parser.add_argument("--mode", choices=["text", "mic", "file"], default="text", help="Interaction mode")
    parser.add_argument("--audio", type=str, default=None, help="Input audio file for file mode")
    parser.add_argument("--event", type=str, default="generic_conversation", choices=list(GAMEPLAY_EVENTS.keys()), help="Skyrim gameplay event")
    parser.add_argument("--play", action="store_true", help="Play synthesized audio through speakers")
    parser.add_argument("--no-stream", action="store_true", help="Disable sentence-level streaming")
    parser.add_argument("--sarvam", action="store_true", help="Use Sarvam AI Indic models (Bulbul TTS & Saaras STT)")
    parser.add_argument("--speaker", type=str, default="arvind", help="Sarvam TTS speaker (arvind, meera, ishita_enhi_companion)")
    args = parser.parse_args()

    pipeline = SkyrimHindiPipeline(
        provider="sarvam" if args.sarvam else None,
        sarvam_speaker=args.speaker,
    )

    current_event = args.event

    if args.mode == "file":
        if not args.audio:
            print("Error: --audio <path> is required in file mode.")
            sys.exit(1)
        res = pipeline.process_turn(
            audio_file_path=args.audio,
            event=current_event,
            play_audio=args.play,
            streaming=not args.no_stream,
        )
        pipeline.print_turn_summary(res)
        return

    print("\n👑 =============================================================== 👑")
    print("   Skyrim Hindi AI Companion — 'सरकार भैया' (Sarkar Bhaiya)        ")
    print("👑 =============================================================== 👑")
    print(f"Active Event: '{current_event}' | Mode: {args.mode.upper()}")
    print("Tip: Change event mid-game by typing ':event <event_name>' (e.g. :event missed_attack)")
    print("Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            if args.mode == "mic":
                audio_arr = FasterWhisperASR.record_microphone_push_to_talk()
                if len(audio_arr) == 0:
                    continue
                res = pipeline.process_turn(
                    raw_audio_array=audio_arr,
                    event=current_event,
                    play_audio=args.play,
                    streaming=not args.no_stream,
                )
            else:
                user_msg = input(f"\n[Player | {current_event}] Type prompt: ")
                msg_str = user_msg.strip()
                if msg_str.lower() in ("exit", "quit", "q"):
                    print("इतिहास गवाह है, सरकार भैया को कभी भुलाया नहीं जा सकेगा। अलविदा!")
                    break
                if not msg_str:
                    continue
                if msg_str.startswith(":event "):
                    new_evt = msg_str.split(" ", 1)[1].strip()
                    if new_evt in GAMEPLAY_EVENTS:
                        current_event = new_evt
                        print(f"⚡ Active event changed to: '{current_event}'")
                    else:
                        print(f"Unknown event. Valid events: {list(GAMEPLAY_EVENTS.keys())}")
                    continue

                res = pipeline.process_turn(
                    user_input_text=msg_str,
                    event=current_event,
                    play_audio=args.play,
                    streaming=not args.no_stream,
                )

            pipeline.print_turn_summary(res)

        except KeyboardInterrupt:
            print("\nExiting companion pipeline. May the Gods watch over your battles!")
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    interactive_cli()
