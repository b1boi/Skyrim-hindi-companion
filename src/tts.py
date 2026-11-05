"""
Coqui XTTS-v2 Zero-Shot Hindi Speech Synthesis & Streaming Chunker.

Optimized for CUDA GPU acceleration with graceful CPU fallback.
Provides streaming sentence-level chunking for sub-second Time-to-First-Audio (TTFA).
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.sarvam_client import SarvamAIClient

logger = logging.getLogger(__name__)

# Hardware detection
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except (ImportError, OSError, Exception):
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    from TTS.api import TTS
    COQUI_TTS_AVAILABLE = True
except (ImportError, Exception):
    COQUI_TTS_AVAILABLE = False
    TTS = None

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False
    sd = None


class XTTSHindiSynthesizer:
    """Production XTTS-v2 synthesizer wrapper with streaming chunker and latency tracking."""

    def __init__(
        self,
        model_id: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        reference_audio_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        language: str = "hi",
        sample_rate: int = 24000,
        temperature: float = 0.65,
        speed: float = 1.0,
        provider: str = "xtts",
        sarvam_api_key: Optional[str] = None,
        sarvam_speaker: str = "arvind",
        sarvam_model: str = "bulbul:v4-flash",
    ):
        self.model_id = model_id
        self.language = language
        self.sample_rate = sample_rate
        self.temperature = temperature
        self.speed = speed
        self.provider = provider.lower()
        self.sarvam_speaker = sarvam_speaker

        # Resolve speaker reference
        repo_root = Path(__file__).resolve().parent.parent
        self.reference_audio_path = (
            Path(reference_audio_path)
            if reference_audio_path
            else repo_root / "data" / "reference_speaker.wav"
        )

        # Resolve compute device
        if device == "auto":
            self.device = "cuda" if CUDA_AVAILABLE else "cpu"
        else:
            self.device = device

        self.tts: Optional[Any] = None
        self.sarvam_client: Optional[SarvamAIClient] = None

        if self.provider == "sarvam":
            self.sarvam_client = SarvamAIClient(
                api_key=sarvam_api_key,
                tts_speaker=sarvam_speaker,
                tts_model=sarvam_model,
            )
            logger.info(f"Initialized Sarvam Bulbul TTS (speaker={sarvam_speaker}, model={sarvam_model})")
        else:
            self._load_model()

    def _load_model(self) -> None:
        """Loads XTTS-v2 model into memory with device optimization."""
        if not COQUI_TTS_AVAILABLE:
            logger.warning(
                "Coqui TTS is not installed in the current environment. "
                "Install with 'pip install TTS'. Synthetic formant audio fallback active."
            )
            return

        try:
            logger.info(f"Loading XTTS-v2 model on {self.device}...")
            gpu_flag = (self.device == "cuda")
            self.tts = TTS(model_name=self.model_id, gpu=gpu_flag)

            # Warm-up / verify speaker wav
            if not self.reference_audio_path.exists():
                logger.warning(
                    f"Reference audio not found at {self.reference_audio_path}. "
                    "Ensure reference_speaker.wav is generated or provided."
                )
            else:
                logger.info(f"Loaded reference speaker audio: {self.reference_audio_path}")

        except Exception as e:
            logger.error(f"Failed to load Coqui XTTS model on {self.device}: {e}")
            if self.device == "cuda":
                logger.warning("Attempting CPU fallback for XTTS...")
                self.device = "cpu"
                try:
                    self.tts = TTS(model_name=self.model_id, gpu=False)
                    logger.info("XTTS loaded successfully on CPU fallback.")
                except Exception as inner_e:
                    logger.error(f"CPU fallback also failed: {inner_e}")
                    self.tts = None
            else:
                self.tts = None

    def split_into_sentence_chunks(self, text: str, min_chars: int = 25) -> List[str]:
        """
        Splits Hindi text into natural speakable sentence/clause units
        delimited by Devanagari purna viram (।), question mark (?), exclamation (!),
        or comma clauses for low-latency streaming playback.
        """
        if not text:
            return []

        # Split on sentence terminals
        raw_chunks = re.split(r"([।?!;\n]+)", text)
        sentences: List[str] = []
        current = ""

        for part in raw_chunks:
            if not part:
                continue
            if re.match(r"[।?!;\n]+", part):
                current += part
                if len(current.strip()) >= min_chars:
                    sentences.append(current.strip())
                    current = ""
            else:
                current += part

        if current.strip():
            sentences.append(current.strip())

        # If no punctuation was present, split on spaces if length is large
        if not sentences and text.strip():
            words = text.split()
            chunk: List[str] = []
            for w in words:
                chunk.append(w)
                if len(" ".join(chunk)) >= min_chars:
                    sentences.append(" ".join(chunk))
                    chunk = []
            if chunk:
                sentences.append(" ".join(chunk))

        return [s for s in sentences if s.strip()]

    def synthesize(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Synthesizes complete Hindi audio for a given string.
        Returns:
            audio_array: np.ndarray float32 [-1.0, 1.0]
            sample_rate: int
            latency_ms: total inference latency in ms
            ttfa_ms: time-to-first-audio (same as latency_ms for single batch)
            duration_sec: duration of synthesized speech
            rtf: real-time factor
        """
        t_start = time.perf_counter()

        if self.provider == "sarvam" and self.sarvam_client:
            try:
                res = self.sarvam_client.synthesize_speech(
                    text=text,
                    output_path=output_path,
                    speaker=self.sarvam_speaker,
                    sample_rate=self.sample_rate,
                )
                res["is_fallback"] = False
                return res
            except Exception as e:
                logger.error(f"Sarvam synthesis failed: {e}. Falling back to harmonic synthesizer.")
                audio_array = self._synthesize_fallback(text)
        elif self.tts is not None and self.reference_audio_path.exists():
            try:
                # Run XTTS-v2 inference
                wav = self.tts.tts(
                    text=text,
                    speaker_wav=str(self.reference_audio_path),
                    language=self.language,
                    temperature=self.temperature,
                    speed=self.speed,
                )
                audio_array = np.array(wav, dtype=np.float32)
            except Exception as e:
                logger.error(f"XTTS synthesis error: {e}. Using harmonic fallback.")
                audio_array = self._synthesize_fallback(text)
        else:
            audio_array = self._synthesize_fallback(text)

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0
        duration_sec = len(audio_array) / float(self.sample_rate)
        rtf = (latency_ms / (duration_sec * 1000.0)) if duration_sec > 0 else 0.0

        if output_path:
            self.save_wav(audio_array, output_path, self.sample_rate)

        return {
            "audio": audio_array,
            "sample_rate": self.sample_rate,
            "duration_sec": round(duration_sec, 3),
            "latency_ms": round(latency_ms, 2),
            "ttfa_ms": round(latency_ms, 2),
            "rtf": round(rtf, 3),
            "is_fallback": (self.tts is None),
        }

    def stream_synthesize(
        self,
        text: str,
        min_chunk_chars: int = 30,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streams audio chunks sentence-by-sentence for minimal Time-to-First-Audio (TTFA).
        Yields chunk dictionaries containing audio slice and latency metrics.
        """
        t_stream_start = time.perf_counter()
        sentence_chunks = self.split_into_sentence_chunks(text, min_chars=min_chunk_chars)
        first_chunk_yielded = False

        for idx, chunk_text in enumerate(sentence_chunks):
            t_chunk_start = time.perf_counter()
            chunk_res = self.synthesize(chunk_text)
            t_chunk_end = time.perf_counter()

            ttfa_ms = (t_chunk_end - t_stream_start) * 1000.0 if not first_chunk_yielded else None
            first_chunk_yielded = True

            yield {
                "chunk_index": idx,
                "text": chunk_text,
                "audio": chunk_res["audio"],
                "sample_rate": self.sample_rate,
                "duration_sec": chunk_res["duration_sec"],
                "chunk_latency_ms": round((t_chunk_end - t_chunk_start) * 1000.0, 2),
                "ttfa_ms": round(ttfa_ms, 2) if ttfa_ms else None,
                "is_fallback": chunk_res["is_fallback"],
            }

    def _synthesize_fallback(self, text: str) -> np.ndarray:
        """
        Deterministic, vocal-like harmonic formant synthesizer for testing & benchmarking
        when XTTS model weights or dependencies are offline.
        """
        # Duration proportional to character count (approx 15 chars/sec speech rate)
        duration = max(1.2, len(text) * 0.08)
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)

        # Companion vocal formant frequencies (F0=130Hz baritone)
        f0 = 130.0
        carrier = (
            0.4 * np.sin(2 * np.pi * f0 * t)
            + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
            + 0.15 * np.sin(2 * np.pi * 3 * f0 * t)
            + 0.08 * np.sin(2 * np.pi * 4 * f0 * t)
        )

        # Speech envelope with word cadence
        cadence = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
        fade_len = int(self.sample_rate * 0.05)
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)

        envelope = np.ones(num_samples)
        envelope[:fade_len] = fade_in
        envelope[-fade_len:] = fade_out

        signal = carrier * cadence * envelope
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = (signal / max_val) * 0.70

        return signal.astype(np.float32)

    @staticmethod
    def save_wav(
        audio_array: np.ndarray,
        output_path: Union[str, Path],
        sample_rate: int = 24000,
    ) -> None:
        """Saves float32 numpy audio array as 16-bit PCM WAV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        clipped = np.clip(audio_array, -1.0, 1.0)
        int16_data = (clipped * 32767).astype(np.int16)

        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int16_data.tobytes())

        logger.debug(f"Saved audio to {output_path}")

    @staticmethod
    def play_audio(audio_array: np.ndarray, sample_rate: int = 24000) -> None:
        """Plays audio through the host audio output if available."""
        if not SOUNDDEVICE_AVAILABLE:
            logger.warning("sounddevice not available for audio playback.")
            return

        try:
            sd.play(audio_array, samplerate=sample_rate)
            sd.wait()
        except Exception as e:
            logger.error(f"Playback error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test XTTS-v2 Hindi Synthesizer")
    parser.add_argument(
        "--text",
        type=str,
        default="मेरी तलवार तुम्हारे साथ है, दोवाकीन। हम ड्रैगन्सरीच की ओर बढ़ेंगे।",
        help="Hindi text to synthesize",
    )
    parser.add_argument("--out", type=str, default="test_output.wav", help="Output WAV path")
    parser.add_argument("--stream", action="store_true", help="Test streaming chunker")
    args = parser.parse_args()

    synthesizer = XTTSHindiSynthesizer()

    if args.stream:
        print(f"Streaming synthesis for: '{args.text}'")
        for chunk in synthesizer.stream_synthesize(args.text):
            print(
                f"Chunk {chunk['chunk_index']}: '{chunk['text']}' | "
                f"Duration={chunk['duration_sec']}s | "
                f"Latency={chunk['chunk_latency_ms']}ms | TTFA={chunk['ttfa_ms']}"
            )
    else:
        res = synthesizer.synthesize(args.text, output_path=args.out)
        print(f"Synthesized: '{args.text}'")
        print(
            f"Saved to: {args.out} | Duration: {res['duration_sec']}s | "
            f"Latency: {res['latency_ms']}ms | RTF: {res['rtf']} | Fallback: {res['is_fallback']}"
        )
