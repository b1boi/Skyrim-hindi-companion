"""
Sarvam AI Indic-Language API Client.

Provides cloud-accelerated Hindi Speech-to-Text (Saaras),
Text-to-Speech (Bulbul), and Chat Completion (Sarvam LLM).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

logger = logging.getLogger(__name__)

SARVAM_BASE_URL = "https://api.sarvam.ai"
SARVAM_TTS_ENDPOINT = f"{SARVAM_BASE_URL}/text-to-speech"
SARVAM_STT_ENDPOINT = f"{SARVAM_BASE_URL}/speech-to-text"
SARVAM_CHAT_ENDPOINT = f"{SARVAM_BASE_URL}/v1/chat/completions"


class SarvamAIClient:
    """Production client for Sarvam AI Indic Speech & LLM services."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        tts_speaker: str = "arvind",
        tts_model: str = "bulbul:v4-flash",
        stt_model: str = "saaras:v4",
        timeout_sec: float = 15.0,
    ):
        # Resolve API key from argument, environment variable, or config.yaml
        self.api_key = (
            api_key
            or os.environ.get("SARVAM_API_KEY")
            or self._load_key_from_config()
        )
        self.tts_speaker = tts_speaker
        self.tts_model = tts_model
        self.stt_model = stt_model
        self.timeout_sec = timeout_sec

        if not self.api_key:
            logger.warning(
                "Sarvam API key not found. Set SARVAM_API_KEY env var or provide it in config.yaml."
            )

    def _load_key_from_config(self) -> Optional[str]:
        """Loads API key from config.yaml if present."""
        cfg_path = repo_root / "config.yaml"
        if not cfg_path.exists():
            return None
        try:
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            key = cfg.get("sarvam", {}).get("api_key")
            if key and not key.startswith("${"):
                return key.strip()
        except Exception:
            pass
        return None

    def _get_headers(self) -> Dict[str, str]:
        """Returns standard Sarvam authorization headers."""
        if not self.api_key:
            raise ValueError(
                "Sarvam API Key is missing. Set it with:\n"
                "  export SARVAM_API_KEY='your_api_key'\n"
                "or place it in config.yaml under sarvam.api_key."
            )
        return {
            "api-subscription-key": self.api_key.strip(),
        }

    # --------------------------------------------------------------------------
    # Text-to-Speech (Bulbul)
    # --------------------------------------------------------------------------
    def synthesize_speech(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        speaker: Optional[str] = None,
        pace: float = 1.02,
        sample_rate: int = 24000,
    ) -> Dict[str, Any]:
        """
        Synthesizes Hindi text to audio using Sarvam Bulbul TTS.
        Returns:
            audio: np.ndarray float32
            sample_rate: int
            duration_sec: float
            latency_ms: float
            rtf: float
        """
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"

        speaker_name = speaker or self.tts_speaker

        # Sarvam TTS payload format
        payload = {
            "text": text,
            "target_language_code": "hi-IN",
            "model": self.tts_model,
            "speaker": speaker_name,
            "pace": pace,
            "speech_sample_rate": sample_rate,
            "enable_preprocessing": True,
        }

        t_start = time.perf_counter()
        try:
            resp = requests.post(
                SARVAM_TTS_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=self.timeout_sec,
            )
            # If payload format rejected, try legacy inputs format
            if resp.status_code == 422:
                payload_alt = {
                    "inputs": [text],
                    "target_language_code": "hi-IN",
                    "model": "bulbul:v1",
                    "speaker": "arvind",
                    "pitch": 0,
                    "pace": pace,
                    "loudness": 1.5,
                    "speech_sample_rate": sample_rate,
                    "enable_preprocessing": True,
                }
                resp = requests.post(
                    SARVAM_TTS_ENDPOINT,
                    headers=headers,
                    json=payload_alt,
                    timeout=self.timeout_sec,
                )

            resp.raise_for_status()
            data = resp.json()
            t_end = time.perf_counter()

            # Parse base64 audio
            base64_audio = ""
            if "audios" in data and len(data["audios"]) > 0:
                base64_audio = data["audios"][0]
            elif "audio" in data:
                base64_audio = data["audio"]
            else:
                raise ValueError(f"Unexpected response structure from Sarvam TTS: {data}")

            audio_bytes = base64.b64decode(base64_audio)
            samples, sr = self._wav_bytes_to_numpy(audio_bytes)

            latency_ms = (t_end - t_start) * 1000.0
            duration_sec = len(samples) / float(sr)
            rtf = (latency_ms / (duration_sec * 1000.0)) if duration_sec > 0 else 0.0

            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(audio_bytes)
                logger.debug(f"Saved Sarvam TTS audio to {output_path}")

            return {
                "audio": samples,
                "sample_rate": sr,
                "duration_sec": round(duration_sec, 3),
                "latency_ms": round(latency_ms, 2),
                "ttfa_ms": round(latency_ms, 2),
                "rtf": round(rtf, 3),
                "provider": "sarvam_bulbul",
            }

        except Exception as e:
            logger.error(f"Sarvam TTS failed: {e}")
            raise

    # --------------------------------------------------------------------------
    # Speech-to-Text (Saaras)
    # --------------------------------------------------------------------------
    def transcribe_audio_file(
        self,
        audio_file_path: Union[str, Path],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribes an audio file using Sarvam Saaras ASR.
        Returns:
            text: Hindi transcription
            latency_ms: time taken in ms
            provider: 'sarvam_saaras'
        """
        headers = self._get_headers()
        audio_path = Path(audio_file_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found at {audio_path}")

        t_start = time.perf_counter()
        with open(audio_path, "rb") as f:
            files = {
                "file": (audio_path.name, f, "audio/wav"),
            }
            data = {
                "model": model or self.stt_model,
                "mode": "transcribe",
                "language_code": "hi-IN",
            }

            resp = requests.post(
                SARVAM_STT_ENDPOINT,
                headers=headers,
                files=files,
                data=data,
                timeout=self.timeout_sec,
            )

        resp.raise_for_status()
        res_data = resp.json()
        t_end = time.perf_counter()

        transcript = res_data.get("transcript", "").strip()
        latency_ms = (t_end - t_start) * 1000.0

        return {
            "text": transcript,
            "latency_ms": round(latency_ms, 2),
            "provider": "sarvam_saaras",
            "language_code": res_data.get("language_code", "hi-IN"),
        }

    # --------------------------------------------------------------------------
    # Chat Completion (Sarvam LLM)
    # --------------------------------------------------------------------------
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "sarvam-2b",
        temperature: float = 0.7,
        max_tokens: int = 250,
    ) -> Dict[str, Any]:
        """Executes chat completion via Sarvam AI API."""
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        t_start = time.perf_counter()
        resp = requests.post(
            SARVAM_CHAT_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()
        t_end = time.perf_counter()

        reply = data["choices"][0]["message"]["content"].strip()
        total_ms = (t_end - t_start) * 1000.0

        return {
            "text": reply,
            "latency_ms": round(total_ms, 2),
            "provider": "sarvam_llm",
        }

    @staticmethod
    def _wav_bytes_to_numpy(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Parses in-memory WAV bytes into float32 numpy array and sample rate."""
        with io.BytesIO(wav_bytes) as bio:
            with wave.open(bio, "rb") as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                raw_bytes = wf.readframes(n_frames)
                int16_arr = np.frombuffer(raw_bytes, dtype=np.int16)
                float32_arr = int16_arr.astype(np.float32) / 32768.0
                return float32_arr, sr


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Sarvam AI Indic Speech & Audio APIs")
    parser.add_argument("--key", type=str, default=None, help="Sarvam API subscription key")
    parser.add_argument("--tts", type=str, default=None, help="Synthesize Hindi text with Bulbul")
    parser.add_argument("--speaker", type=str, default="arvind", help="Speaker name (arvind, meera)")
    parser.add_argument("--out", type=str, default="sarvam_output.wav", help="Output WAV path")
    parser.add_argument("--stt", type=str, default=None, help="Transcribe audio file with Saaras")
    args = parser.parse_args()

    client = SarvamAIClient(api_key=args.key, tts_speaker=args.speaker)

    if not client.api_key:
        print("\n❌ Error: No Sarvam API key provided.")
        print("Please provide it via:")
        print("  export SARVAM_API_KEY='your_sarvam_api_key'")
        print("or run with:")
        print("  python3 src/sarvam_client.py --key 'your_key' --tts 'नमस्ते दोवाकीन'")
        sys.exit(1)

    print(f"✅ Sarvam API Key detected (prefix: {client.api_key[:6]}...)")

    if args.tts:
        print(f"\n🔊 Synthesizing speech with Sarvam Bulbul: '{args.tts}'")
        res = client.synthesize_speech(args.tts, output_path=args.out, speaker=args.speaker)
        print(f"Saved to: {args.out}")
        print(f"Duration: {res['duration_sec']}s | Latency: {res['latency_ms']}ms | RTF: {res['rtf']}")

    elif args.stt:
        print(f"\n🎙️ Transcribing audio with Sarvam Saaras: '{args.stt}'")
        res = client.transcribe_audio_file(args.stt)
        print(f"Transcription: {res['text']}")
        print(f"Latency: {res['latency_ms']}ms | Language: {res['language_code']}")

    else:
        # Default smoke test
        sample_text = "मेरी तलवार तुम्हारे साथ है, दोवाकीन। हम व्हाइटरन की ओर बढ़ेंगे।"
        print(f"\nTesting Sarvam Bulbul TTS on sample companion line: '{sample_text}'")
        res = client.synthesize_speech(sample_text, output_path=args.out)
        print(f"Saved to: {args.out}")
        print(f"Duration: {res['duration_sec']}s | Latency: {res['latency_ms']}ms | RTF: {res['rtf']}")

