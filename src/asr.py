"""
Faster-Whisper Speech-to-Text Module for Hindi Skyrim Companion.

Provides GPU-accelerated (with CPU fallback) transcription for audio files
and live microphone input with Silero VAD filtering.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.sarvam_client import SarvamAIClient

import numpy as np

logger = logging.getLogger(__name__)

# Check CUDA availability via torch if present
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except (ImportError, OSError, Exception):
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False
    sd = None


class FasterWhisperASR:
    """Production-grade Faster-Whisper speech recognition wrapper optimized for Hindi."""

    def __init__(
        self,
        model_id: str = "medium",
        device: str = "auto",
        compute_type: str = "default",
        language: str = "hi",
        vad_filter: bool = True,
        beam_size: int = 5,
        download_root: Optional[str] = None,
        provider: str = "faster_whisper",
        sarvam_api_key: Optional[str] = None,
        sarvam_model: str = "saaras:v4",
    ):
        self.model_id = model_id
        self.language = language
        self.vad_filter = vad_filter
        self.beam_size = beam_size
        self.download_root = download_root
        self.provider = provider.lower()

        # Resolve device and compute_type
        self.device, self.compute_type = self._resolve_hardware(device, compute_type)
        self.model: Optional[Any] = None
        self.sarvam_client: Optional[SarvamAIClient] = None

        if self.provider == "sarvam":
            self.sarvam_client = SarvamAIClient(api_key=sarvam_api_key, stt_model=sarvam_model)
            logger.info(f"Initialized Sarvam Saaras ASR (model={sarvam_model})")
        else:
            logger.info(
                f"Initializing FasterWhisperASR (model={self.model_id}, device={self.device}, compute_type={self.compute_type})"
            )
            self._load_model()

    def _resolve_hardware(self, requested_device: str, requested_compute: str) -> Tuple[str, str]:
        """Auto-detects and optimizes compute settings for CUDA or CPU."""
        if requested_device == "auto":
            device = "cuda" if CUDA_AVAILABLE else "cpu"
        else:
            device = requested_device

        if requested_compute in ("default", "auto", None):
            if device == "cuda":
                # float16 provides the best latency/accuracy trade-off on modern NVIDIA GPUs
                compute_type = "float16"
            else:
                # int8 provides 2x speedup on CPUs via 8-bit quantization
                compute_type = "int8"
        else:
            compute_type = requested_compute

        return device, compute_type

    def _load_model(self) -> None:
        """Loads the Faster-Whisper model weights."""
        if not FASTER_WHISPER_AVAILABLE:
            logger.warning(
                "faster-whisper is not installed. To run STT, install via 'pip install faster-whisper'."
            )
            return

        try:
            self.model = WhisperModel(
                self.model_id,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
                cpu_threads=4 if self.device == "cpu" else 0,
            )
            logger.info("Faster-Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading Faster-Whisper model: {e}")
            if self.device == "cuda":
                logger.warning("Attempting CPU fallback for Faster-Whisper...")
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = WhisperModel(
                    self.model_id,
                    device="cpu",
                    compute_type="int8",
                    download_root=self.download_root,
                )
                logger.info("Faster-Whisper loaded on CPU fallback.")
            else:
                raise

    def transcribe_audio_array(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        vad_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Transcribes a NumPy float32 audio array.

        Returns dictionary with:
        - text: Complete Hindi transcription
        - duration_sec: Duration of input audio in seconds
        - latency_ms: Inference time in milliseconds
        - rtf: Real-time factor (latency_ms / (duration_sec * 1000))
        - segments: List of segment details with timestamps
        """
        if self.provider == "sarvam" and self.sarvam_client:
            temp_path = Path("/tmp/sarvam_mic_input.wav")
            int16_data = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
            with wave.open(str(temp_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(int16_data.tobytes())
            res = self.sarvam_client.transcribe_audio_file(temp_path)
            dur = len(audio_data) / float(sample_rate)
            res["duration_sec"] = round(dur, 3)
            res["rtf"] = round((res["latency_ms"] / (dur * 1000.0)), 3) if dur > 0 else 0.0
            res["segments"] = []
            return res

        if self.model is None:
            raise RuntimeError(
                "ASR model is not loaded. Ensure faster-whisper is installed and model weights exist."
            )

        # Ensure float32 in range [-1.0, 1.0]
        if audio_data.dtype != np.float32:
            if np.issubdtype(audio_data.dtype, np.integer):
                audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
            else:
                audio_data = audio_data.astype(np.float32)

        duration_sec = len(audio_data) / float(sample_rate)
        t_start = time.perf_counter()

        vad_params = vad_parameters or {"min_silence_duration_ms": 500, "speech_pad_ms": 200}

        segments_iter, info = self.model.transcribe(
            audio_data,
            language=self.language,
            task="transcribe",
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            vad_parameters=vad_params,
        )

        segments_list = []
        full_text_parts = []
        for segment in segments_iter:
            full_text_parts.append(segment.text.strip())
            segments_list.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "no_speech_prob": getattr(segment, "no_speech_prob", 0.0),
                }
            )

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0
        rtf = (latency_ms / (duration_sec * 1000.0)) if duration_sec > 0 else 0.0

        full_text = " ".join(full_text_parts).strip()
        logger.debug(f"ASR complete: '{full_text}' ({latency_ms:.1f}ms, RTF={rtf:.3f})")

        return {
            "text": full_text,
            "duration_sec": round(duration_sec, 3),
            "latency_ms": round(latency_ms, 2),
            "rtf": round(rtf, 3),
            "segments": segments_list,
            "language": info.language if hasattr(info, "language") else self.language,
        }

    def transcribe_file(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """Transcribes an audio file from disk."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self.provider == "sarvam" and self.sarvam_client:
            return self.sarvam_client.transcribe_audio_file(audio_path)

        if self.model is None:
            raise RuntimeError("ASR model is not loaded.")

        t_start = time.perf_counter()
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=self.language,
            task="transcribe",
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )

        segments_list = []
        full_text_parts = []
        for segment in segments_iter:
            full_text_parts.append(segment.text.strip())
            segments_list.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "no_speech_prob": getattr(segment, "no_speech_prob", 0.0),
                }
            )

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0
        duration_sec = getattr(info, "duration", 0.0)
        rtf = (latency_ms / (duration_sec * 1000.0)) if duration_sec > 0 else 0.0

        return {
            "text": " ".join(full_text_parts).strip(),
            "duration_sec": round(duration_sec, 3),
            "latency_ms": round(latency_ms, 2),
            "rtf": round(rtf, 3),
            "segments": segments_list,
            "language": getattr(info, "language", self.language),
        }

    @staticmethod
    def record_microphone_push_to_talk(
        sample_rate: int = 16000,
        channels: int = 1,
        output_wav_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Records from the default system microphone using push-to-talk.
        Press [ENTER] to start recording, then press [ENTER] again to stop.
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError(
                "sounddevice library is not available or no host audio device was detected. "
                "Install via 'pip install sounddevice' or provide audio via file."
            )

        input("\n[Push-to-Talk] Press [ENTER] to START speaking in Hindi...")
        print("🎙️ Recording... Speak now! Press [ENTER] when finished.")

        audio_frames = []
        recording = True

        def callback(indata, frames, time_info, status):
            if recording:
                audio_frames.append(indata.copy())

        with sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            callback=callback,
        ):
            input()  # Blocks until user presses Enter again
            recording = False

        if not audio_frames:
            return np.array([], dtype=np.float32)

        audio_data = np.concatenate(audio_frames, axis=0).flatten()
        print(f"Recorded {len(audio_data) / sample_rate:.2f} seconds of audio.")

        if output_wav_path:
            output_wav_path = Path(output_wav_path)
            output_wav_path.parent.mkdir(parents=True, exist_ok=True)
            int16_data = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
            with wave.open(str(output_wav_path), "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(int16_data.tobytes())
            logger.info(f"Saved recorded audio to {output_wav_path}")

        return audio_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Faster-Whisper Hindi ASR")
    parser.add_argument("--model", type=str, default="small", help="Whisper model size")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file")
    parser.add_argument("--mic", action="store_true", help="Record from mic (push-to-talk)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    asr = FasterWhisperASR(model_id=args.model)

    if args.audio:
        res = asr.transcribe_file(args.audio)
        print(f"Transcription: {res['text']}")
        print(f"Latency: {res['latency_ms']}ms | Duration: {res['duration_sec']}s | RTF: {res['rtf']}")
    elif args.mic:
        recorded = asr.record_microphone_push_to_talk()
        if len(recorded) > 0:
            res = asr.transcribe_audio_array(recorded)
            print(f"Transcription: {res['text']}")
            print(f"Latency: {res['latency_ms']}ms | Duration: {res['duration_sec']}s")
    else:
        print("Specify --audio <file.wav> or --mic to run ASR test.")
