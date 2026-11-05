"""
Audio Preprocessing and Dataset Curation Pipeline.

Processes raw audio files (.wav, .mp3, .ogg, .flac) into an LJSpeech-formatted
dataset (wavs/ + metadata.csv) for Tier 1 zero-shot reference conditioning and
Tier 2 XTTS-v2 / RVC custom voice fine-tuning.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
from scipy import signal as sp_signal
from scipy.io import wavfile

from src.asr import FasterWhisperASR
from src.transliterate import SkyrimHindiNormalizer

logger = logging.getLogger(__name__)

# Check for pydub
try:
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    PYDUB_AVAILABLE = True
except (ImportError, Exception):
    PYDUB_AVAILABLE = False


class AudioDatasetPreprocessor:
    """Preprocesses raw voice recordings into normalized 24kHz LJSpeech dataset for TTS."""

    def __init__(
        self,
        raw_dir: str = "data/raw_voice_lines",
        output_dir: str = "data/processed_dataset",
        target_sample_rate: int = 24000,
        min_duration_sec: float = 1.5,
        max_duration_sec: float = 12.0,
        target_rms_db: float = -20.0,
        silence_thresh_db: float = -38.0,
        min_silence_len_ms: int = 400,
        whisper_model_id: str = "small",
    ):
        # Resolve relative to repo root if needed
        self.raw_dir = Path(raw_dir) if Path(raw_dir).is_absolute() else repo_root / raw_dir
        self.output_dir = Path(output_dir) if Path(output_dir).is_absolute() else repo_root / output_dir
        self.wavs_dir = self.output_dir / "wavs"
        self.metadata_path = self.output_dir / "metadata.csv"

        self.target_sample_rate = target_sample_rate
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.target_rms_db = target_rms_db
        self.silence_thresh_db = silence_thresh_db
        self.min_silence_len_ms = min_silence_len_ms

        self.normalizer = SkyrimHindiNormalizer()
        self.whisper_model_id = whisper_model_id
        self._asr: Optional[FasterWhisperASR] = None

    @property
    def asr(self) -> FasterWhisperASR:
        """Lazy-loads ASR instance only when transcription is requested."""
        if self._asr is None:
            self._asr = FasterWhisperASR(model_id=self.whisper_model_id, language="hi")
        return self._asr

    def load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Loads audio file, converts to mono float32, returns (samples, sample_rate)."""
        suffix = file_path.suffix.lower()

        if suffix == ".wav":
            sr, data = wavfile.read(str(file_path))
            if data.dtype == np.int16:
                samples = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                samples = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                samples = (data.astype(np.float32) - 128.0) / 128.0
            else:
                samples = data.astype(np.float32)

            if len(samples.shape) > 1:
                samples = np.mean(samples, axis=1)  # Convert to mono
            return samples, sr

        elif PYDUB_AVAILABLE:
            seg = AudioSegment.from_file(str(file_path))
            seg = seg.set_channels(1)
            sr = seg.frame_rate
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
            return samples, sr

        else:
            raise RuntimeError(
                f"Cannot load {file_path.name}. pydub is required for non-WAV formats. "
                "Install with 'pip install pydub' and ensure ffmpeg is installed."
            )

    def resample(self, samples: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resamples audio to target sample rate using polyphase filtering."""
        if orig_sr == self.target_sample_rate:
            return samples

        num_target_samples = int(len(samples) * float(self.target_sample_rate) / float(orig_sr))
        resampled = sp_signal.resample(samples, num_target_samples)
        return resampled.astype(np.float32)

    def trim_silence(self, samples: np.ndarray, threshold_db: float = -45.0) -> np.ndarray:
        """Trims leading and trailing silence based on energy threshold."""
        if len(samples) == 0:
            return samples

        frame_len = int(self.target_sample_rate * 0.02)  # 20ms frames
        if frame_len == 0:
            return samples

        num_frames = len(samples) // frame_len
        if num_frames == 0:
            return samples

        frames = samples[: num_frames * frame_len].reshape(num_frames, frame_len)
        rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
        rms_db = 20.0 * np.log10(rms)

        active = np.where(rms_db > threshold_db)[0]
        if len(active) == 0:
            return samples

        start_idx = active[0] * frame_len
        end_idx = min(len(samples), (active[-1] + 1) * frame_len)
        return samples[start_idx:end_idx]

    def normalize_rms(self, samples: np.ndarray) -> np.ndarray:
        """Normalizes audio to target RMS dB with soft peak limiter."""
        if len(samples) == 0:
            return samples

        rms = np.sqrt(np.mean(samples**2) + 1e-12)
        current_db = 20.0 * np.log10(rms)
        gain_db = self.target_rms_db - current_db
        gain_linear = 10.0 ** (gain_db / 20.0)

        normalized = samples * gain_linear
        # Peak guard: prevent clipping
        peak = np.max(np.abs(normalized))
        if peak > 0.95:
            normalized = normalized * (0.95 / peak)

        return normalized.astype(np.float32)

    def slice_audio(self, samples: np.ndarray) -> List[np.ndarray]:
        """
        Slices audio array into sentence segments by silence detection.
        Ensures resulting segments satisfy min and max duration bounds.
        """
        duration = len(samples) / float(self.target_sample_rate)
        if duration <= self.max_duration_sec and duration >= self.min_duration_sec:
            trimmed = self.trim_silence(samples, self.silence_thresh_db)
            if len(trimmed) / float(self.target_sample_rate) >= self.min_duration_sec:
                return [trimmed]

        # Energy-based silence slicing
        frame_len = int(self.target_sample_rate * 0.03)  # 30ms frames
        hop_len = int(self.target_sample_rate * 0.015)  # 15ms hop
        if len(samples) < frame_len:
            return []

        # Find quiet frames
        num_frames = (len(samples) - frame_len) // hop_len + 1
        rms_values = []
        for i in range(num_frames):
            start = i * hop_len
            frame = samples[start : start + frame_len]
            rms = np.sqrt(np.mean(frame**2) + 1e-12)
            rms_values.append(20.0 * np.log10(rms))

        rms_arr = np.array(rms_values)
        silent = rms_arr < self.silence_thresh_db

        min_silent_frames = int((self.min_silence_len_ms / 1000.0) / (hop_len / self.target_sample_rate))
        slices = []
        current_start = 0
        silence_count = 0

        for i, is_sil in enumerate(silent):
            if is_sil:
                silence_count += 1
            else:
                if silence_count >= min_silent_frames:
                    split_idx = (i - silence_count // 2) * hop_len
                    chunk = samples[current_start:split_idx]
                    chunk = self.trim_silence(chunk, self.silence_thresh_db)
                    chunk_dur = len(chunk) / float(self.target_sample_rate)
                    if self.min_duration_sec <= chunk_dur <= self.max_duration_sec:
                        slices.append(chunk)
                    current_start = split_idx
                silence_count = 0

        # Final chunk
        final_chunk = samples[current_start:]
        final_chunk = self.trim_silence(final_chunk, self.silence_thresh_db)
        chunk_dur = len(final_chunk) / float(self.target_sample_rate)
        if self.min_duration_sec <= chunk_dur <= self.max_duration_sec:
            slices.append(final_chunk)

        # Fallback if no slices met criteria but original is usable
        if not slices and duration >= self.min_duration_sec:
            max_len = int(self.max_duration_sec * self.target_sample_rate)
            slices.append(samples[:max_len])

        return slices

    def save_wav(self, samples: np.ndarray, file_path: Path) -> None:
        """Writes 24kHz 16-bit mono PCM wav."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        int16_data = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(file_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.target_sample_rate)
            wf.writeframes(int16_data.tobytes())

    def auto_select_reference_speaker(self, candidates: List[Tuple[Path, float, str]]) -> Optional[Path]:
        """
        Finds the ideal 8-12s reference speaker audio clip based on:
        - Target duration (~9.0 seconds ideal for XTTS-v2 zero-shot speaker encoder)
        - Non-empty Hindi transcription
        """
        if not candidates:
            return None

        # Filter by duration window [6.0s, 12.0s]
        valid = [c for c in candidates if 6.0 <= c[1] <= 12.0 and len(c[2].strip()) > 5]
        if not valid:
            valid = candidates

        # Score closest to 9.0s
        best = min(valid, key=lambda c: abs(c[1] - 9.0))
        ref_dest = Path(__file__).resolve().parent.parent / "data" / "reference_speaker.wav"
        shutil.copyfile(str(best[0]), str(ref_dest))
        logger.info(f"Auto-selected reference speaker clip: {best[0].name} ({best[1]:.2f}s) -> {ref_dest}")
        return ref_dest

    def process(self, transcribe: bool = True) -> Dict[str, Any]:
        """Executes full data preparation pipeline across raw audio files."""
        self.wavs_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        audio_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
        raw_files = [f for f in self.raw_dir.glob("*") if f.suffix.lower() in audio_extensions]

        logger.info(f"Found {len(raw_files)} raw audio files in {self.raw_dir}")

        total_slices = 0
        total_duration = 0.0
        metadata_rows: List[List[str]] = []
        candidates_for_ref: List[Tuple[Path, float, str]] = []

        clip_counter = 1

        for raw_file in raw_files:
            try:
                samples, sr = self.load_audio(raw_file)
                samples = self.resample(samples, sr)
                slices = self.slice_audio(samples)

                for chunk in slices:
                    norm_chunk = self.normalize_rms(chunk)
                    dur = len(norm_chunk) / float(self.target_sample_rate)

                    clip_id = f"skyrim_hi_{clip_counter:05d}"
                    clip_path = self.wavs_dir / f"{clip_id}.wav"
                    self.save_wav(norm_chunk, clip_path)

                    text = ""
                    norm_text = ""
                    if transcribe:
                        try:
                            asr_res = self.asr.transcribe_audio_array(
                                norm_chunk, sample_rate=self.target_sample_rate
                            )
                            text = asr_res.get("text", "")
                            norm_text = self.normalizer.normalize(text)
                        except Exception as asr_err:
                            logger.warning(f"ASR failed for {clip_id}: {asr_err}")
                            text = "स्काईरिम साथी संवाद"
                            norm_text = text

                    if not text:
                        text = "स्काईरिम साथी संवाद"
                        norm_text = text

                    metadata_rows.append([clip_id, text, norm_text])
                    candidates_for_ref.append((clip_path, dur, text))

                    total_slices += 1
                    total_duration += dur
                    clip_counter += 1

            except Exception as e:
                logger.error(f"Error processing {raw_file.name}: {e}")

        # Write LJSpeech metadata.csv: ID|Raw_Text|Normalized_Text
        with open(self.metadata_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            for row in metadata_rows:
                writer.writerow(row)

        ref_clip = self.auto_select_reference_speaker(candidates_for_ref)

        summary = {
            "processed_clips": total_slices,
            "total_duration_hours": round(total_duration / 3600.0, 3),
            "total_duration_sec": round(total_duration, 2),
            "metadata_file": str(self.metadata_path),
            "reference_speaker": str(ref_clip) if ref_clip else None,
        }

        logger.info(
            f"Dataset Preparation Complete: {total_slices} clips, "
            f"{summary['total_duration_hours']} hours. Metadata saved to {self.metadata_path}"
        )
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skyrim Hindi Voice Dataset Preprocessor")
    parser.add_argument("--raw_dir", type=str, default="data/raw_voice_lines")
    parser.add_argument("--output_dir", type=str, default="data/processed_dataset")
    parser.add_argument("--no_transcribe", action="store_true", help="Skip Whisper ASR transcription")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    prep = AudioDatasetPreprocessor(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
    )
    result = prep.process(transcribe=not args.no_transcribe)
    print("\n--- Summary ---")
    for k, v in result.items():
        print(f"{k}: {v}")
