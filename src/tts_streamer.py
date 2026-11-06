"""
Streaming TTS Pipeline & Thread-Safe Sequential Audio Playback Queue.

Enables low-latency voice output by:
1. Incrementally chunking incoming LLM tokens into natural Hindi sentences/clauses.
2. Synthesizing sentence N concurrently while sentence N-1 is playing.
3. Enforcing strict FIFO sequential audio playback with zero chunk overlap or reordering.
4. Measuring genuine Time-to-First-Audio (TTFA) and First Audible Playback latency.
5. Concatenating streamed audio chunks for Skyrim Mantella file logging.
"""

from __future__ import annotations

import logging
import queue
import re
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.tts import XTTSHindiSynthesizer, SOUNDDEVICE_AVAILABLE

if SOUNDDEVICE_AVAILABLE:
    import sounddevice as sd
else:
    sd = None

logger = logging.getLogger(__name__)


class StreamingSentenceChunker:
    """
    Incrementally buffers streaming LLM tokens and yields complete Hindi sentences
    or natural pauses delimited by Devanagari purna viram (।), '?', '!', '\n',
    or comma clauses when long enough for speech synthesis.
    """

    def __init__(self, min_chars: int = 20, max_chars: int = 120):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.buffer = ""
        # Delimiters: purna viram, question mark, exclamation, newline
        self.terminal_re = re.compile(r"([।?!;\n]+)")

    def feed(self, token: str) -> List[str]:
        """
        Feeds a newly arrived token from LLM stream into the buffer.
        Returns any newly completed sentences/clauses ready for TTS synthesis.
        """
        self.buffer += token
        completed_chunks: List[str] = []

        while True:
            match = self.terminal_re.search(self.buffer)
            if not match:
                # If buffer exceeds max_chars without punctuation, split at a natural word boundary
                if len(self.buffer) >= self.max_chars:
                    # Look for comma or space
                    comma_pos = self.buffer.rfind(",", 0, self.max_chars)
                    if comma_pos != -1 and comma_pos >= self.min_chars:
                        chunk = self.buffer[: comma_pos + 1].strip()
                        self.buffer = self.buffer[comma_pos + 1 :].lstrip()
                        if chunk:
                            completed_chunks.append(chunk)
                        continue

                    space_pos = self.buffer.rfind(" ", self.min_chars, self.max_chars)
                    if space_pos != -1:
                        chunk = self.buffer[:space_pos].strip()
                        self.buffer = self.buffer[space_pos + 1 :].lstrip()
                        if chunk:
                            completed_chunks.append(chunk)
                        continue
                break

            end_idx = match.end()
            candidate = self.buffer[:end_idx].strip()

            # If candidate is too short (e.g. abbreviation or single word), keep buffering unless long enough
            if len(candidate) < self.min_chars:
                # Check if there is another delimiter later in buffer
                rest = self.buffer[end_idx:]
                next_match = self.terminal_re.search(rest)
                if next_match:
                    # Merge with next segment
                    candidate_merged = self.buffer[: end_idx + next_match.end()].strip()
                    self.buffer = self.buffer[end_idx + next_match.end() :].lstrip()
                    if candidate_merged:
                        completed_chunks.append(candidate_merged)
                    continue
                else:
                    # Not enough chars yet, keep in buffer
                    break
            else:
                completed_chunks.append(candidate)
                self.buffer = self.buffer[end_idx:].lstrip()

        return completed_chunks

    def flush(self) -> List[str]:
        """Flushes any remaining text in the buffer at the end of LLM generation."""
        remainder = self.buffer.strip()
        self.buffer = ""
        if remainder:
            return [remainder]
        return []


class ThreadSafeAudioPlaybackQueue:
    """
    Manages an ordered, sequential FIFO queue for audio playback.
    A dedicated playback thread dequeues and plays audio chunks sequentially,
    ensuring that sentence N is synthesized while sentence N-1 is playing
    without any audio overlap, clipping, or out-of-order playback.
    """

    def __init__(self, sample_rate: int = 24000, mock_playback: bool = False):
        self.sample_rate = sample_rate
        self.mock_playback = mock_playback or not SOUNDDEVICE_AVAILABLE
        self.queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self.collected_audio_chunks: List[np.ndarray] = []
        self._stop_event = threading.Event()
        self._playback_thread: Optional[threading.Thread] = None
        self.first_playback_time: Optional[float] = None
        self.total_playback_duration_sec: float = 0.0
        self.lock = threading.Lock()

    def start(self) -> None:
        """Starts the background audio playback worker thread."""
        self._stop_event.clear()
        self.first_playback_time = None
        self.total_playback_duration_sec = 0.0
        self.collected_audio_chunks.clear()
        self._playback_thread = threading.Thread(
            target=self._playback_worker, daemon=True, name="SarkarBhaiyaAudioWorker"
        )
        self._playback_thread.start()

    def enqueue(self, audio_chunk: np.ndarray) -> None:
        """Enqueues a synthesized float32 numpy audio chunk for playback."""
        if audio_chunk is None or len(audio_chunk) == 0:
            return
        with self.lock:
            self.collected_audio_chunks.append(audio_chunk)
            self.total_playback_duration_sec += len(audio_chunk) / float(self.sample_rate)
        self.queue.put(audio_chunk)

    def finish_stream(self) -> None:
        """Signals that no more chunks will be added for this turn (sentinel None)."""
        self.queue.put(None)

    def wait_until_complete(self, timeout_sec: Optional[float] = None) -> None:
        """Blocks until all queued audio chunks have finished playing."""
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=timeout_sec)

    def stop(self) -> None:
        """Aborts playback immediately and drains the queue."""
        self._stop_event.set()
        self.finish_stream()
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)

    def _playback_worker(self) -> None:
        """Worker loop executing sequentially in background."""
        while not self._stop_event.is_set():
            try:
                chunk = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if chunk is None:
                # Sentinel reached
                self.queue.task_done()
                break

            # Record timestamp of first audible playback
            if self.first_playback_time is None:
                self.first_playback_time = time.perf_counter()

            # Play audio chunk
            duration = len(chunk) / float(self.sample_rate)
            if not self.mock_playback and sd is not None:
                try:
                    sd.play(chunk, samplerate=self.sample_rate)
                    sd.wait()
                except Exception as e:
                    logger.warning(f"Audio playback error: {e}. Switching to sleep simulation.")
                    time.sleep(duration)
            else:
                # Simulate real-time playback duration in headless/mock mode
                time.sleep(min(duration, 0.05))  # Fast-forward slightly in mock mode

            self.queue.task_done()

    def get_full_audio(self) -> np.ndarray:
        """Returns the full concatenated audio array from all enqueued chunks."""
        with self.lock:
            if not self.collected_audio_chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self.collected_audio_chunks)

    def save_full_audio_wav(self, output_path: Union[str, Path]) -> None:
        """Saves all synthesized audio as a single 16-bit PCM WAV."""
        full_audio = self.get_full_audio()
        if len(full_audio) == 0:
            return
        XTTSHindiSynthesizer.save_wav(full_audio, output_path, self.sample_rate)


class StreamingTTSPipeline:
    """
    Coordinates LLM token streaming, sentence chunking, parallel TTS synthesis,
    and sequential audio playback.
    """

    def __init__(
        self,
        tts_synthesizer: XTTSHindiSynthesizer,
        mock_playback: bool = False,
    ):
        self.tts = tts_synthesizer
        self.mock_playback = mock_playback

    def run_streaming_turn(
        self,
        token_stream: Generator[Tuple[str, Optional[Dict[str, Any]]], None, None],
        play_audio: bool = True,
        output_wav_path: Optional[Union[str, Path]] = None,
        turn_start_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Consumes streaming tokens from LLM, chunks into sentences, synthesizes audio
        concurrently with playback, and records latency metrics.
        """
        t_start = turn_start_time if turn_start_time is not None else time.perf_counter()

        chunker = StreamingSentenceChunker(min_chars=20)
        playback_queue = ThreadSafeAudioPlaybackQueue(
            sample_rate=self.tts.sample_rate,
            mock_playback=(not play_audio) or self.mock_playback,
        )

        if play_audio:
            playback_queue.start()

        full_tokens: List[str] = []
        synthesized_sentences: List[str] = []
        llm_metrics: Optional[Dict[str, Any]] = None

        first_chunk_synth_ms: Optional[float] = None
        first_audible_ms: Optional[float] = None
        total_synth_ms = 0.0

        def process_completed_sentences(sentences: List[str]):
            nonlocal first_chunk_synth_ms, total_synth_ms
            for s in sentences:
                clean_s = s.strip()
                if not clean_s:
                    continue
                synthesized_sentences.append(clean_s)

                t_synth_start = time.perf_counter()
                synth_res = self.tts.synthesize(clean_s)
                t_synth_end = time.perf_counter()

                chunk_dur = (t_synth_end - t_synth_start) * 1000.0
                total_synth_ms += chunk_dur

                if first_chunk_synth_ms is None:
                    first_chunk_synth_ms = (t_synth_end - t_start) * 1000.0

                if play_audio:
                    playback_queue.enqueue(synth_res["audio"])

        # Stream tokens from LLM
        for token, metrics in token_stream:
            if token:
                full_tokens.append(token)
                new_sentences = chunker.feed(token)
                if new_sentences:
                    process_completed_sentences(new_sentences)
            if metrics:
                llm_metrics = metrics

        # Flush any remaining text in buffer
        remaining = chunker.flush()
        if remaining:
            process_completed_sentences(remaining)

        # Signal end of audio stream to playback worker
        if play_audio:
            playback_queue.finish_stream()
            playback_queue.wait_until_complete()

            if playback_queue.first_playback_time:
                first_audible_ms = (playback_queue.first_playback_time - t_start) * 1000.0

        t_end = time.perf_counter()
        e2e_total_ms = (t_end - t_start) * 1000.0

        # Save concatenated audio if requested
        if output_wav_path and play_audio:
            playback_queue.save_full_audio_wav(output_wav_path)

        full_text = "".join(full_tokens).strip()

        return {
            "full_text": full_text,
            "sentences": synthesized_sentences,
            "llm_metrics": llm_metrics or {},
            "ttfa_ms": round(first_chunk_synth_ms or 0.0, 2),
            "first_audible_ms": round(first_audible_ms or (first_chunk_synth_ms or 0.0), 2),
            "tts_total_synth_ms": round(total_synth_ms, 2),
            "e2e_latency_ms": round(e2e_total_ms, 2),
            "audio_duration_sec": round(playback_queue.total_playback_duration_sec, 3),
            "audio_path": str(output_wav_path) if output_wav_path else None,
        }

