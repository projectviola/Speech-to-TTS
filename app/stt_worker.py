"""Speech-to-Text worker using faster-whisper on GPU."""

import threading
import time
from queue import Queue, Empty
from typing import Optional, Callable
from dataclasses import dataclass
import numpy as np

from faster_whisper import WhisperModel

from .capture import AudioSegment


@dataclass
class TranscriptResult:
    """Result from STT processing."""
    text: str
    audio: np.ndarray  # Original audio for TTS
    sample_rate: int
    timestamp: float


class STTWorker:
    """
    Worker thread that transcribes audio segments using faster-whisper.
    
    Optimized for response time - runs on GPU with float16.
    """

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cuda",
        compute_type: str = "float16",
        on_transcript: Optional[Callable[[TranscriptResult], None]] = None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.on_transcript = on_transcript

        # Input queue for audio segments
        self.queue: Queue[AudioSegment] = Queue()

        # Load model
        print(f"[STT] Loading faster-whisper {model_size} on {device}...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        print(f"[STT] Model loaded")

        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _worker_loop(self):
        """Main worker loop - processes segments from queue."""
        while self._running:
            try:
                segment = self.queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                start_time = time.time()
                
                # Transcribe
                segments, info = self.model.transcribe(
                    segment.audio,
                    language="en",
                    beam_size=1,  # Faster
                    best_of=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    vad_filter=False,  # We already did VAD
                )

                # Collect text
                text = " ".join(s.text.strip() for s in segments).strip()
                
                elapsed = time.time() - start_time
                
                if text:
                    print(f"[STT] ({elapsed:.2f}s) \"{text}\"")
                    
                    result = TranscriptResult(
                        text=text,
                        audio=segment.audio,
                        sample_rate=segment.sample_rate,
                        timestamp=segment.timestamp,
                    )
                    
                    if self.on_transcript:
                        self.on_transcript(result)
                else:
                    print(f"[STT] ({elapsed:.2f}s) <empty transcript>")

            except Exception as e:
                print(f"[STT] Error: {e}")

            self.queue.task_done()

    def enqueue(self, segment: AudioSegment):
        """Add a segment to the processing queue."""
        self.queue.put(segment)

    def start(self):
        """Start the worker thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        print("[STT] Worker started")

    def stop(self):
        """Stop the worker thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("[STT] Worker stopped")

    def clear_queue(self):
        """Clear pending items from queue."""
        cleared = 0
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
                cleared += 1
            except Empty:
                break
        if cleared:
            print(f"[STT] Cleared {cleared} pending segments")
