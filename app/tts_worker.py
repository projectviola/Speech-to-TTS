"""Text-to-Speech worker using kokoro-onnx on GPU."""

import threading
import time
import os
from queue import Queue, Empty
from typing import Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import urllib.request

import parselmouth
from parselmouth.praat import call

from kokoro_onnx import Kokoro

from .stt_worker import TranscriptResult


# Model URLs
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


@dataclass
class TTSResult:
    """Result from TTS synthesis."""
    audio: np.ndarray
    sample_rate: int
    transcript: str
    timestamp: float


def download_file(url: str, dest: Path, desc: str):
    """Download a file with progress."""
    if dest.exists():
        print(f"[TTS] {desc} already exists")
        return
    
    print(f"[TTS] Downloading {desc}...")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\r[TTS] {desc}: {percent}%", end="", flush=True)
    
    urllib.request.urlretrieve(url, dest, progress_hook)
    print()  # newline after progress


def pitch_shift(audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    """
    Shift pitch by semitones using Praat's PSOLA algorithm.
    
    This is the industry-standard method for natural voice pitch shifting.
    """
    if semitones == 0:
        return audio
    
    # Convert to Praat Sound object
    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sample_rate)
    
    # Calculate pitch shift ratio
    ratio = 2 ** (semitones / 12.0)
    
    # Create manipulation object (PSOLA analysis)
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
    
    # Extract pitch tier
    pitch_tier = call(manipulation, "Extract pitch tier")
    
    # Multiply all pitch points by the ratio
    call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, ratio)
    
    # Replace pitch tier in manipulation
    call([manipulation, pitch_tier], "Replace pitch tier")
    
    # Resynthesize with new pitch (PSOLA)
    shifted_sound = call(manipulation, "Get resynthesis (overlap-add)")
    
    # Extract samples
    result = shifted_sound.values[0]
    
    return result.astype(np.float32)


class TTSWorker:
    """
    Worker thread that synthesizes speech using kokoro-onnx.
    
    Runs on GPU (CUDA) for fast synthesis.
    Immediately synthesizes as transcripts arrive - no waiting.
    """

    def __init__(
        self,
        voice: str = "af_heart",
        speed: float = 1.0,
        pitch_semitones: float = 0.0,
        lang: str = "en-us",
        models_dir: str = "models",
        on_audio: Optional[Callable[[TTSResult], None]] = None,
    ):
        self.voice = voice
        self.speed = speed
        self.pitch_semitones = pitch_semitones
        self.lang = lang
        self.on_audio = on_audio

        # Input queue for transcripts
        self.queue: Queue[TranscriptResult] = Queue()

        # Ensure models directory exists
        models_path = Path(models_dir)
        models_path.mkdir(exist_ok=True)
        
        # Model file paths
        model_file = models_path / "kokoro-v1.0.onnx"
        voices_file = models_path / "voices-v1.0.bin"
        
        # Download models if needed
        download_file(MODEL_URL, model_file, "kokoro-v1.0.onnx (~350MB)")
        download_file(VOICES_URL, voices_file, "voices-v1.0.bin (~5MB)")

        # Load model
        print(f"[TTS] Loading kokoro-onnx with voice '{voice}'...")
        self.model = Kokoro(str(model_file), str(voices_file))
        print(f"[TTS] Model loaded (pitch: {pitch_semitones:+.1f} semitones)")

        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _worker_loop(self):
        """Main worker loop - processes transcripts from queue."""
        while self._running:
            try:
                transcript = self.queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                start_time = time.time()
                
                # Synthesize speech
                audio, sample_rate = self.model.create(
                    transcript.text,
                    voice=self.voice,
                    speed=self.speed,
                    lang=self.lang,
                )
                
                # Apply pitch shift if configured
                if self.pitch_semitones != 0:
                    audio = pitch_shift(audio, sample_rate, self.pitch_semitones)
                
                elapsed = time.time() - start_time
                duration = len(audio) / sample_rate
                
                print(f"[TTS] ({elapsed:.2f}s) Generated {duration:.2f}s audio")

                result = TTSResult(
                    audio=audio,
                    sample_rate=sample_rate,
                    transcript=transcript.text,
                    timestamp=time.time(),
                )

                if self.on_audio:
                    self.on_audio(result)

            except Exception as e:
                print(f"[TTS] Error: {e}")
                import traceback
                traceback.print_exc()

            self.queue.task_done()

    def enqueue(self, transcript: TranscriptResult):
        """Add a transcript to the synthesis queue."""
        self.queue.put(transcript)

    def start(self):
        """Start the worker thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        print("[TTS] Worker started")

    def stop(self):
        """Stop the worker thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("[TTS] Worker stopped")

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
            print(f"[TTS] Cleared {cleared} pending transcripts")
