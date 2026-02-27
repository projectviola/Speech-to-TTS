"""Audio capture with Silero VAD for speech segmentation."""

import threading
import time
import collections
import numpy as np
import sounddevice as sd
from queue import Queue
from typing import Callable, Optional
from dataclasses import dataclass

# Silero VAD
import torch
torch.set_num_threads(1)  # Optimize for real-time


@dataclass
class AudioSegment:
    """A segment of audio with speech."""
    audio: np.ndarray
    sample_rate: int
    timestamp: float  # When speech ended


class VADState:
    IDLE = "idle"
    IN_SPEECH = "in_speech"


class AudioCaptureVAD:
    """
    Continuous audio capture with VAD-based segmentation.
    
    Emits segments via callback when speech ends, including:
    - 2 second prebuffer before speech start
    - The speech itself
    - A short tail after speech end
    """

    def __init__(
        self,
        device_index: int,
        sample_rate: int = 16000,
        frame_ms: int = 32,  # Exactly 32ms = 512 samples at 16kHz (required by Silero)
        prebuffer_seconds: float = 2.0,
        tail_seconds: float = 0.15,
        start_threshold: float = 0.5,
        end_threshold: float = 0.35,
        min_speech_ms: int = 250,
        min_silence_ms: int = 300,
        on_segment: Optional[Callable[[AudioSegment], None]] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
    ):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.prebuffer_seconds = prebuffer_seconds
        self.tail_seconds = tail_seconds
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms

        self.on_segment = on_segment
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end

        # Ring buffer for prebuffer (stores last N seconds)
        prebuffer_frames = int(prebuffer_seconds * 1000 / frame_ms)
        self.prebuffer = collections.deque(maxlen=prebuffer_frames)

        # Utterance buffer (accumulates during speech)
        self.utterance_buffer: list[np.ndarray] = []

        # Tail buffer
        tail_frames = int(tail_seconds * 1000 / frame_ms)
        self.tail_frames_needed = tail_frames
        self.tail_buffer: list[np.ndarray] = []

        # VAD state
        self.state = VADState.IDLE
        self.speech_frames = 0
        self.silence_frames = 0
        self.min_speech_frames = int(min_speech_ms / frame_ms)
        self.min_silence_frames = int(min_silence_ms / frame_ms)

        # Load Silero VAD
        self.vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self.vad_model.eval()

        # Audio input buffer (for accumulating samples)
        self._audio_buffer = np.array([], dtype=np.float32)

        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None

        # Last speech end timestamp (for playback gating)
        self.last_speech_end_time: float = 0.0

    def _vad_probability(self, audio: np.ndarray) -> float:
        """Get speech probability from VAD model."""
        # Ensure correct shape and type
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize if needed
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0

        tensor = torch.from_numpy(audio)
        with torch.no_grad():
            prob = self.vad_model(tensor, self.sample_rate).item()
        return prob

    def _process_frame(self, audio: np.ndarray):
        """Process a single audio frame through VAD state machine."""
        prob = self._vad_probability(audio)
        is_speech = prob >= self.start_threshold if self.state == VADState.IDLE else prob >= self.end_threshold

        if self.state == VADState.IDLE:
            # Always add to prebuffer
            self.prebuffer.append(audio.copy())

            if is_speech:
                self.speech_frames += 1
                if self.speech_frames >= self.min_speech_frames:
                    # Transition to IN_SPEECH
                    self.state = VADState.IN_SPEECH
                    self.silence_frames = 0
                    
                    # Start utterance with prebuffer contents
                    self.utterance_buffer = list(self.prebuffer)
                    
                    if self.on_speech_start:
                        self.on_speech_start()
            else:
                self.speech_frames = 0

        elif self.state == VADState.IN_SPEECH:
            # Add to utterance
            self.utterance_buffer.append(audio.copy())

            if not is_speech:
                self.silence_frames += 1
                if self.silence_frames >= self.min_silence_frames:
                    # Speech ended - emit segment
                    self._emit_segment()
                    
                    # Reset state
                    self.state = VADState.IDLE
                    self.speech_frames = 0
                    self.silence_frames = 0
                    self.utterance_buffer = []
                    self.prebuffer.clear()
                    
                    self.last_speech_end_time = time.time()
                    
                    if self.on_speech_end:
                        self.on_speech_end()
            else:
                self.silence_frames = 0

    def _emit_segment(self):
        """Emit the accumulated audio segment."""
        if not self.utterance_buffer:
            return

        # Concatenate all audio
        audio = np.concatenate(self.utterance_buffer)

        segment = AudioSegment(
            audio=audio,
            sample_rate=self.sample_rate,
            timestamp=time.time(),
        )

        if self.on_segment:
            self.on_segment(segment)

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice stream."""
        if status:
            print(f"Audio status: {status}")

        # Convert to mono float32
        audio = indata[:, 0].astype(np.float32)
        
        # Add to buffer
        self._audio_buffer = np.concatenate([self._audio_buffer, audio])
        
        # Process complete frames (exactly 512 samples each for Silero VAD at 16kHz)
        while len(self._audio_buffer) >= self.frame_samples:
            chunk = self._audio_buffer[:self.frame_samples]
            self._audio_buffer = self._audio_buffer[self.frame_samples:]
            self._process_frame(chunk)

    def start(self):
        """Start audio capture."""
        if self._running:
            return

        self._running = True
        
        # Reset VAD state
        self.vad_model.reset_states()
        self.state = VADState.IDLE
        self.prebuffer.clear()
        self.utterance_buffer = []
        self._audio_buffer = np.array([], dtype=np.float32)

        # Start audio stream
        self._stream = sd.InputStream(
            device=self.device_index,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.frame_samples,
            dtype=np.float32,
            callback=self._audio_callback,
        )
        self._stream.start()
        print(f"[Capture] Started on device {self.device_index} @ {self.sample_rate}Hz")

    def stop(self):
        """Stop audio capture."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print("[Capture] Stopped")

    def is_speaking(self) -> bool:
        """Check if VAD currently detects speech."""
        return self.state == VADState.IN_SPEECH
