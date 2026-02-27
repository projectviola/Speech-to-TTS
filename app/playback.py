"""Playback controller with queue management and OBS transcript output."""

import threading
import time
from queue import Queue, Empty
from typing import Optional, Callable
from collections import deque
from dataclasses import dataclass
import numpy as np
import sounddevice as sd

from .tts_worker import TTSResult


class PlaybackController:
    """
    Controls audio playback with smart queue management.
    
    Features:
    - Waits for silence timeout before starting playback
    - Continuous playback once started
    - Clears queue (but not current clip) on new speech
    - OBS transcript file integration
    """

    def __init__(
        self,
        output_device_index: int,
        vad_end_timeout_seconds: float = 1.0,
        queue_max_items: int = 10,
        obs_enabled: bool = True,
        obs_transcript_file: str = "obs_transcript.txt",
        obs_next_transcript_delay_seconds: float = 0.5,
        obs_clear_delay_seconds: float = 2.0,
        get_last_speech_end_time: Optional[Callable[[], float]] = None,
        is_speaking: Optional[Callable[[], bool]] = None,
    ):
        self.output_device_index = output_device_index
        self.vad_end_timeout_seconds = vad_end_timeout_seconds
        self.queue_max_items = queue_max_items
        
        # OBS settings
        self.obs_enabled = obs_enabled
        self.obs_transcript_file = obs_transcript_file
        self.obs_next_transcript_delay_seconds = obs_next_transcript_delay_seconds
        self.obs_clear_delay_seconds = obs_clear_delay_seconds

        # Callbacks to check VAD state
        self.get_last_speech_end_time = get_last_speech_end_time
        self.is_speaking = is_speaking

        # Playback queue (thread-safe deque with lock)
        self._queue: deque[TTSResult] = deque()
        self._queue_lock = threading.Lock()

        # State
        self._is_playing = False
        self._current_clip: Optional[TTSResult] = None
        
        # OBS state
        self._obs_clear_timer: Optional[threading.Timer] = None
        self._obs_update_timer: Optional[threading.Timer] = None
        self._last_obs_write_time = 0.0

        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _write_obs_transcript(self, text: str):
        """Write transcript to OBS file."""
        if not self.obs_enabled:
            return
        try:
            with open(self.obs_transcript_file, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"[Playback] OBS write error: {e}")

    def _schedule_obs_clear(self):
        """Schedule clearing the OBS transcript file."""
        if not self.obs_enabled:
            return
        
        # Cancel any existing clear timer
        if self._obs_clear_timer:
            self._obs_clear_timer.cancel()
        
        def do_clear():
            # Only clear if we're still not playing
            if not self._is_playing:
                self._write_obs_transcript("")
                print("[Playback] OBS transcript cleared")
        
        self._obs_clear_timer = threading.Timer(self.obs_clear_delay_seconds, do_clear)
        self._obs_clear_timer.start()

    def _cancel_obs_clear(self):
        """Cancel any pending OBS clear."""
        if self._obs_clear_timer:
            self._obs_clear_timer.cancel()
            self._obs_clear_timer = None

    def _play_audio(self, audio: np.ndarray, sample_rate: int):
        """Play audio through the output device."""
        try:
            sd.play(audio, sample_rate, device=self.output_device_index)
            sd.wait()  # Block until done
        except Exception as e:
            print(f"[Playback] Error playing audio: {e}")

    def _worker_loop(self):
        """Main playback controller loop."""
        was_playing = False
        
        while self._running:
            # Check if we should start/continue playback
            now = time.time()
            
            # Get last speech end time
            last_speech_end = 0.0
            if self.get_last_speech_end_time:
                last_speech_end = self.get_last_speech_end_time()
            
            # Check if user is currently speaking
            user_speaking = False
            if self.is_speaking:
                user_speaking = self.is_speaking()
            
            # If user started speaking, clear the queue (but not current clip)
            if user_speaking and not self._is_playing:
                self.clear_queue()
            
            # Check if we can start playback
            silence_elapsed = now - last_speech_end if last_speech_end > 0 else float('inf')
            can_start = (
                silence_elapsed >= self.vad_end_timeout_seconds
                and not user_speaking
            )

            # Get next item from queue if not playing and can start
            if not self._is_playing and can_start:
                with self._queue_lock:
                    if self._queue:
                        self._current_clip = self._queue.popleft()
                        self._is_playing = True
                        self._cancel_obs_clear()
                        
                        # Update OBS transcript
                        if self.obs_enabled:
                            # If coming from idle, write immediately
                            # If continuous playback, delay slightly
                            if was_playing:
                                # Delay the update
                                if self._obs_update_timer:
                                    self._obs_update_timer.cancel()
                                
                                def delayed_write(text):
                                    if self._is_playing:  # Still playing same type of clip
                                        self._write_obs_transcript(text)
                                
                                self._obs_update_timer = threading.Timer(
                                    self.obs_next_transcript_delay_seconds,
                                    delayed_write,
                                    args=[self._current_clip.transcript]
                                )
                                self._obs_update_timer.start()
                            else:
                                self._write_obs_transcript(self._current_clip.transcript)

            # Play current clip if we have one
            if self._is_playing and self._current_clip:
                was_playing = True
                print(f"[Playback] Playing: \"{self._current_clip.transcript[:50]}...\"" 
                      if len(self._current_clip.transcript) > 50 
                      else f"[Playback] Playing: \"{self._current_clip.transcript}\"")
                
                # Check if user starts speaking during playback
                # If so, clear queue but continue current clip
                if self.is_speaking and self.is_speaking():
                    self.clear_queue()
                
                # Play the audio (blocking)
                self._play_audio(self._current_clip.audio, self._current_clip.sample_rate)
                
                # Done with this clip
                self._current_clip = None
                
                # Check if more in queue
                with self._queue_lock:
                    if not self._queue:
                        self._is_playing = False
                        was_playing = False
                        # Schedule OBS clear
                        self._schedule_obs_clear()
            else:
                was_playing = False
                time.sleep(0.05)  # Small sleep when idle

    def enqueue(self, result: TTSResult):
        """Add a TTS result to the playback queue."""
        with self._queue_lock:
            # Check queue cap
            if len(self._queue) >= self.queue_max_items:
                # Drop oldest (never drop currently playing)
                dropped = self._queue.popleft()
                print(f"[Playback] Queue full, dropped oldest: \"{dropped.transcript[:30]}...\"")
            
            self._queue.append(result)
            print(f"[Playback] Queued ({len(self._queue)} items): \"{result.transcript[:30]}...\"")

    def clear_queue(self):
        """Clear all pending items (does not affect currently playing clip)."""
        with self._queue_lock:
            if self._queue:
                count = len(self._queue)
                self._queue.clear()
                print(f"[Playback] Cleared {count} queued items (new speech detected)")

    def start(self):
        """Start the playback controller."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        
        # Initialize OBS file
        if self.obs_enabled:
            self._write_obs_transcript("")
        
        print(f"[Playback] Controller started (output device {self.output_device_index})")

    def stop(self):
        """Stop the playback controller."""
        self._running = False
        
        # Cancel timers
        if self._obs_clear_timer:
            self._obs_clear_timer.cancel()
        if self._obs_update_timer:
            self._obs_update_timer.cancel()
        
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        # Clear OBS file
        if self.obs_enabled:
            self._write_obs_transcript("")
        
        print("[Playback] Controller stopped")

    @property
    def is_playing(self) -> bool:
        """Check if currently playing audio."""
        return self._is_playing

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        with self._queue_lock:
            return len(self._queue)
