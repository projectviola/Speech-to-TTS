"""Main application entry point for STT-TTS system."""

import sys
import os
import signal
import time
import threading
from pathlib import Path

import keyboard

from .config import Config
from .devices import setup_devices, verify_device
from .capture import AudioCaptureVAD, AudioSegment
from .stt_worker import STTWorker, TranscriptResult
from .tts_worker import TTSWorker, TTSResult
from .playback import PlaybackController


class STTTTSApp:
    """
    Main application coordinating:
    - Audio capture with VAD
    - Speech-to-Text transcription
    - Text-to-Speech synthesis
    - Audio playback with queue management
    - OBS transcript output
    - Hotkey for pause/unpause
    """

    def __init__(self, config: Config):
        self.config = config
        
        # Components (initialized in setup)
        self.capture: AudioCaptureVAD = None
        self.stt: STTWorker = None
        self.tts: TTSWorker = None
        self.playback: PlaybackController = None
        
        # Pause state
        self._paused = False
        self._pause_lock = threading.Lock()
        
        self._running = False

    def _toggle_pause(self):
        """Toggle pause state."""
        with self._pause_lock:
            self._paused = not self._paused
            
            if self._paused:
                print("\n" + "=" * 40)
                print("  ⏸️  PAUSED - Press hotkey to resume")
                print("=" * 40)
                
                # Clear all queues
                if self.stt:
                    self.stt.clear_queue()
                if self.tts:
                    self.tts.clear_queue()
                if self.playback:
                    self.playback.clear_queue()
                
                # Stop capture
                if self.capture:
                    self.capture.stop()
            else:
                print("\n" + "=" * 40)
                print("  ▶️  RESUMED")
                print("=" * 40 + "\n")
                
                # Restart capture
                if self.capture:
                    self.capture.start()

    @property
    def is_paused(self) -> bool:
        with self._pause_lock:
            return self._paused

    def _on_segment(self, segment: AudioSegment):
        """Called when VAD emits a speech segment."""
        if self.is_paused:
            return
        print(f"[App] Speech segment: {len(segment.audio) / segment.sample_rate:.2f}s")
        self.stt.enqueue(segment)

    def _on_speech_start(self):
        """Called when VAD detects speech start."""
        if self.is_paused:
            return
        print("[App] Speech started")
        # Clear playback queue on new speech (but not current clip)
        if self.playback:
            self.playback.clear_queue()

    def _on_speech_end(self):
        """Called when VAD detects speech end."""
        if self.is_paused:
            return
        print("[App] Speech ended")

    def _on_transcript(self, result: TranscriptResult):
        """Called when STT produces a transcript."""
        if self.is_paused:
            return
        # Immediately send to TTS - no waiting
        self.tts.enqueue(result)

    def _on_tts_audio(self, result: TTSResult):
        """Called when TTS produces audio."""
        if self.is_paused:
            return
        # Immediately queue for playback
        self.playback.enqueue(result)

    def setup(self):
        """Initialize all components."""
        print("\n" + "=" * 60)
        print("  STT → TTS System")
        print("=" * 60)
        
        # Initialize STT first (takes longest to load)
        print("\n[Setup] Initializing STT...")
        self.stt = STTWorker(
            model_size=self.config.stt.model_size,
            device=self.config.stt.device,
            compute_type=self.config.stt.compute_type,
            on_transcript=self._on_transcript,
        )

        # Initialize TTS
        print("\n[Setup] Initializing TTS...")
        self.tts = TTSWorker(
            voice=self.config.tts.voice,
            speed=self.config.tts.speed,
            pitch_semitones=self.config.tts.pitch_semitones,
            lang="en-us",
            models_dir="models",
            on_audio=self._on_tts_audio,
        )

        # Initialize capture with VAD
        print("\n[Setup] Initializing audio capture with VAD...")
        self.capture = AudioCaptureVAD(
            device_index=self.config.audio.input_device_index,
            sample_rate=self.config.audio.input_sample_rate,
            frame_ms=self.config.vad.frame_ms,
            prebuffer_seconds=self.config.vad.prebuffer_seconds,
            tail_seconds=self.config.vad.tail_seconds,
            start_threshold=self.config.vad.start_threshold,
            end_threshold=self.config.vad.end_threshold,
            min_speech_ms=self.config.vad.min_speech_ms,
            min_silence_ms=self.config.vad.min_silence_ms,
            on_segment=self._on_segment,
            on_speech_start=self._on_speech_start,
            on_speech_end=self._on_speech_end,
        )

        # Initialize playback controller
        print("\n[Setup] Initializing playback controller...")
        self.playback = PlaybackController(
            output_device_index=self.config.audio.output_device_index,
            vad_end_timeout_seconds=self.config.behavior.vad_end_timeout_seconds,
            queue_max_items=self.config.behavior.queue_max_items,
            obs_enabled=self.config.obs.enabled,
            obs_transcript_file=self.config.obs.transcript_file,
            obs_next_transcript_delay_seconds=self.config.obs.next_transcript_delay_seconds,
            obs_clear_delay_seconds=self.config.obs.clear_delay_seconds,
            get_last_speech_end_time=lambda: self.capture.last_speech_end_time,
            is_speaking=lambda: self.capture.is_speaking(),
        )

        print("\n[Setup] All components initialized!")

    def run(self):
        """Start the application."""
        self._running = True
        
        # Start all workers
        self.stt.start()
        self.tts.start()
        self.playback.start()
        self.capture.start()
        
        # Register global hotkey
        hotkey = self.config.behavior.pause_hotkey
        try:
            keyboard.add_hotkey(hotkey, self._toggle_pause)
            print(f"[Hotkey] Registered: {hotkey}")
        except Exception as e:
            print(f"[Hotkey] Failed to register '{hotkey}': {e}")
        
        print("\n" + "=" * 60)
        print("  RUNNING - Press Ctrl+C to stop")
        print("=" * 60)
        print(f"  Mic: device {self.config.audio.input_device_index}")
        print(f"  Speaker: device {self.config.audio.output_device_index}")
        print(f"  VAD timeout: {self.config.behavior.vad_end_timeout_seconds}s")
        print(f"  Pitch: {self.config.tts.pitch_semitones:+.1f} semitones")
        print(f"  Pause hotkey: {hotkey}")
        print(f"  OBS file: {self.config.obs.transcript_file}")
        print("=" * 60 + "\n")
        
        # Keep running until interrupted
        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[App] Shutting down...")
        
        self.stop()

    def stop(self):
        """Stop the application."""
        self._running = False
        
        print("\n[App] Stopping components...")
        
        # Unhook keyboard
        try:
            keyboard.unhook_all()
        except Exception:
            pass
        
        if self.capture:
            self.capture.stop()
        if self.playback:
            self.playback.stop()
        if self.tts:
            self.tts.stop()
        if self.stt:
            self.stt.stop()
        
        print("[App] Stopped")


def main():
    """Main entry point."""
    config_path = "config.json"
    
    # Load or create config
    config = Config.load(config_path)
    
    # Check if device setup needed
    if config.needs_device_setup():
        print("[Setup] Audio devices not configured.")
        mic_index, speaker_index = setup_devices()
        config.audio.input_device_index = mic_index
        config.audio.output_device_index = speaker_index
        config.save(config_path)
        print(f"\n[Setup] Configuration saved to {config_path}")
    else:
        # Verify devices still exist
        if not verify_device(config.audio.input_device_index, is_input=True):
            print(f"[Warning] Input device {config.audio.input_device_index} no longer valid")
            mic_index, speaker_index = setup_devices()
            config.audio.input_device_index = mic_index
            config.audio.output_device_index = speaker_index
            config.save(config_path)
        elif not verify_device(config.audio.output_device_index, is_input=False):
            print(f"[Warning] Output device {config.audio.output_device_index} no longer valid")
            mic_index, speaker_index = setup_devices()
            config.audio.input_device_index = mic_index
            config.audio.output_device_index = speaker_index
            config.save(config_path)
        else:
            print(f"[Setup] Using saved devices from {config_path}")
    
    # Handle command line args
    if len(sys.argv) > 1:
        if sys.argv[1] in ("--setup", "-s"):
            mic_index, speaker_index = setup_devices()
            config.audio.input_device_index = mic_index
            config.audio.output_device_index = speaker_index
            config.save(config_path)
            print(f"\n[Setup] Configuration saved to {config_path}")
            return
        elif sys.argv[1] in ("--help", "-h"):
            print("STT-TTS Application")
            print()
            print("Usage: python -m app.main [options]")
            print()
            print("Options:")
            print("  --setup, -s    Re-configure audio devices")
            print("  --help, -h     Show this help")
            return
    
    # Create and run app
    app = STTTTSApp(config)
    
    # Handle signals
    def signal_handler(sig, frame):
        app.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup and run
    app.setup()
    app.run()


if __name__ == "__main__":
    main()
