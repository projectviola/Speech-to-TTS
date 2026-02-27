"""Configuration management for STT-TTS application."""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


@dataclass
class AudioConfig:
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None
    input_sample_rate: int = 16000
    channels: int = 1


@dataclass
class VADConfig:
    model: str = "silero"
    start_threshold: float = 0.5
    end_threshold: float = 0.35
    min_speech_ms: int = 250
    min_silence_ms: int = 300
    prebuffer_seconds: float = 2.0
    tail_seconds: float = 0.15
    frame_ms: int = 32  # Exactly 32ms = 512 samples at 16kHz (required by Silero)


@dataclass
class STTConfig:
    model_size: str = "tiny"
    device: str = "cuda"
    compute_type: str = "float16"


@dataclass
class TTSConfig:
    engine: str = "kokoro_onnx"
    voice: str = "af_heart"  # Cute anime-like voice
    speed: float = 1.0
    pitch_semitones: float = 0.0  # Pitch shift in semitones (e.g., +2 = higher, -2 = lower)
    require_cuda: bool = True


@dataclass
class BehaviorConfig:
    vad_end_timeout_seconds: float = 1.0
    queue_max_items: int = 10
    pause_hotkey: str = "ctrl+shift+p"  # Hotkey to pause/unpause


@dataclass
class OBSConfig:
    enabled: bool = True
    transcript_file: str = "obs_transcript.txt"
    next_transcript_delay_seconds: float = 0.5
    clear_delay_seconds: float = 2.0


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    obs: OBSConfig = field(default_factory=OBSConfig)

    @classmethod
    def load(cls, path: str = "config.json") -> "Config":
        """Load config from JSON file, or return defaults if not found."""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                audio=AudioConfig(**data.get("audio", {})),
                vad=VADConfig(**data.get("vad", {})),
                stt=STTConfig(**data.get("stt", {})),
                tts=TTSConfig(**data.get("tts", {})),
                behavior=BehaviorConfig(**data.get("behavior", {})),
                obs=OBSConfig(**data.get("obs", {})),
            )
        return cls()

    def save(self, path: str = "config.json"):
        """Save config to JSON file."""
        data = {
            "audio": asdict(self.audio),
            "vad": asdict(self.vad),
            "stt": asdict(self.stt),
            "tts": asdict(self.tts),
            "behavior": asdict(self.behavior),
            "obs": asdict(self.obs),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def needs_device_setup(self) -> bool:
        """Check if audio devices need to be configured."""
        return (
            self.audio.input_device_index is None
            or self.audio.output_device_index is None
        )
