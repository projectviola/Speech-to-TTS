# STT → TTS System (Windows, CUDA 13.1)

Real-time speech-to-text-to-speech system with VAD-based segmentation and OBS integration.

## Features

- **Always-listening** microphone capture
- **Silero VAD** for accurate speech detection (avoids Whisper hallucinations)
- **faster-whisper** (tiny model) for fast GPU transcription
- **kokoro-onnx** for anime-style TTS synthesis
- **Smart playback queue** with configurable silence timeout
- **OBS integration** via text file for live transcript display

## Requirements

- Windows 10/11
- NVIDIA GPU
- CUDA 13.1 installed
- cuDNN 9.1.0 installed
- ~4GB disk space for models

## Installation

1. Double-click `install.bat`
2. Wait for dependencies to download and install (~10-15 minutes)
3. Done!

## Usage

1. Double-click `run.bat`
2. On first run, select your microphone and speaker
3. Start talking!

### Command Line Options

```batch
run.bat --setup    # Re-configure audio devices
run.bat --help     # Show help
```

## Configuration

Edit `config.json` to customize:

```json
{
  "audio": {
    "input_device_index": null,      // Set by setup, or manually
    "output_device_index": null,
    "input_sample_rate": 16000,
    "frame_ms": 30,
    "channels": 1
  },
  "vad": {
    "start_threshold": 0.5,          // Higher = less sensitive
    "end_threshold": 0.35,           // Lower = waits longer for silence
    "min_speech_ms": 250,            // Minimum speech duration
    "min_silence_ms": 300,           // Silence before speech end
    "prebuffer_seconds": 2.0         // Audio kept before speech start
  },
  "stt": {
    "model_size": "tiny",            // tiny, base, small, medium, large
    "device": "cuda",
    "compute_type": "float16"
  },
  "tts": {
    "voice": "af_heart",             // Kokoro voice name
    "speed": 1.0
  },
  "behavior": {
    "vad_end_timeout_seconds": 1.0,  // Wait after speech ends
    "queue_max_items": 10
  },
  "obs": {
    "enabled": true,
    "transcript_file": "obs_transcript.txt",
    "next_transcript_delay_seconds": 0.5,
    "clear_delay_seconds": 2.0
  }
}
```

## OBS Setup

1. Add a "Text (GDI+)" source
2. Check "Read from file"
3. Browse to `obs_transcript.txt` in this folder
4. Customize font, color, etc.

## Architecture

```
┌─────────────────┐
│   Microphone    │
└────────┬────────┘
         │ audio frames
         ▼
┌─────────────────┐
│  Capture + VAD  │ (Silero VAD)
│   Thread        │
└────────┬────────┘
         │ speech segments
         ▼
┌─────────────────┐
│   STT Worker    │ (faster-whisper GPU)
│   Thread        │
└────────┬────────┘
         │ transcripts
         ▼
┌─────────────────┐
│   TTS Worker    │ (kokoro-onnx GPU)
│   Thread        │
└────────┬────────┘
         │ audio clips
         ▼
┌─────────────────┐
│    Playback     │──────► Speaker
│   Controller    │──────► obs_transcript.txt
└─────────────────┘
```

## Behavior

- TTS synthesis starts **immediately** after STT finishes (no waiting)
- Playback only starts after **1 second of silence** (configurable)
- If you start speaking while audio is playing:
  - Current clip continues
  - Queued clips are cleared
- OBS transcript shows what's currently playing

## Troubleshooting

### No audio devices found
- Make sure your microphone/speakers are connected
- Run `run.bat --setup` to re-select devices

### CUDA errors
- Verify CUDA 13.1 is installed: `nvcc --version`
- Verify cuDNN 9.1.0 is installed
- Check GPU drivers are up to date

### VAD too sensitive / not sensitive enough
- Adjust `vad.start_threshold` (higher = less sensitive)
- Adjust `vad.end_threshold` (lower = requires more silence)

### Whisper hallucinations
- The 2-second prebuffer should help
- Try adjusting `vad.min_speech_ms` higher

## License

MIT
