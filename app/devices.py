"""Audio device selection utilities."""

import sounddevice as sd
from typing import Optional, Tuple


def list_input_devices() -> list[Tuple[int, str]]:
    """List available input (microphone) devices."""
    devices = sd.query_devices()
    inputs = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            inputs.append((i, dev["name"]))
    return inputs


def list_output_devices() -> list[Tuple[int, str]]:
    """List available output (speaker) devices."""
    devices = sd.query_devices()
    outputs = []
    for i, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            outputs.append((i, dev["name"]))
    return outputs


def select_device(devices: list[Tuple[int, str]], prompt: str) -> int:
    """Interactive device selection."""
    print(f"\n{prompt}")
    print("-" * 50)
    for idx, (dev_id, name) in enumerate(devices):
        print(f"  [{idx}] {name} (device {dev_id})")
    print()

    while True:
        try:
            choice = input("Enter selection number: ").strip()
            idx = int(choice)
            if 0 <= idx < len(devices):
                return devices[idx][0]  # Return actual device ID
            print(f"Please enter a number between 0 and {len(devices) - 1}")
        except ValueError:
            print("Please enter a valid number")


def setup_devices() -> Tuple[int, int]:
    """Interactive setup for input and output devices."""
    print("\n" + "=" * 50)
    print("  AUDIO DEVICE SETUP")
    print("=" * 50)

    inputs = list_input_devices()
    if not inputs:
        raise RuntimeError("No input devices found!")
    mic_index = select_device(inputs, "Select MICROPHONE (input):")

    outputs = list_output_devices()
    if not outputs:
        raise RuntimeError("No output devices found!")
    speaker_index = select_device(outputs, "Select SPEAKER (output):")

    print(f"\n✓ Microphone: device {mic_index}")
    print(f"✓ Speaker: device {speaker_index}")

    return mic_index, speaker_index


def verify_device(device_index: int, is_input: bool) -> bool:
    """Verify a device index is still valid."""
    try:
        devices = sd.query_devices()
        if device_index >= len(devices):
            return False
        dev = devices[device_index]
        if is_input:
            return dev["max_input_channels"] > 0
        else:
            return dev["max_output_channels"] > 0
    except Exception:
        return False
