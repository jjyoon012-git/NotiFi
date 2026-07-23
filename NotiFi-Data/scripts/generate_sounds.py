from __future__ import annotations

import math
import struct
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "sounds"
SAMPLE_RATE = 44100


def tone(frequency: float, duration: float, volume: float = 0.45) -> bytes:
    count = int(SAMPLE_RATE * duration)
    attack = max(1, int(SAMPLE_RATE * 0.01))
    release = max(1, int(SAMPLE_RATE * 0.03))
    samples = []
    for index in range(count):
        envelope = 1.0
        if index < attack:
            envelope = index / attack
        if index >= count - release:
            envelope = min(envelope, (count - index - 1) / release)
        value = int(32767 * volume * envelope * math.sin(2 * math.pi * frequency * index / SAMPLE_RATE))
        samples.append(struct.pack("<h", value))
    return b"".join(samples)


def silence(duration: float) -> bytes:
    return b"\x00\x00" * int(SAMPLE_RATE * duration)


def write_wav(name: str, chunks: list[bytes]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(b"".join(chunks))
    print(f"[OK] {path}")


def main() -> None:
    write_wav(
        "sound1_trial_start.wav",
        [tone(880, 0.16), silence(0.08), tone(1175, 0.26)],
    )
    write_wav(
        "sound2_trial_end.wav",
        [tone(660, 0.18), silence(0.07), tone(440, 0.30)],
    )
    write_wav(
        "sound3_set_done.wav",
        [
            tone(523, 0.14),
            silence(0.05),
            tone(659, 0.14),
            silence(0.05),
            tone(784, 0.14),
            silence(0.05),
            tone(1047, 0.45),
        ],
    )
    write_wav("action_cue.wav", [tone(1568, 0.20)])
    write_wav("error.wav", [tone(220, 0.25), silence(0.10), tone(220, 0.45)])


if __name__ == "__main__":
    main()
