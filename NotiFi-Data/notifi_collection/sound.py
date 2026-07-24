from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOUND_DIR = ROOT / "assets" / "sounds"
SOUND_FILES = {
    "trial_start": SOUND_DIR / "sound1_trial_start.wav",
    "trial_end": SOUND_DIR / "sound2_trial_end.wav",
    "set_done": SOUND_DIR / "sound3_set_done.wav",
    "action_cue": SOUND_DIR / "action_cue.wav",
    "error": SOUND_DIR / "error.wav",
}


def _play_windows(path: Path) -> None:
    import winsound

    winsound.PlaySound(str(path), winsound.SND_FILENAME)


def _play_command(command: str, path: Path) -> bool:
    executable = shutil.which(command)
    if not executable:
        return False
    subprocess.run(
        [executable, str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return True


def play_sound(name: str, enabled: bool = True) -> None:
    if not enabled:
        return
    path = SOUND_FILES.get(name)
    if path is None or not path.exists():
        print(f"[WARN] sound file missing: {path}", file=sys.stderr)
        return
    try:
        if os.name == "nt":
            _play_windows(path)
            return
        if sys.platform == "darwin" and _play_command("afplay", path):
            return
        if _play_command("paplay", path) or _play_command("aplay", path):
            return
        print(f"[WARN] no WAV player found for {path.name}", file=sys.stderr)
    except Exception as exc:
        print(f"[WARN] sound playback failed: {exc}", file=sys.stderr)
        time.sleep(0.05)


def play_error_alarm(enabled: bool = True) -> None:
    """Play a collection-failure alarm: error tone followed by three beeps."""
    if not enabled:
        return
    play_sound("error", True)
    for _ in range(3):
        time.sleep(0.12)
        play_sound("action_cue", True)
