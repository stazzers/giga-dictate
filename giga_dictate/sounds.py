import os
import threading
from typing import Optional
from AppKit import NSSound


class SoundManager:
    """Plays system sounds on macOS for auditory dictation feedback."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._sounds = {}
        # Preload common sounds
        for name in ("Tink", "Pop", "Hero", "Ping", "Basso", "Funk"):
            sound = NSSound.soundNamed_(name)
            if sound is not None:
                self._sounds[name] = sound

    def play(self, sound_name: str) -> None:
        if not self.enabled:
            return

        def _play():
            sound = self._sounds.get(sound_name) or NSSound.soundNamed_(sound_name)
            if sound is not None:
                # Stop if currently playing and restart
                sound.stop()
                sound.play()

        threading.Thread(target=_play, daemon=True).start()

    def play_start(self) -> None:
        """Recording started."""
        self.play("Tink")

    def play_stop(self) -> None:
        """Recording stopped, transcribing..."""
        self.play("Pop")

    def play_success(self) -> None:
        """Transcription pasted successfully."""
        self.play("Hero")

    def play_cancel(self) -> None:
        """Recording cancelled or too short."""
        self.play("Basso")
