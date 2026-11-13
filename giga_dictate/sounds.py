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

        sound = self._sounds.get(sound_name) or NSSound.soundNamed_(sound_name)
        if sound is not None:
            # NSSound.play() is natively asynchronous and non-blocking in Cocoa/CoreAudio
            sound.stop()
            sound.play()

    def play_start(self) -> None:
        """Recording started."""
        self.play("Tink")

    def play_stop(self) -> None:
        """Recording stopped, transcribing..."""
        self.play("Pop")

    def play_success(self) -> None:
        """Paste events sent; the target app does not acknowledge delivery."""
        self.play("Hero")

    def play_cancel(self) -> None:
        """Recording cancelled or too short."""
        self.play("Basso")
