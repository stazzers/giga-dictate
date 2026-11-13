"""Platform-independent session state and short-lived recovery text."""
import time
from dataclasses import dataclass, field
from enum import Enum


class DictationState(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    INSERTING = "inserting"
    ERROR = "error"


@dataclass
class DictationSession:
    state: DictationState = DictationState.IDLE
    generation: int = 0
    error: str = ""
    started_at: float = 0.0

    @property
    def busy(self) -> bool:
        return self.state not in (DictationState.IDLE, DictationState.ERROR)

    def begin(self) -> bool:
        if self.busy:
            return False
        self.generation += 1
        self.state = DictationState.RECORDING
        self.error = ""
        self.started_at = time.monotonic()
        return True

    def accepts(self, generation: int) -> bool:
        return generation == self.generation

    def finish(self, generation: int, error: str = "") -> bool:
        if not self.accepts(generation) or not self.busy:
            return False
        self.error = error
        self.state = DictationState.ERROR if error else DictationState.IDLE
        return True

    def invalidate(self):
        self.generation += 1
        self.state = DictationState.IDLE


@dataclass
class RecoveryText:
    ttl: float = 300.0
    _text: str = field(default="", repr=False)
    _expires_at: float = 0.0

    def remember(self, text: str):
        if text:
            self._text = text
            self._expires_at = time.monotonic() + self.ttl

    def get(self) -> str:
        if time.monotonic() >= self._expires_at:
            self.clear()
        return self._text

    def clear(self):
        self._text = ""
        self._expires_at = 0.0
