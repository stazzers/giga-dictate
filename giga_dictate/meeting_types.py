"""Meeting contracts and editable results; deliberately independent of AppKit."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np


class MeetingState(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    RECORDING = "recording"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass(frozen=True)
class MeetingOptions:
    mode: str = "online"
    device: int | None = None
    audio_path: str | None = None

    def validate(self):
        if self.mode not in ("online", "room", "enrollment"):
            raise ValueError("Неизвестный режим встречи")
        if self.audio_path and Path(self.audio_path).suffix.lower() not in (".wav", ".m4a"):
            raise ValueError("Выберите файл .wav или .m4a")


@dataclass(frozen=True)
class AudioFrame:
    source: str
    sequence: int
    start: float  # seconds on the shared monotonic clock
    sample_rate: int
    pcm: np.ndarray = field(repr=False, compare=False)

    @property
    def duration(self):
        return len(self.pcm) / self.sample_rate


@dataclass
class SpeechSegment:
    source: str
    start: float
    end: float
    pcm: np.ndarray = field(repr=False)


@dataclass
class Speaker:
    id: str
    name: str


@dataclass
class TranscriptTurn:
    id: int
    start: float
    end: float
    source: str
    speaker_id: str
    text: str = field(repr=False)
    attribution: str = "automatic"
    manual_speaker_id: str | None = None

    @property
    def effective_speaker_id(self):
        return self.manual_speaker_id or self.speaker_id


def timestamp(seconds):
    total = max(0, int(seconds))
    return f"{total // 3600:02}:{total // 60 % 60:02}:{total % 60:02}"


@dataclass
class MeetingResult:
    session_id: int
    started_at: datetime
    duration: float = 0.0
    turns: list[TranscriptTurn] = field(default_factory=list, repr=False)
    speakers: dict[str, Speaker] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    audio_path: str | None = None
    enrollment_complete: bool = False

    def ordered_turns(self):
        return sorted(self.turns, key=lambda turn: (turn.start, turn.id))

    def rename_speaker(self, speaker_id, name):
        name = name.strip().replace("\n", " ").replace("\r", " ")[:100]
        if not name:
            raise ValueError("Введите имя спикера")
        self.speakers[speaker_id].name = name

    def assign_speaker(self, turn_id, speaker_id):
        if speaker_id not in self.speakers:
            raise ValueError("Неизвестный спикер")
        turn = next(turn for turn in self.turns if turn.id == turn_id)
        turn.manual_speaker_id = speaker_id

    def render(self, markdown=True):
        title = f"Стенограмма встречи — {self.started_at:%d.%m.%Y (%H:%M)}"
        lines = [("# " if markdown else "") + title,
                 f"Длительность: {timestamp(self.duration)}", ""]
        for turn in self.ordered_turns():
            speaker = self.speakers.get(turn.effective_speaker_id)
            name = speaker.name if speaker else "Спикер не определён"
            lines.append(f"[{timestamp(turn.start)}] {name}: {turn.text}")
        if self.errors:
            lines.extend(["", "Предупреждения:", *self.errors])
        return "\n".join(lines) + "\n"

    def clear(self):
        self.turns.clear()
        self.speakers.clear()
        self.errors.clear()
