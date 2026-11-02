import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional
import torch

CONFIG_DIR = os.path.expanduser("~/.config/giga_dictate")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")


def get_default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class AppConfig:
    hotkey: str = "right_cmd"
    hotkey_mode: str = "toggle"  # "toggle" or "push_to_talk"
    model_name: str = "v3_e2e_ctc"
    device: str = field(default_factory=get_default_device)
    sound_feedback: bool = True
    restore_clipboard: bool = False
    conceal_clipboard: bool = True  # Protect pasteboard from clipboard history managers and iCloud sync
    add_trailing_space: bool = True
    paste_delay: float = 0.03
    min_duration: float = 0.25
    audio_device: Optional[int] = None
    show_settings_on_launch: bool = False

    @classmethod
    def load(cls) -> "AppConfig":
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
            except Exception as e:
                print(f"[Config] Error loading config from {CONFIG_PATH}: {e}, using defaults.")
        config = cls()
        config.save()
        return config

    def save(self) -> None:
        try:
            os.makedirs(CONFIG_DIR, mode=0o700, exist_ok=True)
            try:
                os.chmod(CONFIG_DIR, 0o700)
            except OSError:
                pass

            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            fd = os.open(CONFIG_PATH, flags, 0o600)
            with open(fd, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, ensure_ascii=False, indent=2)
            try:
                os.chmod(CONFIG_PATH, 0o600)
            except OSError:
                pass
        except Exception as e:
            print(f"[Config] Error saving config: {e}")
